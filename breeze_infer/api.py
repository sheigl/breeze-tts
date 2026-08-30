"""Thin streaming API over the PyTorch Breeze inference runtime."""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_CONFIG = REPO_ROOT / "configs" / "fast.json"
DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
OPTIONAL_AUDIO_FILE = File(None)


@dataclass(frozen=True)
class ApiSettings:
    model: Path
    fast_all: bool | None
    fast_text_encoder: bool
    fast_backbone_prefill: bool
    fast_backbone_decode: bool
    fast_depth_decoder: bool
    fast_codec: bool
    compile_xpu: bool


_settings: ApiSettings | None = None
_request_lock = threading.Lock()
# FIFO-fair lock that serializes inference requests (one at a time) on the
# single XPU/CUDA card. Requests wait their turn instead of failing with 409.
_gpu_lock = asyncio.Lock()


def _pcm16(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2", copy=False).tobytes()


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "reference.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(
        prefix="breeze_ref_", suffix=suffix, delete=False
    ) as temporary:
        path = Path(temporary.name)
        try:
            payload = await upload.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Reference audio is empty.")
            temporary.write(payload)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return path


def _load_app(app: FastAPI, settings: ApiSettings) -> None:
    tokenizer, model, audio_tokenizer = load_runtime(
        settings.model,
        device=resolve_device(),
        attn_implementation="eager",
    )
    update_generation_config_for_breeze(model)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=settings.fast_all,
        fast_text_encoder=settings.fast_text_encoder,
        fast_backbone_prefill=settings.fast_backbone_prefill,
        fast_backbone_decode=settings.fast_backbone_decode,
        fast_depth_decoder=settings.fast_depth_decoder,
        fast_codec=settings.fast_codec,
        compile_xpu=settings.compile_xpu,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, config, tokenizer=tokenizer
    )
    if runtime.fast_enabled:
        profile = load_warmup_profile(FAST_CONFIG)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms", flush=True)

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.audio_tokenizer = audio_tokenizer
    app.state.runtime = runtime


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if _settings is None:
        raise RuntimeError("API settings are not initialized")
    _load_app(app, _settings)
    yield


app = FastAPI(title="Breeze TTS API", lifespan=_lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if not hasattr(app.state, "runtime"):
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok", "sample_rate": app.state.runtime.sample_rate})


@app.post("/v1/audio/speech")
async def speech(
    text: str = Form(...),
    instruction: str = Form("Speak clearly and naturally."),
    cfg_scale: float = Form(DEFAULT_CFG_SCALE),
    ref_audio: UploadFile | None = OPTIONAL_AUDIO_FILE,
    ref_text: str = Form(""),
    seed: int = Form(42),
) -> StreamingResponse:
    reference_path: Path | None = None
    try:
        if not np.isfinite(cfg_scale) or cfg_scale <= 0:
            raise HTTPException(
                status_code=400, detail="cfg_scale must be greater than 0."
            )
        ref_text = ref_text.strip()
        has_reference = ref_audio is not None and bool(ref_audio.filename)
        if has_reference != bool(ref_text):
            raise HTTPException(
                status_code=400,
                detail="ref_audio and ref_text must be provided together or both omitted.",
            )
        if has_reference:
            assert ref_audio is not None
            reference_path = await _save_upload(ref_audio)

        request_id = f"api-{uuid.uuid4().hex}"
        request_payload = {
            "id": request_id,
            "text": text,
            "instruction": instruction,
            "speaker": "S0",
        }
        template_name = "tts_instruction"
        if reference_path is not None:
            request_payload["ref_audio_path"] = str(reference_path)
            request_payload["ref_text"] = ref_text
            template_name = "ref_edit_tata"

        set_all_seeds(seed)
        inputs = prepare_inputs(
            app.state.tokenizer,
            app.state.audio_tokenizer,
            app.state.model,
            [request_payload],
            get_template(template_name),
            guidance_scale=cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
    except HTTPException:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
        raise
    except Exception:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
        raise

    async def body() -> AsyncIterator[bytes]:
        # FIFO queue: wait for the GPU slot instead of failing with 409.
        cancel_event = threading.Event()
        out_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def producer() -> None:
            try:
                for chunk in app.state.runtime.iter_audio_chunks(
                    inputs, request_id=request_id, cancel_event=cancel_event
                ):
                    pcm = _pcm16(chunk.audio)
                    if pcm:
                        loop.call_soon_threadsafe(out_queue.put_nowait, pcm)
            # Swallow generation errors (client may have already received the
            # partial audio stream); the finally below guarantees end-of-stream.
            except Exception:  # noqa: BLE001
                loop.call_soon_threadsafe(out_queue.put_nowait, None)
            finally:
                # Sentinel: signals end of stream to the async consumer.
                loop.call_soon_threadsafe(out_queue.put_nowait, None)

        try:
            async with _gpu_lock:
                thread = threading.Thread(target=producer, daemon=True)
                thread.start()
                try:
                    while True:
                        try:
                            pcm = await asyncio.wait_for(out_queue.get(), timeout=0.5)
                        except TimeoutError:
                            # No chunk yet; loop back so client disconnects are
                            # noticed promptly (Starlette closes this async
                            # generator on disconnect, raising GeneratorExit).
                            continue
                        if pcm is None:
                            break
                        yield pcm
                finally:
                    # Fires both on normal completion and on client disconnect
                    # (Starlette aclose()s the generator), releasing the GPU
                    # slot promptly for the next queued request.
                    cancel_event.set()
                    thread.join(timeout=2)
        finally:
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)

    return StreamingResponse(
        body(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(app.state.runtime.sample_rate),
            "X-Sample-Format": "s16le",
            "Cache-Control": "no-store",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Breeze TTS 2 streaming inference"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-codec", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile-xpu",
        action="store_true",
        help="Use torch.compile for the backbone decode on XPU (CUDA uses "
        "graph capture instead; ignored otherwise)",
    )
    args = parser.parse_args()

    global _settings
    _settings = ApiSettings(
        model=args.model,
        fast_all=args.fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
        compile_xpu=args.compile_xpu,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
