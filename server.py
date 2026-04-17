"""
Parakeet v3 streaming ASR server for Sales Matrix CRM.

Endpoints:
- GET  /health                          → liveness + model status
- GET  /metrics                         → CPU/RAM + session stats (JSON)
- POST /v1/audio/transcriptions         → OpenAI-compatible batch (multipart file)
- WS   /ws?token=...                    → realtime streaming (PCM16 16kHz mono)

Auth: simple shared token via query param (PARAKEET_API_TOKEN env var).
      Empty PARAKEET_API_TOKEN = no auth (dev only).
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psutil

from transcriber import (
    MIN_CHUNK_MS,
    MAX_CHUNK_MS,
    SAMPLE_RATE,
    ParakeetTranscriber,
    StreamSession,
    TranscriptionResult,
)


logging.basicConfig(
    level=os.environ.get("PARAKEET_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("parakeet")


API_TOKEN = os.environ.get("PARAKEET_API_TOKEN", "").strip()
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "PARAKEET_ALLOWED_ORIGINS",
        ",".join([
            "https://www.salesenhancer.it",
            "https://salesenhancer.it",
            # Local dev servers (Vite defaults + project-specific 8080)
            "http://localhost:3000",
            "http://localhost:4173",
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:8080",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ]),
    ).split(",") if o.strip()
]
MAX_SESSION_MINUTES = int(os.environ.get("PARAKEET_MAX_SESSION_MINUTES", "10"))
MAX_FRAME_BYTES = int(os.environ.get("PARAKEET_MAX_FRAME_BYTES", "8192"))  # 256ms max per frame
# Rolling overlap between consecutive chunks to avoid cutting words mid-flow.
# Too large => Parakeet re-transcribes the same segment differently on each pass
# => duplicated / mutated words. Too small => words may be cut at boundaries.
# 100ms is a good compromise for speech.
OVERLAP_MS = int(os.environ.get("PARAKEET_OVERLAP_MS", "100"))

transcriber = ParakeetTranscriber()

# Lightweight in-memory session counter for /metrics
_active_sessions = 0
_total_sessions = 0
_total_transcription_s = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model in a background thread so /health responds 200 immediately.
    # Easypanel/k8s health probes can mark the container healthy during the
    # ~20s cold start (model download + ONNX init).
    import threading

    def _load():
        try:
            transcriber.load()
            logger.info("Startup complete. Ready to accept connections.")
        except Exception:
            logger.exception("Background model load failed")

    logger.info("Starting background model loader...")
    threading.Thread(target=_load, daemon=True, name="parakeet-model-loader").start()
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Parakeet v3 Streaming ASR — Sales Matrix",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _check_token(token: Optional[str]) -> None:
    if API_TOKEN and token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


# ==================================================================
# Health & metrics
# ==================================================================


@app.get("/health")
def health():
    # Always 200 when the uvicorn process is alive.
    # model_loaded=false during the first ~20s of cold start.
    # Easypanel uses this for liveness; use /ready for readiness.
    ready = transcriber.is_ready()
    return JSONResponse(
        {
            "status": "ready" if ready else "starting",
            "model_loaded": ready,
            "model": "parakeet-tdt-0.6b-v3",
            "version": "0.1.0",
        },
        status_code=200,
    )


@app.get("/ready")
def ready():
    """Readiness probe: 200 only when model is fully loaded and can serve traffic."""
    if transcriber.is_ready():
        return JSONResponse({"ready": True}, status_code=200)
    return JSONResponse({"ready": False}, status_code=503)


@app.get("/metrics")
def metrics():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    return {
        "cpu_percent": cpu,
        "ram_percent": mem.percent,
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "active_sessions": _active_sessions,
        "total_sessions_since_start": _total_sessions,
        "total_transcription_seconds": round(_total_transcription_s, 2),
        "model_ready": transcriber.is_ready(),
    }


# ==================================================================
# Batch REST (OpenAI-compatible) — backward-compat with previous deploy
# ==================================================================


@app.post("/v1/audio/transcriptions")
async def transcribe_batch(
    file: UploadFile = File(...),
    model: str = Form("parakeet-tdt-0.6b-v3"),
    response_format: str = Form("json"),
    token: Optional[str] = Query(None),
):
    _check_token(token)

    if not transcriber.is_ready():
        raise HTTPException(status_code=503, detail="Model still loading")

    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    loop = asyncio.get_running_loop()
    try:
        result: TranscriptionResult = await loop.run_in_executor(
            None, transcriber.transcribe_file, file_bytes
        )
    except Exception as e:
        logger.exception("Batch transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")

    global _total_transcription_s
    _total_transcription_s += result.duration_s

    if response_format == "text":
        return result.text
    return {"text": result.text}


# ==================================================================
# WebSocket streaming
# ==================================================================


@app.websocket("/ws")
async def ws_transcribe(ws: WebSocket, token: Optional[str] = Query(None)):
    # Auth
    if API_TOKEN and token != API_TOKEN:
        await ws.close(code=4401, reason="invalid token")
        return

    # Origin check (best-effort; browser enforces CORS but WS doesn't trigger preflight)
    origin = ws.headers.get("origin")
    if ALLOWED_ORIGINS and origin and origin not in ALLOWED_ORIGINS:
        logger.warning(f"WS rejected: origin={origin} not in allowlist")
        await ws.close(code=4403, reason="origin not allowed")
        return

    if not transcriber.is_ready():
        await ws.close(code=4503, reason="model loading")
        return

    await ws.accept()

    async def safe_send_json(payload: dict) -> bool:
        """Send JSON, swallow disconnect errors. Returns False if socket is dead."""
        try:
            await ws.send_json(payload)
            return True
        except Exception:
            return False

    global _active_sessions, _total_sessions, _total_transcription_s
    _active_sessions += 1
    _total_sessions += 1

    session = StreamSession()
    # Rolling overlap window (tunable via PARAKEET_OVERLAP_MS env var)
    session.overlap_bytes = (OVERLAP_MS * SAMPLE_RATE * 2) // 1000

    loop = asyncio.get_running_loop()

    await safe_send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "min_chunk_ms": MIN_CHUNK_MS,
        "max_chunk_ms": MAX_CHUNK_MS,
    })

    logger.info(f"WS session started (active={_active_sessions})")

    # Background task: periodically flush buffer when it reaches chunk size
    stop_flag = asyncio.Event()

    async def flusher():
        # Re-declare globals for this closure scope — Python's `global` is
        # per-function, NOT inherited by nested functions. Without this,
        # `_total_transcription_s += ...` below raises UnboundLocalError
        # on the FIRST assignment and kills the flusher silently.
        global _total_transcription_s

        logger.info("flusher task started")
        try:
            while not stop_flag.is_set():
                await asyncio.sleep(0.2)
                if stop_flag.is_set():
                    break

                if session.age_s > MAX_SESSION_MINUTES * 60:
                    await safe_send_json({"type": "info", "message": "session time limit reached"})
                    await ws.close(code=1000, reason="time limit")
                    break

                if session.buffer_ms >= MIN_CHUNK_MS:
                    chunk = session.take_chunk()
                    if len(chunk) < (SAMPLE_RATE // 10) * 2:  # < 100ms -> skip
                        continue
                    try:
                        result: TranscriptionResult = await loop.run_in_executor(
                            None, transcriber.transcribe_pcm16, chunk
                        )
                    except Exception as e:
                        logger.exception("Chunk transcription failed")
                        await safe_send_json({"type": "error", "code": "TRANSCRIBE_FAIL", "message": str(e)[:200]})
                        continue

                    text = result.text
                    if text:
                        session.committed_text = (session.committed_text + " " + text).strip()
                        await safe_send_json({
                            "type": "final",
                            "text": text,
                            "utterance_id": session.utterance_id,
                            "committed_text": session.committed_text,
                        })
                        session.utterance_id += 1
                        _total_transcription_s += result.duration_s

                # Anti-degrade reset after long silence (no audio received for N seconds)
                if session.should_reset():
                    session.reset_count += 1
                    session.buffer = bytearray()
                    await safe_send_json({
                        "type": "info",
                        "message": "reset after silence",
                        "reason": "anti_degrade",
                        "reset_count": session.reset_count,
                    })
                    session.last_audio_at = time.time()
        except Exception:
            logger.exception("flusher task crashed")
        finally:
            logger.info(f"flusher task exiting (stop_flag={stop_flag.is_set()}, utterances={session.utterance_id - 1})")

    flusher_task = asyncio.create_task(flusher())

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                if len(data) > MAX_FRAME_BYTES:
                    await safe_send_json({
                        "type": "error",
                        "code": "FRAME_TOO_LARGE",
                        "message": f"frame {len(data)}B > max {MAX_FRAME_BYTES}B",
                    })
                    continue
                session.append(data)
                continue

            if "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await safe_send_json({"type": "error", "code": "INVALID_JSON"})
                    continue

                mtype = payload.get("type")
                if mtype == "stop":
                    # Flush remaining buffer then close
                    if session.buffer_ms > 200:
                        chunk = session.take_chunk()
                        try:
                            result = await loop.run_in_executor(
                                None, transcriber.transcribe_pcm16, chunk
                            )
                            if result.text:
                                session.committed_text = (session.committed_text + " " + result.text).strip()
                                await safe_send_json({
                                    "type": "final",
                                    "text": result.text,
                                    "utterance_id": session.utterance_id,
                                    "committed_text": session.committed_text,
                                })
                                _total_transcription_s += result.duration_s
                        except Exception as e:
                            logger.exception("Final flush failed")
                    await safe_send_json({"type": "closed", "committed_text": session.committed_text})
                    break
                elif mtype == "ping":
                    await safe_send_json({"type": "pong"})
                elif mtype == "config":
                    # Future: language hint, custom vocab — currently ignored
                    await safe_send_json({"type": "config_ack"})
                else:
                    await safe_send_json({"type": "error", "code": "UNKNOWN_TYPE", "message": f"type={mtype}"})

    except WebSocketDisconnect:
        logger.info("WS client disconnected")
    except Exception:
        logger.exception("WS handler error")
    finally:
        stop_flag.set()
        try:
            await flusher_task
        except Exception:
            pass
        _active_sessions -= 1
        elapsed = session.age_s
        logger.info(
            f"WS session ended: age={elapsed:.1f}s, utterances={session.utterance_id - 1}, "
            f"resets={session.reset_count}"
        )
        try:
            if ws.client_state.value != 3:  # not already closed
                await ws.close()
        except Exception:
            pass
