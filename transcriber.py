"""
Parakeet v3 multilingual transcriber (CPU, ONNX).

Wraps the `onnx-asr` library for use in both:
- Streaming WebSocket (chunked PCM16 16kHz mono)
- Batch REST (whole audio file)

Model: istupakov/parakeet-tdt-0.6b-v3-onnx (INT8 quantized, ~600MB)
"""

import asyncio
import io
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


MODEL_NAME = os.environ.get("PARAKEET_MODEL", "istupakov/parakeet-tdt-0.6b-v3-onnx")
SAMPLE_RATE = 16000
MIN_CHUNK_MS = int(os.environ.get("PARAKEET_MIN_CHUNK_MS", "2000"))
MAX_CHUNK_MS = int(os.environ.get("PARAKEET_MAX_CHUNK_MS", "6000"))
SILENCE_RESET_MS = int(os.environ.get("PARAKEET_SILENCE_RESET_MS", "30000"))
# Energy gate: skip inference if chunk RMS below this threshold (anti-hallucination)
# 0.005 is about -46 dBFS, empirically safe for speech vs background noise
SILENCE_RMS_THRESHOLD = float(os.environ.get("PARAKEET_SILENCE_RMS_THRESHOLD", "0.005"))

# Common ASR hallucinations on silent/noisy chunks — filter these out entirely.
# Matches when the WHOLE output is one of these (ignore leading/trailing punct).
HALLUCINATION_STRINGS = {
    s.lower() for s in [
        "mm-hmm", "mm hmm", "mmhmm", "mm-mm", "mm mm", "mmmm",
        "yeah", "yep", "uh", "um", "ah", "eh", "ehm", "hmm",
        "ok", "okay", "so", "no", "no no", "no, no",
        "bedade", "nova", "two", "yeah.", "yeah,",
        ".", "...", "?", "!",
    ]
}


@dataclass
class TranscriptionResult:
    text: str
    duration_s: float


class ParakeetTranscriber:
    """Thread-safe wrapper around onnx-asr Parakeet v3."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._loaded = False
        self._loading_error: Optional[str] = None

    def load(self) -> None:
        """Load ONNX model from HuggingFace (cached locally) + warmup."""
        import onnx_asr

        logger.info(f"Loading Parakeet v3 ONNX model: {MODEL_NAME}")
        start = time.time()
        try:
            self._model = onnx_asr.load_model(MODEL_NAME)
            elapsed = time.time() - start
            logger.info(f"Model loaded in {elapsed:.1f}s (sample_rate={SAMPLE_RATE})")

            # Warmup: run a dummy inference to JIT-compile ONNX kernels.
            # Without this, the FIRST real streaming chunk is 5-10x slower
            # than subsequent ones, which causes the live transcription to
            # fall behind and batch everything at session end.
            logger.info("Warming up model...")
            warmup_start = time.time()
            dummy = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2s of silence
            with self._lock:
                _ = self._model.recognize(dummy)
            logger.info(f"Warmup done in {time.time()-warmup_start:.1f}s")

            self._loaded = True
        except Exception as e:
            logger.exception("Model load failed")
            self._loading_error = str(e)
            raise

    def is_ready(self) -> bool:
        return self._loaded and self._model is not None

    def transcribe_pcm16(self, pcm16: bytes) -> TranscriptionResult:
        """Transcribe raw PCM int16 little-endian audio bytes.

        Audio is expected at SAMPLE_RATE mono.

        Silence/noise gating:
        - If the chunk RMS is below SILENCE_RMS_THRESHOLD, we skip inference
          entirely to avoid hallucinations ('Mm-hmm', 'Yeah', 'Bedade', ...)
        - If the model returns a string that matches a known hallucination
          (common filler words), we drop it.
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded")

        # Convert PCM16 LE bytes -> float32 in [-1, 1]
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        duration_s = len(samples) / SAMPLE_RATE

        # --- ENERGY GATE: skip silent/noise chunks entirely ---
        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) > 0 else 0.0
        if rms < SILENCE_RMS_THRESHOLD:
            logger.info(
                f"stream chunk: audio={duration_s:.2f}s, SKIPPED (rms={rms:.4f} < {SILENCE_RMS_THRESHOLD})"
            )
            return TranscriptionResult(text="", duration_s=duration_s)

        with self._lock:  # onnx-asr is not necessarily thread-safe
            start = time.time()
            text = self._model.recognize(samples)
            elapsed = time.time() - start

        text = (text or "").strip()

        # --- HALLUCINATION FILTER: drop known filler-only outputs ---
        normalized = text.lower().rstrip(".,!?;:").strip()
        if normalized in HALLUCINATION_STRINGS:
            logger.info(
                f"stream chunk: audio={duration_s:.2f}s, infer={elapsed:.3f}s, "
                f"rms={rms:.4f}, HALLUCINATION_FILTERED: {text!r}"
            )
            return TranscriptionResult(text="", duration_s=duration_s)

        logger.info(
            f"stream chunk: audio={duration_s:.2f}s, infer={elapsed:.3f}s, "
            f"rtf={elapsed/duration_s:.2f}x, rms={rms:.4f}, text={text[:60]!r}"
        )
        return TranscriptionResult(text=text, duration_s=duration_s)

    def transcribe_file(self, file_bytes: bytes) -> TranscriptionResult:
        """Transcribe a full audio file (any format supported by soundfile/ffmpeg).

        Resamples to 16kHz mono if needed.
        """
        import soundfile as sf

        try:
            data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
        except Exception:
            # Fallback via ffmpeg subprocess for wider format support
            data, sr = self._ffmpeg_decode(file_bytes)

        if data.ndim > 1:
            data = data.mean(axis=1)  # downmix to mono

        if sr != SAMPLE_RATE:
            data = self._resample_linear(data, sr, SAMPLE_RATE)

        duration_s = len(data) / SAMPLE_RATE
        with self._lock:
            start = time.time()
            text = self._model.recognize(data.astype(np.float32))
            elapsed = time.time() - start

        logger.info(
            f"File transcribed: {duration_s:.1f}s audio in {elapsed:.2f}s "
            f"(rtf={elapsed/duration_s:.2f}x)"
        )
        return TranscriptionResult(text=(text or "").strip(), duration_s=duration_s)

    @staticmethod
    def _ffmpeg_decode(file_bytes: bytes) -> tuple[np.ndarray, int]:
        """Fallback decoder via ffmpeg for formats soundfile can't handle (e.g., mp3)."""
        import subprocess

        proc = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate(file_bytes, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {err.decode(errors='ignore')[:200]}")
        data = np.frombuffer(out, dtype=np.float32)
        return data, SAMPLE_RATE

    @staticmethod
    def _resample_linear(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return data
        ratio = dst_sr / src_sr
        new_len = int(round(len(data) * ratio))
        # Simple linear interpolation (sufficient for 8k/22k/44k -> 16k speech)
        x_old = np.arange(len(data))
        x_new = np.linspace(0, len(data) - 1, new_len)
        return np.interp(x_new, x_old, data).astype(np.float32)


# ----- Streaming session state (per-WebSocket) -----


@dataclass
class StreamSession:
    """State for a single streaming WebSocket session.

    Accumulates PCM bytes, emits partials + finals based on fixed chunk boundaries.
    Strategy (MVP): fixed-interval chunks every MIN_CHUNK_MS, with a rolling
    window overlap of 500ms to avoid word cuts at boundaries.
    """
    started_at: float = field(default_factory=time.time)
    buffer: bytearray = field(default_factory=bytearray)
    last_audio_at: float = field(default_factory=time.time)
    utterance_id: int = 1
    total_samples: int = 0  # for timestamp computation
    overlap_bytes: int = 0  # rolling window overlap in bytes
    committed_text: str = ""
    partial_text: str = ""
    reset_count: int = 0

    @property
    def buffer_ms(self) -> int:
        # PCM16 = 2 bytes/sample at 16kHz mono
        return (len(self.buffer) // 2) * 1000 // SAMPLE_RATE

    @property
    def age_s(self) -> float:
        return time.time() - self.started_at

    def append(self, pcm_bytes: bytes) -> None:
        self.buffer.extend(pcm_bytes)
        self.last_audio_at = time.time()

    def take_chunk(self, max_ms: int = MAX_CHUNK_MS) -> bytes:
        """Extract accumulated PCM up to `max_ms` (reset buffer keeping overlap).

        If the buffer is larger than max_ms we truncate: this prevents an
        ever-growing buffer if inference is slower than ingest. The remaining
        audio stays in the buffer and will be processed on the next tick.
        """
        max_bytes = (max_ms * SAMPLE_RATE * 2) // 1000
        buf_len = len(self.buffer)

        if buf_len > max_bytes:
            chunk = bytes(self.buffer[:max_bytes])
            # Keep everything beyond max_bytes, minus overlap at the start
            start = max(0, max_bytes - self.overlap_bytes)
            self.buffer = bytearray(self.buffer[start:])
            return chunk

        if self.overlap_bytes > 0 and buf_len > self.overlap_bytes:
            chunk = bytes(self.buffer)
            self.buffer = bytearray(self.buffer[-self.overlap_bytes:])
            return chunk

        chunk = bytes(self.buffer)
        self.buffer = bytearray()
        return chunk

    def should_reset(self) -> bool:
        return (time.time() - self.last_audio_at) * 1000 > SILENCE_RESET_MS
