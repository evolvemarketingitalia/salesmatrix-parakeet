# Sales Matrix Parakeet v3 Streaming ASR

Self-hosted streaming speech-to-text server for Sales Matrix CRM.
Powered by NVIDIA Parakeet TDT 0.6B v3 (multilingual, 25 EU languages including Italian) via ONNX on CPU.

## Endpoints

- `GET /health` — liveness + model ready
- `GET /metrics` — CPU/RAM + session stats
- `POST /v1/audio/transcriptions` — OpenAI-compatible batch (multipart file)
- `WS /ws?token=...` — realtime streaming (PCM16 16kHz mono binary frames)

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `PARAKEET_API_TOKEN` | (empty = no auth) | Shared token for `/ws` and `/v1/audio/transcriptions` |
| `PARAKEET_ALLOWED_ORIGINS` | salesenhancer.it + localhost | CORS whitelist (comma-separated) |
| `PARAKEET_LOG_LEVEL` | INFO | Log verbosity |
| `PARAKEET_MIN_CHUNK_MS` | 2000 | Min audio buffer before transcription |
| `PARAKEET_MAX_CHUNK_MS` | 6000 | Max audio buffer before forced transcription |
| `PARAKEET_SILENCE_RESET_MS` | 30000 | Silence duration triggering state reset |
| `PARAKEET_MAX_SESSION_MINUTES` | 10 | Max WS session duration |
| `PARAKEET_MAX_FRAME_BYTES` | 8192 | Max single binary frame size |
| `HF_HOME` | /app/models | HuggingFace cache dir |
| `HF_HUB_CACHE` | /app/models | HuggingFace model cache dir |

## WebSocket protocol

**Client → Server:**
- Binary frame: raw PCM int16 little-endian, 16kHz mono, max 8192 bytes/frame (~256ms)
- JSON text: `{"type": "stop"}` to flush and close cleanly
- JSON text: `{"type": "ping"}` for keepalive
- JSON text: `{"type": "config", "language": "it"}` (future use)

**Server → Client (JSON text):**
- `{"type": "ready", "sample_rate": 16000, "min_chunk_ms": 2000, "max_chunk_ms": 6000}` — handshake ACK
- `{"type": "final", "text": "...", "utterance_id": N, "committed_text": "..."}` — transcribed chunk
- `{"type": "info", "message": "..."}` — reset / session limit notices
- `{"type": "error", "code": "...", "message": "..."}` — errors
- `{"type": "closed", "committed_text": "..."}` — clean close after `stop`
- `{"type": "pong"}` — response to ping

## Run locally

```bash
docker build -t parakeet-ws .
docker run -p 5092:5092 -v parakeet-models:/app/models parakeet-ws
```

First startup downloads ~600MB model from HuggingFace.

## Test

```bash
# Health
curl http://localhost:5092/health

# Batch transcription
curl -X POST http://localhost:5092/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "model=parakeet-tdt-0.6b-v3"

# WebSocket (with wscat)
wscat -c ws://localhost:5092/ws
```
