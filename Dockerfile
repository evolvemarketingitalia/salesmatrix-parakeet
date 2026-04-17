FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py transcriber.py ./

ENV HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models \
    PYTHONUNBUFFERED=1

EXPOSE 5092

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:5092/health',timeout=5); sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5092", "--ws-ping-interval", "20", "--ws-ping-timeout", "10"]
