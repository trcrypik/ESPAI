# VoiceAssist API — edge-tts + Gemini (no Piper / no local TTS models)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: edge-tts uses asyncio + websockets only; no ffmpeg/onnx needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

# Northflank / container defaults
ENV PORT=8080 \
    GEMINI_MODEL=gemini-3.5-flash-lite \
    EDGE_VOICE=ru-RU-DmitryNeural

EXPOSE 8080

# GEMINI_API_KEY must be set at runtime (Northflank secret)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 75"]
