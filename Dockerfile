FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Piper Russian voice (ruslan-medium, ~60 MB)
RUN mkdir -p /app/models \
 && wget -q -O /app/models/ru_RU-ruslan-medium.onnx \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx" \
 && wget -q -O /app/models/ru_RU-ruslan-medium.onnx.json \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx.json"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
