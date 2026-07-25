FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Northflank передаёт PORT через env (обычно 80 или 10000)
# Shell-форма CMD позволяет раскрыть $PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-80}
