from fastapi import FastAPI, Query
from fastapi.responses import Response, PlainTextResponse
import edge_tts
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tts")

app = FastAPI()

@app.on_event("startup")
async def startup():
    log.info("TTS service started")

@app.get("/")
async def root():
    # Чтобы в браузере по корню было что-то понятное
    return PlainTextResponse(
        "Start system...\n"
        "Система запущена!\n"
        "Тест: ок\n"
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/tts")
async def tts(text: str = Query("Привет. Тест связи.")):
    log.info(f"TTS request: {text[:60]}")
    comm = edge_tts.Communicate(text, voice="ru-RU-DmitryNeural")
    chunks = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    data = b"".join(chunks)
    log.info(f"TTS done: {len(data)} bytes")
    return Response(content=data, media_type="audio/mpeg")
