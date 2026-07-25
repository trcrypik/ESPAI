from fastapi import FastAPI, Query
from fastapi.responses import Response
import edge_tts
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tts")

app = FastAPI()

@app.on_event("startup")
async def startup():
    log.info("TTS service started")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/tts")
async def tts(text: str = Query("Привет. Тест связи.")):
    log.info(f"TTS request: {text[:60]}...")
    comm = edge_tts.Communicate(text, voice="ru-RU-DmitryNeural")
    chunks = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    log.info(f"TTS done: {sum(len(c) for c in chunks)} bytes")
    return Response(content=b"".join(chunks), media_type="audio/mpeg")
