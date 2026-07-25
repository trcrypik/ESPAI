from fastapi import FastAPI, Query
from fastapi.responses import Response
import edge_tts

app = FastAPI()

@app.get("/tts")
async def tts(text: str = Query("Привет. Я твой голосовой ассистент. Тест связи.")):
    """Принимает текст, возвращает MP3."""
    comm = edge_tts.Communicate(text, voice="ru-RU-DmitryNeural")
    chunks = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return Response(content=b"".join(chunks), media_type="audio/mpeg")

@app.get("/health")
async def health():
    return {"status": "ok"}
