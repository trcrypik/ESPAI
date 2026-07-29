import os
import re
import io
import wave
import logging
import asyncio
import time
from urllib.parse import quote

from fastapi import FastAPI, Query, Request
from fastapi.responses import Response, PlainTextResponse, StreamingResponse
from google import genai
from google.genai import types
import edge_tts

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("va")

app = FastAPI()

# ---- config ----
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY", "")
# Russian neural voices; alternatives: ru-RU-SvetlanaNeural
EDGE_VOICE = os.getenv("EDGE_VOICE", "ru-RU-DmitryNeural")
# edge-tts default is \~48 kbps mono MP3 @ 24 kHz — fine for ESP path
PROMPT = (
    "Ты голосовой помощник на русском языке. Отвечай естественно и по делу, "
    "без воды, обычно 2-4 предложения. Без списков, без скобок, без markdown, "
    "чтобы ответ было удобно озвучить синтезатором речи."
)

_gemini = genai.Client(api_key=API_KEY) if API_KEY else None


@app.on_event("startup")
async def startup():
    if not API_KEY:
        log.warning("GEMINI_API_KEY is not set!")
    else:
        log.info(f"Gemini model = {MODEL}")
    log.info(f"edge-tts voice = {EDGE_VOICE}")


# ---- translit ru -> lat (for OLED, ASCII only) ----
_RU = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
_LAT = [
    "a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o",
    "p", "r", "s", "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya",
]


def translit(text: str) -> str:
    out = []
    for ch in text:
        idx = _RU.find(ch.lower())
        if idx >= 0:
            t = _LAT[idx]
            out.append(t.capitalize() if (ch.isupper() and t) else t)
        else:
            out.append(ch)
    return "".join(out)


class SlashNormalizer:
    def __init__(self, a):
        self.a = a

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            p = scope.get("path", "")
            if "//" in p:
                scope = dict(scope)
                scope["path"] = re.sub(r"/+", "/", p)
                scope["raw_path"] = scope["path"].encode()
        await self.a(scope, receive, send)


app.add_middleware(SlashNormalizer)


# ---- input audio -> WAV (for Gemini) ----
def pcm_to_wav(pcm: bytes, rate: int, ch: int = 1, w: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(w)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def edge_mp3_chunks(text: str):
    """Yield MP3 bytes from edge-tts as soon as Microsoft sends them."""
    communicate = edge_tts.Communicate(text, EDGE_VOICE)
    total = 0
    t0 = time.time()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio" and chunk.get("data"):
            data = chunk["data"]
            total += len(data)
            yield data
    log.info(f"edge-tts done: {total}B mp3 in {time.time() - t0:.1f}s voice={EDGE_VOICE}")


async def edge_mp3_all(text: str) -> bytes:
    """Collect full MP3 (for /pcm test endpoint)."""
    parts = []
    async for b in edge_mp3_chunks(text):
        parts.append(b)
    return b"".join(parts)


# ===================== endpoints =====================

@app.get("/")
async def root():
    return PlainTextResponse(
        "VoiceAssist (edge-tts) ok. /health /tts /pcm /chat /stream /speedtest\n"
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "key_set": bool(API_KEY),
        "tts": "edge-tts",
        "voice": EDGE_VOICE,
    }


@app.get("/tts")
async def tts_test(text: str = Query("Привет. Тест связи.")):
    mp3 = await edge_mp3_all(text)
    return Response(content=mp3, media_type="audio/mpeg")


@app.get("/pcm")
async def pcm(text: str = Query("Привет. Тест связи.")):
    """
    Kept for ESP SET button compatibility.
    Name is historical: body is now MP3 (audio/mpeg), not raw PCM.
    ESP path for SET uses playPCM on whatever bytes arrive — for TTS test
    prefer /tts or switch firmware SET to expect mp3 later.
    For now we still return MP3; if firmware playPCM is used, SET test
    will sound wrong until you point SET at a decode path or keep a
    small PCM fallback. Prefer using voice cycle (/stream) as primary.
    """
    log.info(f"PCM/TTS(text): {text[:60]}")
    mp3 = await edge_mp3_all(text)
    return Response(content=mp3, media_type="audio/mpeg")


@app.post("/chat")
async def chat(request: Request, rate: int = Query(16000)):
    """Non-streaming: full MP3 body + reply headers (same as before)."""
    body = await request.body()
    if len(body) < int(rate * 2 * 0.3):
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    if _gemini is None:
        return Response(status_code=500, content="GEMINI_API_KEY not configured")

    wav = pcm_to_wav(body, rate)
    log.info(f"CHAT in: {len(body)} bytes @{rate}Hz")
    try:
        resp = await asyncio.to_thread(
            _gemini.models.generate_content,
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                types.Part.from_text(text=PROMPT),
            ],
        )
        text = (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        log.exception("gemini error")
        return Response(status_code=502, content=f"gemini error: {e}")

    if not text:
        text = "Я не расслышал, повтори пожалуйста."
    log.info(f"REPLY: {text[:160]}")

    mp3 = await edge_mp3_all(text)
    headers = {
        "X-Reply-Text": quote(text, safe=""),
        "X-Reply-Oled": quote(translit(text), safe=""),
        "X-Audio-Format": "mp3",
    }
    return Response(content=mp3, media_type="audio/mpeg", headers=headers)


@app.post("/stream")
async def stream(request: Request, rate: int = Query(16000)):
    """
    POST mic PCM -> Gemini -> edge-tts MP3, streamed chunked.

    Critical for ESP/Northflank:
    - Headers (incl. X-Reply-*) are fixed as soon as Gemini returns text,
      before any TTS bytes. Client can show OLED text immediately.
    - Body is Transfer-Encoding: chunked (no Content-Length) so first MP3
      packets leave the server as Microsoft sends them — less idle time
      on the TLS link after the Gemini wait.
    """
    body = await request.body()
    if len(body) < int(rate * 2 * 0.3):
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    if _gemini is None:
        return Response(status_code=500, content="GEMINI_API_KEY not configured")

    wav = pcm_to_wav(body, rate)
    log.info(f"STREAM in: {len(body)} bytes @{rate}Hz")
    t0 = time.time()
    try:
        resp = await asyncio.to_thread(
            _gemini.models.generate_content,
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                types.Part.from_text(text=PROMPT),
            ],
        )
        text = (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        log.exception("gemini error")
        return Response(status_code=502, content=f"gemini error: {e}")

    if not text:
        text = "Я не расслышал, повтори пожалуйста."
    log.info(f"REPLY ({time.time() - t0:.1f}s): {text[:160]}")

    async def gen():
        tg0 = time.time()
        total = 0
        try:
            async for data in edge_mp3_chunks(text):
                total += len(data)
                yield data
        except Exception:
            log.exception("edge-tts stream error")
        log.info(
            f"STREAM done: mp3={total}B edge-tts in {time.time() - tg0:.1f}s "
            f"(total wall {time.time() - t0:.1f}s)"
        )

    # Same headers the ESP firmware already parses in netStreamChat():
    #   x-reply-text, x-reply-oled, x-audio-format / content-type audio/mpeg
    #   transfer-encoding: chunked  (no Content-Length on purpose)
    headers = {
        "X-Reply-Text": quote(text, safe=""),
        "X-Reply-Oled": quote(translit(text), safe=""),
        "X-Audio-Format": "mp3",
        "X-Accel-Buffering": "no",
        "Cache-Control": "no-store",
    }
    return StreamingResponse(gen(), media_type="audio/mpeg", headers=headers)


@app.get("/speedtest")
async def speedtest(size: int = Query(1000000)):
    async def gen():
        chunk = b"\0" * 65536
        remaining = size
        sent = 0
        while remaining > 0:
            n = 65536 if remaining > 65536 else remaining
            yield chunk[:n]
            sent += n
            remaining -= n
        log.info(f"speedtest: SERVER sent all {sent} bytes")

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-store",
            "Content-Length": str(size),
        },
)
