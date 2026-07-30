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
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_VOICE = os.getenv("EDGE_VOICE", "ru-RU-DmitryNeural")

ALLOWED_VOICES = {
    "ru-RU-DmitryNeural",
    "ru-RU-SvetlanaNeural",
}

PROMPT = (
    "Ты голосовой помощник на русском языке. Отвечай естественно и по делу, "
    "без воды, обычно 2-4 предложения. Без списков, без скобок, без markdown, "
    "чтобы ответ было удобно озвучить синтезатором речи."
)

_gemini = genai.Client(api_key=API_KEY) if API_KEY else None


def pick_voice(requested: str | None) -> str:
    v = (requested or "").strip()
    if v in ALLOWED_VOICES:
        return v
    return DEFAULT_VOICE if DEFAULT_VOICE in ALLOWED_VOICES else "ru-RU-DmitryNeural"


@app.on_event("startup")
async def startup():
    if not API_KEY:
        log.warning("GEMINI_API_KEY is not set!")
    else:
        log.info("Gemini model = %s", MODEL)
    log.info("default edge-tts voice = %s", DEFAULT_VOICE)


# ---- translit ru -> lat (OLED, ASCII) ----
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
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            p = scope.get("path", "")
            if "//" in p:
                scope = dict(scope)
                scope["path"] = re.sub(r"/+", "/", p)
                scope["raw_path"] = scope["path"].encode()
        await self.app(scope, receive, send)


app.add_middleware(SlashNormalizer)


def pcm_to_wav(pcm: bytes, rate: int, ch: int = 1, w: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(w)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def edge_mp3_chunks(text: str, voice: str):
    """Yield MP3 bytes from edge-tts as soon as Microsoft sends them."""
    communicate = edge_tts.Communicate(text, voice)
    total = 0
    t0 = time.time()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio" and chunk.get("data"):
            data = chunk["data"]
            total += len(data)
            yield data
    log.info(
        "edge-tts done: %dB mp3 in %.1fs voice=%s",
        total,
        time.time() - t0,
        voice,
    )


async def edge_mp3_all(text: str, voice: str) -> bytes:
    parts = []
    async for b in edge_mp3_chunks(text, voice):
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
        "default_voice": DEFAULT_VOICE,
        "allowed_voices": sorted(ALLOWED_VOICES),
    }


@app.get("/tts")
async def tts_test(
    text: str = Query("Привет. Тест связи."),
    voice: str = Query(None),
):
    use_voice = pick_voice(voice)
    mp3 = await edge_mp3_all(text, use_voice)
    return Response(content=mp3, media_type="audio/mpeg")


@app.get("/pcm")
async def pcm(
    text: str = Query("Привет. Тест связи."),
    voice: str = Query(None),
):
    """
    Historical name. Body is MP3 (audio/mpeg), not raw PCM.
    Kept for compatibility; primary path is POST /stream.
    """
    use_voice = pick_voice(voice)
    log.info("PCM/TTS(text): %s voice=%s", text[:60], use_voice)
    mp3 = await edge_mp3_all(text, use_voice)
    return Response(content=mp3, media_type="audio/mpeg")


@app.post("/chat")
async def chat(
    request: Request,
    rate: int = Query(16000),
    voice: str = Query(None),
):
    """Non-streaming: full MP3 body + reply headers."""
    body = await request.body()
    if len(body) < int(rate * 2 * 0.3):
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    if _gemini is None:
        return Response(status_code=500, content="GEMINI_API_KEY not configured")

    use_voice = pick_voice(voice)
    wav = pcm_to_wav(body, rate)
    log.info("CHAT in: %d bytes @%dHz voice=%s", len(body), rate, use_voice)

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
    log.info("REPLY: %s", text[:160])

    mp3 = await edge_mp3_all(text, use_voice)
    headers = {
        "X-Reply-Text": quote(text, safe=""),
        "X-Reply-Oled": quote(translit(text), safe=""),
        "X-Audio-Format": "mp3",
    }
    return Response(content=mp3, media_type="audio/mpeg", headers=headers)


@app.post("/stream")
async def stream(
    request: Request,
    rate: int = Query(16000),
    voice: str = Query(None),
):
    """
    POST mic PCM -> Gemini -> edge-tts MP3, streamed chunked.

    - Headers (X-Reply-*) fixed as soon as Gemini returns text.
    - Body is Transfer-Encoding: chunked (no Content-Length).
    - voice=ru-RU-DmitryNeural | ru-RU-SvetlanaNeural
    """
    body = await request.body()
    if len(body) < int(rate * 2 * 0.3):
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    if _gemini is None:
        return Response(status_code=500, content="GEMINI_API_KEY not configured")

    use_voice = pick_voice(voice)
    wav = pcm_to_wav(body, rate)
    log.info("STREAM in: %d bytes @%dHz voice=%s", len(body), rate, use_voice)
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
    log.info("REPLY (%.1fs): %s", time.time() - t0, text[:160])

    async def gen():
        tg0 = time.time()
        total = 0
        try:
            async for data in edge_mp3_chunks(text, use_voice):
                total += len(data)
                yield data
        except Exception:
            log.exception("edge-tts stream error")
        log.info(
            "STREAM done: mp3=%dB edge-tts in %.1fs (total wall %.1fs) voice=%s",
            total,
            time.time() - tg0,
            time.time() - t0,
            use_voice,
        )

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
        log.info("speedtest: SERVER sent all %d bytes", sent)

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-store",
            "Content-Length": str(size),
        },
    )
