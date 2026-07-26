import os, re, io, wave, subprocess, logging, asyncio
from urllib.parse import quote
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response, PlainTextResponse
import edge_tts
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("va")

app = FastAPI()
VOICE     = "ru-RU-DmitryNeural"
PLAY_RATE = 24000
MODEL     = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY   = os.getenv("GEMINI_API_KEY", "")
PROMPT    = ("Ты голосовой помощник на русском языке. Отвечай естественно и по делу, "
             "без воды, обычно 2-4 предложения. Без списков, без скобок, без markdown, "
             "чтобы ответ было удобно озвучить синтезатором речи.")

_gemini = genai.Client(api_key=API_KEY) if API_KEY else None
if not API_KEY:
    log.warning("GEMINI_API_KEY is not set!")
else:
    log.info(f"Gemini model = {MODEL}")

# ---- translit ru -> lat (for OLED, ASCII only) ----
_RU  = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
_LAT = ["a","b","v","g","d","e","e","zh","z","i","y","k","l","m","n","o",
        "p","r","s","t","u","f","h","ts","ch","sh","sch","","y","","e","yu","ya"]
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
    def __init__(self, a): self.a = a
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            p = scope.get("path", "")
            if "//" in p:
                scope = dict(scope); scope["path"] = re.sub(r"/+", "/", p)
                scope["raw_path"] = scope["path"].encode()
        await self.a(scope, receive, send)
app.add_middleware(SlashNormalizer)

# ---- TTS: ONE request for the whole reply, native receive_timeout (no asyncio.wait_for) ----
async def synth_mp3(text: str, retries: int = 2, receive_timeout: float = 25.0) -> bytes:
    last_err = None
    for attempt in range(retries + 1):
        try:
            comm = edge_tts.Communicate(text, voice=VOICE, receive_timeout=receive_timeout)
            out = []
            async for c in comm.stream():
                if c["type"] == "audio":
                    out.append(c["data"])
            data = b"".join(out)
            if not data:
                raise RuntimeError("empty audio")
            return data
        except Exception as e:
            last_err = e
            log.warning(f"synth_mp3 attempt {attempt} failed: {e}")
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))
    raise last_err

def mp3_to_pcm(mp3: bytes, rate: int = PLAY_RATE) -> bytes:
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(rate), "-ac", "1", "pipe:1"],
        input=mp3, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf8", "ignore"))
    return p.stdout

def pcm_to_wav(pcm: bytes, rate: int, ch: int = 1, w: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch); wf.setsampwidth(w); wf.setframerate(rate); wf.writeframes(pcm)
    return buf.getvalue()

# ===================== endpoints =====================

@app.get("/")
async def root():
    return PlainTextResponse("VoiceAssist ok. /health /tts /pcm /chat /stream\n")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "key_set": bool(API_KEY)}

@app.get("/tts")
async def tts(text: str = Query("Привет. Тест связи.")):
    return Response(content=await synth_mp3(text), media_type="audio/mpeg")

@app.get("/pcm")
async def pcm(text: str = Query("Привет. Тест связи.")):
    log.info(f"PCM(text): {text[:60]}")
    return Response(content=mp3_to_pcm(await synth_mp3(text)), media_type="application/octet-stream")

@app.post("/chat")
async def chat(request: Request, rate: int = Query(16000)):
    body = await request.body()
    if len(body) < int(rate * 2 * 0.3):
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    if _gemini is None:
        return Response(status_code=500, content="GEMINI_API_KEY not configured")
    wav = pcm_to_wav(body, rate)
    log.info(f"CHAT in: {len(body)} bytes @{rate}Hz")
    try:
        resp = _gemini.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                      types.Part.from_text(text=PROMPT)])
        text = (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        log.exception("gemini error")
        return Response(status_code=502, content=f"gemini error: {e}")
    if not text:
        text = "Я не расслышал, повтори пожалуйста."
    log.info(f"REPLY: {text[:160]}")
    out = mp3_to_pcm(await synth_mp3(text), PLAY_RATE)
    headers = {"X-Reply-Text": quote(text, safe=""),
               "X-Reply-Oled": quote(translit(text), safe="")}
    return Response(content=out, media_type="application/octet-stream", headers=headers)

# STREAM endpoint: now ONE Gemini call (audio->text) + ONE Edge TTS call (text->pcm)
@app.post("/stream")
async def stream(request: Request, rate: int = Query(16000)):
    body = await request.body()
    if len(body) < int(rate * 2 * 0.3):
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    if _gemini is None:
        return Response(status_code=500, content="GEMINI_API_KEY not configured")
    wav = pcm_to_wav(body, rate)
    log.info(f"STREAM in: {len(body)} bytes @{rate}Hz")

    # 1) Gemini: speech -> answer text
    try:
        resp = _gemini.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                      types.Part.from_text(text=PROMPT)])
        text = (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        log.exception("gemini error")
        return Response(status_code=502, content=f"gemini error: {e}")
    if not text:
        text = "Я не расслышал, повтори пожалуйста."
    log.info(f"REPLY: {text[:160]}")

    # 2) Edge TTS: whole answer in ONE request (with retry + native timeout)
    try:
        mp3 = await synth_mp3(text)
    except Exception as e:
        log.exception("tts error")
        return Response(status_code=502, content=f"tts error: {e}")

    pcm = mp3_to_pcm(mp3, PLAY_RATE)
    log.info(f"TTS ok: {len(pcm)} bytes (~{len(pcm)/2/PLAY_RATE:.1f}s)")

    headers = {"X-Reply-Text": quote(text, safe=""),
               "X-Reply-Oled": quote(translit(text), safe="")}
    return Response(content=pcm, media_type="application/octet-stream", headers=headers)
