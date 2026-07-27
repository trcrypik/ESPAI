import os, re, io, wave, subprocess, logging, asyncio, time
from urllib.parse import quote
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response, PlainTextResponse, StreamingResponse
from piper import PiperVoice
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("va")

app = FastAPI()
PLAY_RATE   = 22050                                   # Piper native rate, sent as-is
VOICE_ONNX  = os.getenv("PIPER_ONNX", "/app/models/ru_RU-ruslan-medium.onnx")
VOICE_JSON  = os.getenv("PIPER_JSON", "/app/models/ru_RU-ruslan-medium.onnx.json")
MODEL       = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY     = os.getenv("GEMINI_API_KEY", "")
PROMPT      = ("Ты голосовой помощник на русском языке. Отвечай естественно и по делу, "
               "без воды, обычно 2-4 предложения. Без списков, без скобок, без markdown, "
               "чтобы ответ было удобно озвучить синтезатором речи.")

_gemini = genai.Client(api_key=API_KEY) if API_KEY else None
_voice = None  # PiperVoice, loaded on startup

def load_voice():
    global _voice
    t0 = time.time()
    _voice = PiperVoice.load(VOICE_ONNX, config_path=VOICE_JSON)
    log.info(f"Piper voice loaded in {time.time()-t0:.1f}s, sample_rate={_voice.config.sample_rate}")

@app.on_event("startup")
async def startup():
    load_voice()
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

# ---- Piper: text -> raw PCM (model rate, 16-bit mono) ----
def synth_pcm(text: str) -> bytes:
    if hasattr(_voice, "synthesize_stream_raw"):
        return b"".join(_voice.synthesize_stream_raw(text))
    # fallback via WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        _voice.synthesize_wav(text, wf)
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        return wf.readframes(wf.getnframes())

# ---- input audio -> WAV (for Gemini) ----
def pcm_to_wav(pcm: bytes, rate: int, ch: int = 1, w: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch); wf.setsampwidth(w); wf.setframerate(rate); wf.writeframes(pcm)
    return buf.getvalue()

# ===================== endpoints =====================

@app.get("/")
async def root():
    return PlainTextResponse("VoiceAssist (Piper stream) ok. /health /pcm /chat /stream\n")

@app.get("/speedtest")
async def speedtest(size: int = Query(1000000)):
    async def gen():
        chunk = b"\0" * 65536
        remaining = size
        while remaining > 0:
            n = 65536 if remaining > 65536 else remaining
            yield chunk[:n]
            remaining -= n
    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"Content-Length": str(size)})

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "key_set": bool(API_KEY),
            "voice_loaded": _voice is not None, "play_rate": PLAY_RATE}

# non-streaming PCM (for SET test on device) — full buffer, Content-Length
@app.get("/pcm")
async def pcm(text: str = Query("Привет. Тест связи.")):
    log.info(f"PCM(text): {text[:60]}")
    out = await asyncio.to_thread(synth_pcm, text)
    return Response(content=out, media_type="application/octet-stream")

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
    out = await asyncio.to_thread(synth_pcm, text)
    headers = {"X-Reply-Text": quote(text, safe=""),
               "X-Reply-Oled": quote(translit(text), safe="")}
    return Response(content=out, media_type="application/octet-stream", headers=headers)

# STREAMING: Gemini text -> Piper PCM -> chunked stream (no buffering)
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

    # 2) Piper: text -> raw PCM (model rate)
    try:
        t0 = time.time()
        pcm = await asyncio.to_thread(synth_pcm, text)
        log.info(f"Piper TTS ok: {len(pcm)} bytes (~{len(pcm)/2/PLAY_RATE:.1f}s) in {time.time()-t0:.1f}s")
    except Exception as e:
        log.exception("piper error")
        return Response(status_code=502, content=f"piper error: {e}")

    # 3) stream PCM in small chunks (chunked transfer, no proxy buffering)
    async def pcm_stream():
        CHUNK = 4096
        for i in range(0, len(pcm), CHUNK):
            yield pcm[i:i + CHUNK]

    headers = {"X-Reply-Text": quote(text, safe=""),
               "X-Reply-Oled": quote(translit(text), safe=""),
               "X-Accel-Buffering": "no"}
    return StreamingResponse(pcm_stream(), media_type="application/octet-stream", headers=headers)
