import os, re, io, wave, subprocess, logging, asyncio, time
from urllib.parse import quote
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response, PlainTextResponse
from piper import PiperVoice
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("va")

app = FastAPI()
PLAY_RATE   = 24000                                   # PCM rate sent to ESP32
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

# ---- Piper: text -> WAV (model sample rate) ----
def synthesize_piper(text: str) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_voice.config.sample_rate)
        _voice.synthesize(text, wf)
    return buf.getvalue()

# ---- resample WAV -> raw PCM 24 kHz mono (for ESP32) ----
def wav_to_pcm_24k(wav_bytes: bytes) -> bytes:
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(PLAY_RATE), "-ac", "1", "pipe:1"],
        input=wav_bytes, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf8", "ignore"))
    return p.stdout

# ---- TTS pipeline (runs in a thread so it doesn't block the event loop) ----
def _tts_pipeline(text: str) -> bytes:
    return wav_to_pcm_24k(synthesize_piper(text))

async def tts(text: str) -> bytes:
    return await asyncio.to_thread(_tts_pipeline, text)

# ---- input audio -> WAV (for Gemini) ----
def pcm_to_wav(pcm: bytes, rate: int, ch: int = 1, w: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch); wf.setsampwidth(w); wf.setframerate(rate); wf.writeframes(pcm)
    return buf.getvalue()

# ===================== endpoints =====================

@app.get("/")
async def root():
    return PlainTextResponse("VoiceAssist (Piper) ok. /health /tts /pcm /chat /stream\n")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "key_set": bool(API_KEY),
            "voice_loaded": _voice is not None}

@app.get("/tts")
async def tts_mp3(text: str = Query("Привет. Тест связи.")):
    pcm = await tts(text)
    return Response(content=pcm, media_type="application/octet-stream")

@app.get("/pcm")
async def pcm(text: str = Query("Привет. Тест связи.")):
    log.info(f"PCM(text): {text[:60]}")
    out = await tts(text)
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
    out = await tts(text)
    headers = {"X-Reply-Text": quote(text, safe=""),
               "X-Reply-Oled": quote(translit(text), safe="")}
    return Response(content=out, media_type="application/octet-stream", headers=headers)

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

    # 2) Piper: text -> PCM 24k (local, stable)
    try:
        t0 = time.time()
        pcm = await tts(text)
        log.info(f"Piper TTS ok: {len(pcm)} bytes (~{len(pcm)/2/PLAY_RATE:.1f}s) in {time.time()-t0:.1f}s")
    except Exception as e:
        log.exception("piper error")
        return Response(status_code=502, content=f"piper error: {e}")

    headers = {"X-Reply-Text": quote(text, safe=""),
               "X-Reply-Oled": quote(translit(text), safe="")}
    return Response(content=pcm, media_type="application/octet-stream", headers=headers)
