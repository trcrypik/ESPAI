import os, re, io, wave, subprocess, logging
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
# shorter answers -> faster TTS -> fewer ingress TTFB timeouts
PROMPT    = ("Ты голосовой помощник на русском языке. Отвечай ОЧЕНЬ кратко: "
             "максимум 1-2 предложения, до 20 слов. Без списков, без скобок, "
             "без markdown, чтобы ответ было удобно озвучить.")

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

# collapse multiple slashes (//chat -> /chat)
class SlashNormalizer:
    def init(self, a): self.a = a
    async def call(self, scope, receive, send):
        if scope["type"] == "http":
            p = scope.get("path", "")
            if "//" in p:
                scope = dict(scope); scope["path"] = re.sub(r"/+", "/", p)
                scope["raw_path"] = scope["path"].encode()
        await self.a(scope, receive, send)
app.add_middleware(SlashNormalizer)

async def synth_mp3(text: str) -> bytes:
    comm = edge_tts.Communicate(text, voice=VOICE)
    out = []
    async for c in comm.stream():
        if c["type"] == "audio":
            out.append(c["data"])
    return b"".join(out)

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

@app.get("/")
async def root():
    return PlainTextResponse("VoiceAssist ok. /health /tts /pcm /chat\n")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "key_set": bool(API_KEY)}

@app.get("/tts")
async def tts(text: str = Query("Привет. Тест связи.")):
    return Response(content=await synth_mp3(text), media_type="audio/mpeg")

@app.get("/pcm")
async def pcm(text: str = Query("Привет. Тест связи.")):
    log.info(f"PCM(text): {text[:60]}")
    mp3 = await synth_mp3(text)
    data = mp3_to_pcm(mp3)
    return Response(content=data, media_type="application/octet-stream")

@app.post("/chat")
async def chat(request: Request, rate: int = Query(16000)):
    body = await request.body()
    min_bytes = int(rate * 2 * 0.3)
    if len(body) < min_bytes:
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    if _gemini is None:
        return Response(status_code=500, content="GEMINI_API_KEY not configured")

    wav = pcm_to_wav(body, rate)
    log.info(f"CHAT in: {len(body)} bytes @{rate}Hz -> wav {len(wav)}B")
