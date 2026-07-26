import os, re, io, wave, subprocess, logging, asyncio
from urllib.parse import quote
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response, PlainTextResponse, StreamingResponse
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
PROMPT    = ("Ты голосовой помощник на русском языке. Отвечай кратко и естественно, "
             "обычно 1-3 предложения. Без списков, без скобок, без markdown, "
             "чтобы ответ было удобно озвучить.")

_gemini = genai.Client(api_key=API_KEY) if API_KEY else None
if not API_KEY:
    log.warning("GEMINI_API_KEY is not set!")
else:
    log.info(f"Gemini model = {MODEL}")

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

# TTS with retry (Edge TTS sometimes returns NoAudioReceived)
async def synth_mp3(text: str, retries: int = 2) -> bytes:
    last_err = None
    for attempt in range(retries + 1):
        try:
            comm = edge_tts.Communicate(text, voice=VOICE)
            out = []
            async for c in comm.stream():
                if c["type"] == "audio":
                    out.append(c["data"])
            data = b"".join(out)
            if data:
                return data
            raise RuntimeError("empty audio")
        except Exception as e:
            last_err = e
            if attempt < retries:
                await asyncio.sleep(0.4 * (attempt + 1))
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

def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 110:
            buf = ""
            for s in re.split(r'(?<=,)\s+', p):
                if buf and len(buf) + len(s) + 1 <= 100:
                    buf += " " + s
                else:
                    if buf: out.append(buf)
                    buf = s
            if buf: out.append(buf)
        else:
            out.append(p)
    return out or [text]

async def tts_stream(sentences):
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        try:
            mp3 = await synth_mp3(sent)          # has retry inside
            yield mp3_to_pcm(mp3, PLAY_RATE)
        except Exception:
            log.exception("tts chunk error")     # skip sentence, keep going

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

@app.post("/stream")
async def stream(request: Request, rate: int = Query(16000)):
    body = await request.body()
    if len(body) < int(rate * 2 * 0.3):
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    if _gemini is None:
        return Response(status_code=500, content="GEMINI_API_KEY not configured")
    wav = pcm_to_wav(body, rate)
    log.info(f"STREAM in: {len(body)} bytes @{rate}Hz")
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
    sentences = split_sentences(text)
    headers = {"X-Reply-Text": quote(text, safe=""),
               "X-Reply-Oled": quote(translit(text), safe="")}
    return StreamingResponse(tts_stream(sentences),
                             media_type="application/octet-stream", headers=headers)
