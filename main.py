import os, re, io, wave, subprocess, logging, asyncio, time
from urllib.parse import quote
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response, PlainTextResponse, StreamingResponse
from piper import PiperVoice
from google import genai
from google.genai import types
import lameenc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("va")

app = FastAPI()
PLAY_RATE   = 22050                                   # Piper native rate, sent as-is
VOICE_ONNX  = os.getenv("PIPER_ONNX", "/app/models/ru_RU-ruslan-medium.onnx")
VOICE_JSON  = os.getenv("PIPER_JSON", "/app/models/ru_RU-ruslan-medium.onnx.json")
MODEL       = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY     = os.getenv("GEMINI_API_KEY", "")
MP3_BITRATE_KBPS = int(os.getenv("MP3_BITRATE_KBPS", "32"))   # lower = smaller/worse; 24-40 is the usable speech range
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

def make_mp3_encoder():
    # One encoder instance per reply, fed sentence-by-sentence so it stays a
    # single continuous, valid MP3 bitstream regardless of how many HTTP
    # chunks it ends up split across.
    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_BITRATE_KBPS)
    enc.set_in_sample_rate(PLAY_RATE)
    enc.set_channels(1)
    enc.set_quality(2)   # 2 = highest quality / slowest, 7 = fastest; CPU isn't the bottleneck here
    enc.silence()        # don't let LAME spam stdout
    return enc

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

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\u2026])\s+")
def split_sentences(text: str):
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return parts or ([text] if text else [])

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
    return PlainTextResponse("Error 404!\n")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "key_set": bool(API_KEY),
            "voice_loaded": _voice is not None, "play_rate": PLAY_RATE}

@app.get("/tts")
async def tts_test(text: str = Query("Привет. Тест связи.")):
    pcm = await asyncio.to_thread(synth_pcm, text)
    return Response(content=pcm, media_type="application/octet-stream")

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
        resp = await asyncio.to_thread(
            _gemini.models.generate_content,
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

# STREAM: Gemini -> Piper -> MP3, sent sentence-by-sentence as real chunked transfer.
# Why MP3: raw PCM at 22050Hz/16-bit/mono is 44100 bytes/sec. At the
# 0.1-0.5 Mbit/s this device is getting to Northflank that alone can be
# slower than realtime, so any transport hiccup empties the playback buffer.
# MP3 at 32 kbps mono is ~4000 bytes/sec -- about 1/10th the data -- which
# both finishes transferring faster (less time for something in the network
# path to time the connection out) and leaves a much bigger cushion before
# the device's buffer runs dry.
# Why still per-sentence: the old version synthesized the WHOLE reply before
# sending a single byte. On a slow/high-latency link that's many seconds of
# total silence on the TCP connection right after headers -- exactly the
# pattern idle-timeout proxies kill. The ESP32 firmware already has a
# chunked-transfer MP3 player that this now feeds.
@app.post("/stream")
async def stream(request: Request, rate: int = Query(16000)):
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
            contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                      types.Part.from_text(text=PROMPT)])
        text = (getattr(resp, "text", None) or "").strip()
    except Exception as e:
        log.exception("gemini error")
        return Response(status_code=502, content=f"gemini error: {e}")
    if not text:
        text = "Я не расслышал, повтори пожалуйста."
    log.info(f"REPLY ({time.time()-t0:.1f}s): {text[:160]}")

    sentences = split_sentences(text)

    async def gen():
        tg0 = time.time()
        total_pcm = 0
        total_mp3 = 0
        encoder = await asyncio.to_thread(make_mp3_encoder)
        for i, sent in enumerate(sentences):
            ts = time.time()
            try:
                pcm = await asyncio.to_thread(synth_pcm, sent)
            except Exception:
                log.exception(f"piper error on chunk {i+1}/{len(sentences)}")
                continue
            total_pcm += len(pcm)
            try:
                mp3_bytes = await asyncio.to_thread(encoder.encode, pcm)
            except Exception:
                log.exception(f"mp3 encode error on chunk {i+1}/{len(sentences)}")
                continue
            total_mp3 += len(mp3_bytes)
            log.info(f"chunk {i+1}/{len(sentences)}: pcm={len(pcm)}B mp3={len(mp3_bytes)}B in {time.time()-ts:.2f}s")
            if mp3_bytes:
                yield bytes(mp3_bytes)
        try:
            tail = await asyncio.to_thread(encoder.flush)
        except Exception:
            log.exception("mp3 flush error")
            tail = b""
        if tail:
            total_mp3 += len(tail)
            yield bytes(tail)
        log.info(f"STREAM done: pcm={total_pcm}B mp3={total_mp3}B ({MP3_BITRATE_KBPS}kbps), "
                 f"{len(sentences)} chunk(s) in {time.time()-tg0:.1f}s")

    headers = {"X-Reply-Text": quote(text, safe=""),
               "X-Reply-Oled": quote(translit(text), safe=""),
               "X-Audio-Format": "mp3",
               "X-Accel-Buffering": "no",
               "Cache-Control": "no-store"}
    # No Content-Length here on purpose -> real Transfer-Encoding: chunked.
    return StreamingResponse(gen(), media_type="audio/mpeg", headers=headers)

# speedtest: stream N zero-bytes, log how much the server actually sent
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
    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store",
                                      "Content-Length": str(size)})
