import os
import re
import io
import wave
import logging
import asyncio
import time
from collections import deque
from datetime import datetime
from urllib.parse import quote
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import Response, PlainTextResponse, StreamingResponse
from google import genai
from google.genai import types
import edge_tts
import httpx

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("va")

app = FastAPI()

# ---- config ----
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_VOICE = os.getenv("EDGE_VOICE", "ru-RU-DmitryNeural")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
HISTORY_MAX = int(os.getenv("HISTORY_MAX", "30"))  # max messages (user+model)

ALLOWED_VOICES = {
    "ru-RU-DmitryNeural",
    "ru-RU-SvetlanaNeural",
}

_gemini = genai.Client(api_key=API_KEY) if API_KEY else None

# In-memory dialogue history for single-device assistant.
# Each item: {"role": "user"|"model", "text": str}
_history: deque[dict[str, str]] = deque(maxlen=HISTORY_MAX)
_history_lock = asyncio.Lock()


def kyiv_now_str() -> str:
    """Current date/time in Europe/Kyiv (no shell, no extra tokens from date cmd)."""
    now = datetime.now(ZoneInfo("Europe/Kyiv"))
    # Example: 2026-07-30 Thursday 14:44:12 EEST (UTC+03:00)
    return now.strftime("%Y-%m-%d %A %H:%M:%S %Z (UTC%z)")


def pick_voice(requested: str | None) -> str:
    v = (requested or "").strip()
    if v in ALLOWED_VOICES:
        return v
    return DEFAULT_VOICE if DEFAULT_VOICE in ALLOWED_VOICES else "ru-RU-DmitryNeural"


def system_instruction(kyiv: str) -> str:
    return (
        "Ты голосовой помощник на русском языке. "
        "Отвечай естественно и по делу, обычно 2–5 предложений. "
        "Без markdown, без списков со звёздочками, без скобок-пояснений — текст должен быть удобен для озвучки.\n\n"
        f"Текущие дата и время по Киеву: {kyiv}.\n"
        "Это актуальное время «сейчас». При любых вопросах про «сегодня», «сейчас», «на этой неделе», "
        "новости, курс, погоду, расписание — обязательно опирайся на это время, "
        "а не на знания из обучения. Для свежих фактов из интернета вызывай инструмент internet_search; "
        "в поисковый запрос включай год/дату по необходимости, исходя из киевского времени выше.\n"
        "Если поиск не нужен — отвечай сразу."
    )


@app.on_event("startup")
async def startup():
    if not API_KEY:
        log.warning("GEMINI_API_KEY is not set!")
    else:
        log.info("Gemini model = %s", MODEL)
    log.info("default edge-tts voice = %s", DEFAULT_VOICE)
    log.info("Tavily configured = %s", bool(TAVILY_API_KEY))
    log.info("history max messages = %d", HISTORY_MAX)
    log.info("Kyiv now = %s", kyiv_now_str())


# ---- translit (OLED fallback) ----
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


# ---- Tavily ----
async def tavily_search(query: str, max_results: int = 5) -> str:
    if not TAVILY_API_KEY:
        return "Поиск недоступен: TAVILY_API_KEY не задан на сервере."
    kyiv = kyiv_now_str()
    # Soft-hint date into query context without burning model tokens on shell date
    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": True,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if r.status_code != 200:
            log.warning("Tavily HTTP %s: %s", r.status_code, r.text[:300])
            return f"Ошибка поиска Tavily HTTP {r.status_code}"
        data = r.json()
    except Exception as e:
        log.exception("Tavily request failed")
        return f"Ошибка поиска: {e}"

    parts: list[str] = [f"(Справка: сейчас по Киеву {kyiv})"]
    if data.get("answer"):
        parts.append(f"Краткий ответ Tavily: {data['answer']}")
    for i, item in enumerate(data.get("results") or [], 1):
        title = item.get("title") or ""
        url = item.get("url") or ""
        content = (item.get("content") or "")[:500]
        parts.append(f"{i}. {title}\n{content}\nИсточник: {url}")
    if len(parts) == 1:
        parts.append("Ничего не найдено.")
    return "\n\n".join(parts)


SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="internet_search",
            description=(
                "Поиск актуальной информации в интернете (новости, погода, курсы, факты, "
                "события). Используй, когда нужны свежие данные или пользователь просит найти/погуглить. "
                "Учитывай текущую киевскую дату/время из системной инструкции."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Поисковый запрос на русском или английском",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)


def _extract_function_calls(resp: Any) -> list[tuple[str, dict, str]]:
    """Return list of (name, args, call_id)."""
    out: list[tuple[str, dict, str]] = []
    try:
        fcs = getattr(resp, "function_calls", None)
        if fcs:
            for fc in fcs:
                name = getattr(fc, "name", "") or ""
                args = dict(getattr(fc, "args", None) or {})
                cid = getattr(fc, "id", None) or name
                out.append((name, args, cid))
            if out:
                return out
    except Exception:
        pass
    try:
        for cand in resp.candidates or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in content.parts or []:
                fc = getattr(part, "function_call", None)
                if not fc:
                    continue
                name = getattr(fc, "name", "") or ""
                args = dict(getattr(fc, "args", None) or {})
                cid = getattr(fc, "id", None) or name
                out.append((name, args, cid))
    except Exception:
        log.exception("parse function_calls")
    return out


def _model_content_from_response(resp: Any) -> types.Content | None:
    try:
        for cand in resp.candidates or []:
            if getattr(cand, "content", None):
                return cand.content
    except Exception:
        pass
    return None


async def gemini_transcribe(wav: bytes) -> str:
    """Speech -> text only."""
    assert _gemini is not None
    resp = await asyncio.to_thread(
        _gemini.models.generate_content,
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=wav, mime_type="audio/wav"),
            types.Part.from_text(
                text="Распознай речь и верни только текст того, что сказал пользователь. "
                "Без кавычек, без пояснений, на том же языке."
            ),
        ],
        config=types.GenerateContentConfig(temperature=0.1),
    )
    text = (getattr(resp, "text", None) or "").strip()
    return text


async def gemini_reply_with_tools(user_text: str) -> str:
    """
    Text chat with sliding history + optional Tavily via function calling.
    Kyiv time is always in system_instruction (no shell date, no extra model step).
    """
    assert _gemini is not None
    kyiv = kyiv_now_str()
    sys_inst = system_instruction(kyiv)

    async with _history_lock:
        hist_snapshot = list(_history)

    contents: list[types.Content] = []
    for msg in hist_snapshot:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["text"])])
        )
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
    )

    config = types.GenerateContentConfig(
        system_instruction=sys_inst,
        tools=[SEARCH_TOOL] if TAVILY_API_KEY else None,
        temperature=0.7,
    )

    # Manual tool loop (max 4 rounds)
    for round_i in range(4):
        resp = await asyncio.to_thread(
            _gemini.models.generate_content,
            model=MODEL,
            contents=contents,
            config=config,
        )
        calls = _extract_function_calls(resp)
        if not calls:
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return text
            # empty — break
            break

        model_content = _model_content_from_response(resp)
        if model_content is not None:
            contents.append(model_content)
        else:
            # synthesize model turn with function calls
            parts = []
            for name, args, _cid in calls:
                parts.append(types.Part.from_function_call(name=name, args=args))
            contents.append(types.Content(role="model", parts=parts))

        fr_parts = []
        for name, args, cid in calls:
            log.info("tool call round=%d name=%s args=%s", round_i, name, args)
            if name == "internet_search":
                q = str(args.get("query") or user_text)
                result = await tavily_search(q)
            else:
                result = f"Неизвестный инструмент: {name}"
            fr_parts.append(
                types.Part.from_function_response(
                    name=name,
                    response={"result": result},
                )
            )
        contents.append(types.Content(role="user", parts=fr_parts))

    return "Не удалось сформировать ответ, попробуй ещё раз."


async def history_add(user_text: str, model_text: str) -> None:
    async with _history_lock:
        _history.append({"role": "user", "text": user_text})
        _history.append({"role": "model", "text": model_text})
        # deque maxlen already enforces HISTORY_MAX on total messages


async def history_clear() -> None:
    async with _history_lock:
        _history.clear()


async def edge_mp3_chunks(text: str, voice: str):
    communicate = edge_tts.Communicate(text, voice)
    total = 0
    t0 = time.time()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio" and chunk.get("data"):
            data = chunk["data"]
            total += len(data)
            yield data
    log.info("edge-tts done: %dB in %.1fs voice=%s", total, time.time() - t0, voice)


async def edge_mp3_all(text: str, voice: str) -> bytes:
    parts = []
    async for b in edge_mp3_chunks(text, voice):
        parts.append(b)
    return b"".join(parts)


async def process_voice(pcm: bytes, rate: int) -> str:
    """PCM -> transcript -> Gemini(+tools, history) -> reply text."""
    if _gemini is None:
        raise RuntimeError("GEMINI_API_KEY not configured")
    wav = pcm_to_wav(pcm, rate)
    t0 = time.time()
    user_text = await gemini_transcribe(wav)
    if not user_text:
        user_text = ""
        log.warning("empty transcript")
    log.info("STT (%.1fs): %s", time.time() - t0, user_text[:160])
    if not user_text.strip():
        return "Я не расслышал, повтори пожалуйста."

    t1 = time.time()
    reply = await gemini_reply_with_tools(user_text)
    if not reply:
        reply = "Я не расслышал, повтори пожалуйста."
    log.info("REPLY (%.1fs): %s", time.time() - t1, reply[:160])
    await history_add(user_text, reply)
    return reply


# ===================== endpoints =====================

@app.get("/")
async def root():
    return PlainTextResponse(
        "VoiceAssist ok. /health /tts /pcm /chat /stream /speedtest /history\n"
    )


@app.get("/health")
async def health():
    async with _history_lock:
        hlen = len(_history)
    return {
        "status": "ok",
        "model": MODEL,
        "key_set": bool(API_KEY),
        "tavily": bool(TAVILY_API_KEY),
        "tts": "edge-tts",
        "default_voice": DEFAULT_VOICE,
        "history_len": hlen,
        "history_max": HISTORY_MAX,
        "kyiv_now": kyiv_now_str(),
    }


@app.get("/history")
async def history_get():
    async with _history_lock:
        items = list(_history)
    return {"count": len(items), "max": HISTORY_MAX, "messages": items}


@app.delete("/history")
async def history_delete():
    await history_clear()
    return {"ok": True}


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
    use_voice = pick_voice(voice)
    mp3 = await edge_mp3_all(text, use_voice)
    return Response(content=mp3, media_type="audio/mpeg")


@app.post("/chat")
async def chat(
    request: Request,
    rate: int = Query(16000),
    voice: str = Query(None),
):
    body = await request.body()
    if len(body) < int(rate * 2 * 0.3):
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    use_voice = pick_voice(voice)
    try:
        text = await process_voice(body, rate)
    except Exception as e:
        log.exception("chat pipeline")
        return Response(status_code=502, content=f"error: {e}")
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
    body = await request.body()
    if len(body) < int(rate * 2 * 0.3):
        return Response(status_code=400, content=f"audio too short ({len(body)} bytes)")
    if _gemini is None:
        return Response(status_code=500, content="GEMINI_API_KEY not configured")

    use_voice = pick_voice(voice)
    t0 = time.time()
    try:
        text = await process_voice(body, rate)
    except Exception as e:
        log.exception("stream pipeline")
        return Response(status_code=502, content=f"error: {e}")

    log.info("pipeline total %.1fs", time.time() - t0)

    async def gen():
        try:
            async for data in edge_mp3_chunks(text, use_voice):
                yield data
        except Exception:
            log.exception("edge-tts stream error")

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
        while remaining > 0:
            n = 65536 if remaining > 65536 else remaining
            yield chunk[:n]
            remaining -= n

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-store",
            "Content-Length": str(size),
        },
    )
