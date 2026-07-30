"""
STT + translation (Day 6) — Whisper STT + NLLB / OPUS translation.

Priority:
  STT: OpenAI Whisper API → local faster-whisper (tiny/base) → degrade
  Translate: NLLB-200-distilled → Helsinki OPUS-MT → LLM gateway → heuristic
"""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field

app = FastAPI(title="Lankara STT Translation", version="1.0.0")

# NLLB language codes
NLLB_LANG = {
    "en": "eng_Latn",
    "si": "sin_Sinh",
    "ta": "tam_Taml",
}


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parents[3]
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _envelope(tool: str, status: str, confidence: float, data, data_slice: str, error_reason=None):
    return {
        "tool": tool,
        "status": status,
        "confidence": confidence,
        "data": data,
        "data_slice": data_slice,
        "error_reason": error_reason,
    }


@lru_cache(maxsize=1)
def _whisper_model():
    """Lazy-load local faster-whisper (CPU)."""
    model_size = os.getenv("WHISPER_MODEL", "tiny")
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


@lru_cache(maxsize=1)
def _nllb_pipeline():
    """Lazy-load NLLB distilled for SI/TA/EN."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    model_name = os.getenv(
        "NLLB_MODEL", "facebook/nllb-200-distilled-600M"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model, pipeline


def _translate_nllb(text: str, source: str, target: str) -> str | None:
    try:
        tokenizer, model, _ = _nllb_pipeline()
        src = NLLB_LANG.get(source or "en", "eng_Latn")
        tgt = NLLB_LANG.get(target, "eng_Latn")
        tokenizer.src_lang = src
        inputs = tokenizer(text, return_tensors="pt")
        forced = tokenizer.convert_tokens_to_ids(tgt)
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced,
            max_length=256,
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception:
        return None


def _translate_opus(text: str, source: str, target: str) -> str | None:
    """Smaller Helsinki OPUS models when available for the pair."""
    pair_map = {
        ("en", "si"): "Helsinki-NLP/opus-mt-en-mul",
        ("si", "en"): "Helsinki-NLP/opus-mt-mul-en",
        ("en", "ta"): "Helsinki-NLP/opus-mt-en-dra",
        ("ta", "en"): "Helsinki-NLP/opus-mt-dra-en",
        ("en", "en"): None,
    }
    model_name = pair_map.get((source or "en", target))
    if not model_name:
        if source == target:
            return text
        return None
    try:
        from transformers import pipeline

        pipe = pipeline("translation", model=model_name)
        out = pipe(text, max_length=256)
        return out[0]["translation_text"]
    except Exception:
        return None


def _llm_translate(text: str, target_lang: str) -> str | None:
    _load_dotenv()
    lang_name = {"en": "English", "si": "Sinhala", "ta": "Tamil"}.get(target_lang, target_lang)
    prompt = (
        f"Translate the following text to {lang_name}. "
        f"Return ONLY the translation, no quotes or commentary.\n\n{text}"
    )
    ar_key = os.getenv("AGENTROUTER_API_KEY")
    ar_base = os.getenv("AGENTROUTER_BASE_URL", "https://agentrouter.org/v1")
    ar_model = os.getenv("AGENTROUTER_MODEL", "claude-sonnet-4-5-20250929")
    if ar_key:
        try:
            r = httpx.post(
                f"{ar_base}/chat/completions",
                headers={"Authorization": f"Bearer {ar_key}"},
                json={
                    "model": ar_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                },
                timeout=45.0,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
    return None


_PHRASES = {
    ("si", "en"): {
        "විකුණුම් අඩුයි": "Sales look weak",
        "නැවත ඇණවුම්": "reorder",
    },
    ("ta", "en"): {
        "விற்பனை குறைவு": "Sales look weak",
        "மறு ஆர்டர்": "reorder",
    },
}


def _heuristic_translate(text: str, target_lang: str, source_lang: str | None) -> str:
    src = (source_lang or "auto").lower()
    table = _PHRASES.get((src, target_lang)) or {}
    for k, v in table.items():
        if k in text:
            return text.replace(k, v)
    if target_lang == "en":
        return text
    return f"[{target_lang}] {text}"


def _detect_lang(text: str) -> str:
    # Very light heuristic for SI/TA script ranges
    for ch in text:
        o = ord(ch)
        if 0x0D80 <= o <= 0x0DFF:
            return "si"
        if 0x0B80 <= o <= 0x0BFF:
            return "ta"
    return "en"


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = Field(default="en", description="en | si | ta")
    source_lang: str | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "stt-translation",
        "ready": True,
        "stt": "openai-whisper-api | faster-whisper | browser",
        "translate": "nllb | opus | llm | heuristic",
        "whisper_model": os.getenv("WHISPER_MODEL", "tiny"),
        "nllb_model": os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M"),
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile | None = File(default=None),
    language: str | None = Form(default=None),
):
    _load_dotenv()
    if audio is None:
        return _envelope(
            "stt",
            "degraded",
            0.2,
            {"text": "", "language": language},
            "no audio",
            "Upload audio or use frontend Web Speech API",
        )

    content = await audio.read()
    openai_key = os.getenv("OPENAI_API_KEY")

    # 1) OpenAI Whisper API
    if openai_key:
        try:
            files = {
                "file": (
                    audio.filename or "audio.webm",
                    content,
                    audio.content_type or "audio/webm",
                )
            }
            data = {"model": "whisper-1"}
            if language:
                data["language"] = language
            r = httpx.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {openai_key}"},
                files=files,
                data=data,
                timeout=90.0,
            )
            r.raise_for_status()
            text = r.json().get("text", "")
            return _envelope(
                "stt",
                "ok",
                0.9,
                {"text": text, "language": language or "auto", "engine": "openai-whisper"},
                "whisper-api",
            )
        except Exception as exc:
            api_err = str(exc)
        else:
            api_err = None
    else:
        api_err = "no OPENAI_API_KEY"

    # 2) Local faster-whisper
    try:
        suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        model = _whisper_model()
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=1,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        Path(tmp_path).unlink(missing_ok=True)
        return _envelope(
            "stt",
            "ok",
            0.8,
            {
                "text": text,
                "language": getattr(info, "language", language) or "auto",
                "engine": f"faster-whisper:{os.getenv('WHISPER_MODEL', 'tiny')}",
            },
            "faster-whisper-local",
        )
    except Exception as exc:
        return _envelope(
            "stt",
            "degraded",
            0.25,
            {
                "text": "",
                "language": language,
                "bytes_received": len(content),
                "hint": "Install faster-whisper or set OPENAI_API_KEY; browser mic still works",
            },
            "local whisper failed",
            f"api={api_err}; local={exc}",
        )


@app.post("/translate")
def translate(body: TranslateRequest):
    _load_dotenv()
    src = body.source_lang or _detect_lang(body.text)
    tgt = body.target_lang

    if src == tgt:
        return _envelope(
            "translate",
            "ok",
            0.95,
            {
                "source_text": body.text,
                "source_lang": src,
                "target_lang": tgt,
                "translated_text": body.text,
            },
            "identity",
        )

    # Prefer NLLB (plan Day 6)
    out = _translate_nllb(body.text, src, tgt)
    if out:
        return _envelope(
            "translate",
            "ok",
            0.88,
            {
                "source_text": body.text,
                "source_lang": src,
                "target_lang": tgt,
                "translated_text": out,
                "engine": "nllb",
            },
            f"nllb {src}->{tgt}",
        )

    out = _translate_opus(body.text, src, tgt)
    if out:
        return _envelope(
            "translate",
            "ok",
            0.8,
            {
                "source_text": body.text,
                "source_lang": src,
                "target_lang": tgt,
                "translated_text": out,
                "engine": "opus-mt",
            },
            f"opus {src}->{tgt}",
        )

    out = _llm_translate(body.text, tgt)
    if out:
        return _envelope(
            "translate",
            "ok",
            0.75,
            {
                "source_text": body.text,
                "source_lang": src,
                "target_lang": tgt,
                "translated_text": out,
                "engine": "llm",
            },
            f"llm translate -> {tgt}",
        )

    heuristic = _heuristic_translate(body.text, tgt, src)
    return _envelope(
        "translate",
        "degraded",
        0.4,
        {
            "source_text": body.text,
            "source_lang": src,
            "target_lang": tgt,
            "translated_text": heuristic,
            "engine": "heuristic",
        },
        "heuristic/pass-through translate",
        "NLLB/OPUS/LLM unavailable",
    )
