"""
LLM factory for agent-core.

Priority:
  1) AgentRouter (OpenAI-compatible gateway) — AGENTROUTER_API_KEY
  2) Gemini on Vertex AI via service account
  3) GEMINI_API_KEY / OpenAI / Anthropic direct

AgentRouter docs: https://docs.agentrouter.org/ — base URL https://agentrouter.org/v1
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_path = _repo_root() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        # Prefer .env values for LLM routing knobs so local config wins
        key = k.strip()
        val = v.strip()
        if key in {
            "LLM_PROVIDER",
            "AGENTROUTER_API_KEY",
            "AGENTROUTER_BASE_URL",
            "AGENTROUTER_MODEL",
            "GEMINI_MODEL",
        }:
            os.environ[key] = val
        else:
            os.environ.setdefault(key, val)


def _resolve_credentials() -> None:
    cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not cred:
        return
    path = Path(cred)
    if not path.is_absolute():
        path = _repo_root() / path
    if path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path.resolve())
        if not os.getenv("GCP_PROJECT_ID"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("project_id"):
                os.environ["GCP_PROJECT_ID"] = data["project_id"]


def _agentrouter_llm():
    """OpenAI-compatible ChatOpenAI pointed at AgentRouter gateway."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("AGENTROUTER_API_KEY") or os.getenv("AGENT_ROUTER_TOKEN")
    if not api_key:
        raise RuntimeError("AGENTROUTER_API_KEY not set")
    base_url = os.getenv("AGENTROUTER_BASE_URL", "https://agentrouter.org/v1")
    model = os.getenv("AGENTROUTER_MODEL", "claude-sonnet-4-5-20250929")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        timeout=90,
        max_retries=2,
    )


def _gemini_llm():
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    project = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    api_key = os.getenv("GEMINI_API_KEY")

    if creds and Path(creds).exists() and project:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model_name,
                project=project,
                location=location,
                vertexai=True,
                temperature=0.2,
            )
        except Exception:
            from langchain_google_vertexai import ChatVertexAI

            return ChatVertexAI(
                model_name=model_name,
                project=project,
                location=location,
                temperature=0.2,
            )

    if api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
        )

    raise RuntimeError("Gemini not configured")


_CACHED_LLM = None
_CACHED_PROVIDER = None


def get_llm():
    """Return a LangChain chat model. AgentRouter preferred; Gemini is fallback."""
    global _CACHED_LLM, _CACHED_PROVIDER
    if _CACHED_LLM is not None:
        return _CACHED_LLM

    _load_dotenv()
    _resolve_credentials()
    provider = os.getenv("LLM_PROVIDER", "agentrouter").lower()
    errors: list[str] = []

    def _try_agentrouter():
        llm = _agentrouter_llm()
        if os.getenv("LLM_SKIP_PROBE", "").lower() in ("1", "true", "yes"):
            return llm
        llm.invoke("Reply with OK")
        return llm

    if provider in ("agentrouter", "agent_router", "router", "auto"):
        try:
            _CACHED_LLM = _try_agentrouter()
            _CACHED_PROVIDER = "agentrouter"
            return _CACHED_LLM
        except Exception as exc:
            errors.append(f"agentrouter: {exc}")
            try:
                _CACHED_LLM = _gemini_llm()
                _CACHED_PROVIDER = "gemini_fallback"
                return _CACHED_LLM
            except Exception as exc2:
                errors.append(f"gemini: {exc2}")
                raise RuntimeError("All LLM providers failed: " + " | ".join(errors)) from exc2

    if provider == "gemini":
        _CACHED_LLM = _gemini_llm()
        _CACHED_PROVIDER = "gemini"
        return _CACHED_LLM

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AGENTROUTER_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv(
            "AGENTROUTER_BASE_URL", "https://agentrouter.org/v1"
        )
        model = os.getenv("OPENAI_MODEL") or os.getenv("AGENTROUTER_MODEL", "gpt-4o")
        _CACHED_LLM = ChatOpenAI(
            model=model, api_key=api_key, base_url=base_url, temperature=0.2
        )
        _CACHED_PROVIDER = "openai"
        return _CACHED_LLM

    if provider == "anthropic":
        if os.getenv("AGENTROUTER_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            _CACHED_LLM = _agentrouter_llm()
            _CACHED_PROVIDER = "agentrouter"
            return _CACHED_LLM
        from langchain_anthropic import ChatAnthropic

        _CACHED_LLM = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.2)
        _CACHED_PROVIDER = "anthropic"
        return _CACHED_LLM

    if os.getenv("AGENTROUTER_API_KEY"):
        try:
            _CACHED_LLM = _try_agentrouter()
            _CACHED_PROVIDER = "agentrouter"
            return _CACHED_LLM
        except Exception as exc:
            errors.append(str(exc))
    _CACHED_LLM = _gemini_llm()
    _CACHED_PROVIDER = "gemini"
    return _CACHED_LLM


def active_provider() -> str | None:
    return _CACHED_PROVIDER
