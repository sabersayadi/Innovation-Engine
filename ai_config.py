"""Provider-agnostic AI configuration: OpenAI, Gemini, Claude, OpenAI-compatible endpoints, and web search settings.

Settings are loaded from .env and ai_config.json (in the project root). Runtime updates from the UI
are persisted to ai_config.json so switching models, providers, or search backends works immediately
across both apps without restarting.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
CONFIG_FILE = ROOT / "ai_config.json"

PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "o3", "o4-mini"],
        "requires_key": True,
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GEMINI_API_KEY",
        "alt_env_keys": ["GOOGLE_API_KEY"],
        "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
        "requires_key": True,
    },
    "claude": {
        "label": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "models": ["claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805", "claude-haiku-4-5-20251001"],
        "requires_key": True,
    },
    "openai_compatible": {
        "label": "OpenAI-Compatible Endpoint",
        "base_url": "http://localhost:11434/v1",
        "env_key": "OPENAI_COMPAT_API_KEY",
        "models": ["llama3.3", "qwen2.5:32b", "deepseek-chat", "mistral"],
        "requires_key": False,
    },
}

COMPAT_TEMPLATES = [
    {"id": "ollama", "name": "Ollama (Local)", "base_url": "http://localhost:11434/v1", "model": "qwen2.5:32b"},
    {"id": "lmstudio", "name": "LM Studio (Local)", "base_url": "http://localhost:1234/v1", "model": "local-model"},
    {"id": "vllm", "name": "vLLM (Local)", "base_url": "http://localhost:8000/v1", "model": "Qwen/Qwen2.5-32B-Instruct"},
    {"id": "openrouter", "name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4.1"},
    {"id": "groq", "name": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    {"id": "together", "name": "Together AI", "base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    {"id": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
]

SEARCH_BACKENDS = {
    "auto": "Auto (Tavily/Serper/Brave -> SearXNG -> Provider-Native -> Fallback)",
    "tavily": "Tavily API",
    "serper": "Serper Google SERP API",
    "brave": "Brave Search API",
    "searxng": "Local SearXNG",
    "provider_native": "Provider-Native Search (Gemini / OpenAI / Claude)",
    "fallback": "Basic Fallback (Wikipedia + DDG Instant Answer)",
}

ROLES = {
    "disclosure": "Disclosure analysis agents",
    "chat": "Interactive assistant chat",
    "library": "Sector library updates",
    "profiles": "Company & person profiles",
    "research": "Missed research analysis",
    "summary": "Patent similarity summaries",
    "extractor": "WebFetch page distillation (fast model)",
    "firewall_secure": "Firewall secure zone",
    "firewall_research": "Firewall research zone",
}

DEFAULT_CONFIG = {
    "active": {
        "provider": "openai",
        "model": "gpt-4.1",
        "base_url": "",
        "endpoint_id": "",
    },
    "keys": {
        "openai": "",
        "gemini": "",
        "claude": "",
        "openai_compatible": "",
        "tavily": "",
        "serper": "",
        "brave": "",
        "jina": "",
    },
    "search": {
        "backend": "auto",
        "searxng_url": "",
        "jina_fallback": True,
    },
    "custom_endpoints": [],
    "overrides": {},
}


def read_env() -> dict[str, str]:
    """Read .env in ROOT plus process environment variables (process env wins over .env)."""
    values: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            values[key.strip()] = val.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if v})
    return values


def setting(name: str, default: str | None = None) -> str | None:
    return read_env().get(name) or default


def load_config() -> dict:
    """Load ai_config.json merged with defaults and environment-based initial active provider."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text())
            if isinstance(saved, dict):
                cfg["active"].update(saved.get("active") or {})
                cfg["keys"].update(saved.get("keys") or {})
                cfg["search"].update(saved.get("search") or {})
                cfg["custom_endpoints"] = list(saved.get("custom_endpoints") or [])
                cfg["overrides"] = dict(saved.get("overrides") or {})
        except Exception:
            pass
    else:
        env = read_env()
        if env.get("OPENAI_COMPAT_BASE_URL"):
            cfg["active"] = {
                "provider": "openai_compatible",
                "model": env.get("OPENAI_COMPAT_MODEL", "qwen2.5:32b"),
                "base_url": env["OPENAI_COMPAT_BASE_URL"],
                "endpoint_id": "",
            }
        elif env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
            cfg["active"] = {"provider": "gemini", "model": "gemini-2.5-flash", "base_url": "", "endpoint_id": ""}
        elif env.get("ANTHROPIC_API_KEY"):
            cfg["active"] = {"provider": "claude", "model": "claude-sonnet-4-5-20250929", "base_url": "", "endpoint_id": ""}
        elif env.get("OPENAI_API_KEY"):
            cfg["active"] = {"provider": "openai", "model": "gpt-4.1", "base_url": "", "endpoint_id": ""}
    return cfg


def save_config(cfg: dict) -> dict:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    return cfg


def mask_key(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def get_api_key(provider: str, cfg: dict | None = None, endpoint_id: str | None = None) -> str:
    """Resolve API key for a provider or custom endpoint (saved config first, then .env/env)."""
    cfg = cfg or load_config()
    if provider == "openai_compatible" and endpoint_id:
        for ep in cfg.get("custom_endpoints", []):
            if ep.get("id") == endpoint_id and ep.get("api_key"):
                return ep["api_key"]
    key = (cfg.get("keys") or {}).get(provider) or ""
    if key:
        return key
    spec = PROVIDERS.get(provider, {})
    env = read_env()
    for env_name in [spec.get("env_key"), *(spec.get("alt_env_keys") or [])]:
        if env_name and env.get(env_name):
            return env[env_name]
    return ""


def get_search_key(name: str, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    val = (cfg.get("keys") or {}).get(name)
    if val:
        return val
    return setting(f"{name.upper()}_API_KEY") or ""


def get_searxng_url(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    url = ((cfg.get("search") or {}).get("searxng_url") or setting("SEARXNG_URL") or "").strip().rstrip("/")
    return url


def resolve_search_backend(cfg: dict | None = None) -> dict:
    """Return the effective WebSearch backend and human-readable status."""
    cfg = cfg or load_config()
    pref = ((cfg.get("search") or {}).get("backend") or "auto").strip()
    searx_url = get_searxng_url(cfg)

    def native_provider():
        active_prov = (cfg.get("active") or {}).get("provider")
        order = [active_prov] if active_prov in ("gemini", "openai", "claude") else []
        for p in ("gemini", "openai", "claude"):
            if p not in order:
                order.append(p)
        for p in order:
            if get_api_key(p, cfg):
                return p
        return None

    if pref == "tavily" and get_search_key("tavily", cfg):
        return {"mode": "tavily", "label": "Tavily API", "full_web": True}
    if pref == "serper" and get_search_key("serper", cfg):
        return {"mode": "serper", "label": "Serper Google SERP", "full_web": True}
    if pref == "brave" and get_search_key("brave", cfg):
        return {"mode": "brave", "label": "Brave Search API", "full_web": True}
    if pref == "searxng" and searx_url:
        return {"mode": "searxng", "label": f"SearXNG ({searx_url})", "full_web": True}
    if pref == "provider_native":
        nat = native_provider()
        if nat:
            return {"mode": "provider_native", "provider": nat, "label": f"Provider-Native ({PROVIDERS[nat]['label']})", "full_web": True}
    if pref == "fallback":
        return {"mode": "fallback", "label": "Basic Fallback (Wikipedia + DDG Instant Answer)", "full_web": False}

    # Auto cascade
    for name, label in (("tavily", "Tavily API"), ("serper", "Serper Google SERP"), ("brave", "Brave Search API")):
        if get_search_key(name, cfg):
            return {"mode": name, "label": label, "full_web": True}
    if searx_url:
        return {"mode": "searxng", "label": f"SearXNG ({searx_url})", "full_web": True}
    nat = native_provider()
    if nat:
        return {"mode": "provider_native", "provider": nat, "label": f"Provider-Native ({PROVIDERS[nat]['label']})", "full_web": True}
    return {"mode": "fallback", "label": "Basic Fallback (Wikipedia + DDG Instant Answer)", "full_web": False}


def resolve(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    endpoint_id: str | None = None,
    role: str | None = None,
) -> dict:
    """Resolve the effective {provider, model, base_url, api_key, endpoint_id} for an agent call."""
    cfg = load_config()
    active = dict(cfg["active"])
    if role and role in cfg.get("overrides", {}):
        ov = cfg["overrides"][role] or {}
        if ov.get("provider"):
            active["provider"] = ov["provider"]
        if ov.get("model"):
            active["model"] = ov["model"]
        if "base_url" in ov and ov["base_url"] is not None:
            active["base_url"] = ov["base_url"]
        if "endpoint_id" in ov and ov["endpoint_id"] is not None:
            active["endpoint_id"] = ov["endpoint_id"]

    eff_provider = (provider or active.get("provider") or "openai").strip()
    if eff_provider not in PROVIDERS:
        raise ValueError(f"Unknown AI provider '{eff_provider}'. Choose from: {', '.join(PROVIDERS)}")

    eff_endpoint_id = endpoint_id if endpoint_id is not None else active.get("endpoint_id", "")
    matched_ep = next((ep for ep in cfg.get("custom_endpoints", []) if ep.get("id") == eff_endpoint_id), None)

    eff_model = (model or (matched_ep.get("model") if matched_ep and not model and eff_provider == "openai_compatible" else None)
                 or active.get("model") or PROVIDERS[eff_provider]["models"][0]).strip()

    eff_base_url = (
        base_url
        or (matched_ep.get("base_url") if matched_ep and eff_provider == "openai_compatible" else "")
        or (active.get("base_url") if eff_provider == active.get("provider") else "")
        or (setting("OPENAI_COMPAT_BASE_URL") if eff_provider == "openai_compatible" else "")
        or PROVIDERS[eff_provider]["base_url"]
    ).strip().rstrip("/")

    eff_key = api_key if api_key is not None else get_api_key(eff_provider, cfg, eff_endpoint_id)
    return {
        "provider": eff_provider,
        "model": eff_model,
        "base_url": eff_base_url,
        "api_key": eff_key,
        "endpoint_id": eff_endpoint_id,
    }


def public_config() -> dict:
    """Return UI-safe configuration with masked API keys."""
    cfg = load_config()
    key_status = {}
    for name in ("openai", "gemini", "claude", "openai_compatible", "tavily", "serper", "brave", "jina"):
        raw = (cfg.get("keys") or {}).get(name) or ""
        env_raw = ""
        if name in PROVIDERS:
            spec = PROVIDERS[name]
            env = read_env()
            for k in [spec.get("env_key"), *(spec.get("alt_env_keys") or [])]:
                if k and env.get(k):
                    env_raw = env[k]
                    break
        else:
            env_raw = setting(f"{name.upper()}_API_KEY") or ""
        effective = raw or env_raw
        key_status[name] = {
            "configured": bool(effective),
            "masked": mask_key(effective),
            "source": "ui" if raw else ("env" if env_raw else "none"),
        }

    endpoints = []
    for ep in cfg.get("custom_endpoints", []):
        endpoints.append({
            "id": ep.get("id", ""),
            "name": ep.get("name", ""),
            "base_url": ep.get("base_url", ""),
            "model": ep.get("model", ""),
            "models": ep.get("models", []),
            "has_key": bool(ep.get("api_key")),
            "masked_key": mask_key(ep.get("api_key")),
        })

    search_cfg = {
        "backend": (cfg.get("search") or {}).get("backend", "auto"),
        "searxng_url": get_searxng_url(cfg),
        "jina_fallback": bool((cfg.get("search") or {}).get("jina_fallback", True)),
    }

    return {
        "active": cfg["active"],
        "providers": PROVIDERS,
        "compat_templates": COMPAT_TEMPLATES,
        "search_backends": SEARCH_BACKENDS,
        "search": search_cfg,
        "search_status": resolve_search_backend(cfg),
        "roles": ROLES,
        "keys": key_status,
        "custom_endpoints": endpoints,
        "overrides": cfg.get("overrides", {}),
    }


def update_config(payload: dict) -> dict:
    """Merge partial config updates from the API/UI into ai_config.json without exposing or clobbering keys."""
    cfg = load_config()
    if "active" in payload and isinstance(payload["active"], dict):
        act = payload["active"]
        provider = act.get("provider", cfg["active"]["provider"])
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        cfg["active"]["provider"] = provider
        if act.get("model"):
            cfg["active"]["model"] = act["model"].strip()
        if "base_url" in act:
            cfg["active"]["base_url"] = (act["base_url"] or "").strip().rstrip("/")
        if "endpoint_id" in act:
            cfg["active"]["endpoint_id"] = (act["endpoint_id"] or "").strip()

    if "keys" in payload and isinstance(payload["keys"], dict):
        for name, value in payload["keys"].items():
            if name not in cfg["keys"] or value is None:
                continue
            cleaned = value.strip()
            if cleaned == "__CLEAR__":
                cfg["keys"][name] = ""
            elif cleaned and "..." not in cleaned and set(cleaned) != {"*"}:
                cfg["keys"][name] = cleaned

    if "search" in payload and isinstance(payload["search"], dict):
        sr = payload["search"]
        if "backend" in sr:
            backend = (sr["backend"] or "auto").strip()
            if backend not in SEARCH_BACKENDS:
                raise ValueError(f"Unknown search backend: {backend}")
            cfg["search"]["backend"] = backend
        if "searxng_url" in sr:
            cfg["search"]["searxng_url"] = (sr["searxng_url"] or "").strip().rstrip("/")
        if "jina_fallback" in sr:
            cfg["search"]["jina_fallback"] = bool(sr["jina_fallback"])

    if "custom_endpoints" in payload and isinstance(payload["custom_endpoints"], list):
        existing = {ep.get("id"): ep for ep in cfg.get("custom_endpoints", [])}
        updated_eps = []
        for ep in payload["custom_endpoints"]:
            ep_id = (ep.get("id") or "").strip()
            if not ep_id:
                continue
            prev_key = existing.get(ep_id, {}).get("api_key", "")
            raw_key = (ep.get("api_key") or "").strip()
            if raw_key == "__CLEAR__":
                ep_key = ""
            elif raw_key and "..." not in raw_key and set(raw_key) != {"*"}:
                ep_key = raw_key
            else:
                ep_key = prev_key
            updated_eps.append({
                "id": ep_id,
                "name": (ep.get("name") or ep_id).strip(),
                "base_url": (ep.get("base_url") or "").strip().rstrip("/"),
                "model": (ep.get("model") or "").strip(),
                "models": [m.strip() for m in (ep.get("models") or []) if m and m.strip()],
                "api_key": ep_key,
            })
        cfg["custom_endpoints"] = updated_eps

    if "overrides" in payload and isinstance(payload["overrides"], dict):
        cleaned_ov = {}
        for role, ov in payload["overrides"].items():
            if role not in ROLES or not isinstance(ov, dict):
                continue
            if ov.get("provider") or ov.get("model"):
                cleaned_ov[role] = {
                    "provider": ov.get("provider", ""),
                    "model": (ov.get("model") or "").strip(),
                    "base_url": (ov.get("base_url") or "").strip().rstrip("/"),
                    "endpoint_id": (ov.get("endpoint_id") or "").strip(),
                }
        cfg["overrides"] = cleaned_ov

    save_config(cfg)
    return public_config()
