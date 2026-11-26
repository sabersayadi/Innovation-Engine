"""Free data sources plus provider-agnostic WebSearch, WebFetch, and scoped file tools.

To give an agent a tool, add the tool's name to the 'tools:' line at the top of its file in agents/.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import html
import io
import json
import mimetypes
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import uuid
from datetime import date
from pathlib import Path

import httpx
import trafilatura
from pypdf import PdfReader

import ai_config
import epo
from llm import ToolDef


def tool(name, description, schema, read_only=True):
    """Decorator turning an async function into a provider-agnostic ToolDef."""
    def decorator(fn):
        return ToolDef(name=name, description=description, parameters=schema, fn=fn, read_only=read_only)
    return decorator


def _get(url, params=None, data=None, headers=None):
    """Download a URL and return the body as text."""
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "Mozilla/5.0 (innovation-agents)", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


async def _fetch(*args, **kwargs):
    return await asyncio.to_thread(_get, *args, **kwargs)


def _reply(text):
    return {"content": [{"type": "text", "text": text}]}


def _clean(fragment):
    """HTML to plain text on one line."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment or ""))).strip()


def _html_to_text(raw_html: str) -> str:
    """Convert an HTML page to readable plain text without scripts/styles."""
    cleaned = re.sub(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I)
    cleaned = re.sub(r"</(p|div|li|tr|h[1-6]|br|section|article)>", "\n", cleaned, flags=re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    lines = [re.sub(r"[ \t]+", " ", html.unescape(line)).strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line)


# ---------- Short-TTL cache for web search & page fetch ----------

_WEB_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL_SECONDS = 1800
_CACHE_MAX_ENTRIES = 300


def _cache_get(key: str) -> str | None:
    entry = _WEB_CACHE.get(key)
    if not entry:
        return None
    ts, val = entry
    if time.monotonic() - ts > _CACHE_TTL_SECONDS:
        _WEB_CACHE.pop(key, None)
        return None
    return val


def _cache_set(key: str, val: str) -> None:
    if len(_WEB_CACHE) >= _CACHE_MAX_ENTRIES:
        oldest = min(_WEB_CACHE, key=lambda k: _WEB_CACHE[k][0])
        _WEB_CACHE.pop(oldest, None)
    _WEB_CACHE[key] = (time.monotonic(), val)


# ---------- Web search & fetch (multi-backend cascade) ----------

async def _search_tavily(client: httpx.AsyncClient, q: str, api_key: str) -> str:
    resp = await client.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": q, "max_results": 8, "include_answer": True},
    )
    resp.raise_for_status()
    data = resp.json()
    lines = [data["answer"]] if data.get("answer") else []
    for r in data.get("results", []):
        lines.append(f"- {r.get('title')} ({r.get('url')})\n  {r.get('content', '')[:450]}")
    return "\n".join(lines) or "No results found."


async def _search_serper(client: httpx.AsyncClient, q: str, api_key: str) -> str:
    resp = await client.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": q, "num": 8},
    )
    resp.raise_for_status()
    data = resp.json()
    lines = []
    if data.get("answerBox") and data["answerBox"].get("snippet"):
        lines.append(data["answerBox"]["snippet"])
    for r in data.get("organic", []):
        lines.append(f"- {r.get('title')} ({r.get('link')})\n  {r.get('snippet', '')}")
    return "\n".join(lines) or "No results found."


async def _search_brave(client: httpx.AsyncClient, q: str, api_key: str) -> str:
    resp = await client.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        params={"q": q, "count": 8},
    )
    resp.raise_for_status()
    data = resp.json()
    lines = [
        f"- {r.get('title')} ({r.get('url')})\n  {r.get('description', '')}"
        for r in (data.get("web") or {}).get("results", [])
    ]
    return "\n".join(lines) or "No results found."


async def _search_searxng(client: httpx.AsyncClient, q: str, base_url: str) -> str:
    resp = await client.get(
        f"{base_url}/search",
        params={"q": q, "format": "json", "language": "en"},
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    lines = []
    for r in (data.get("results") or [])[:10]:
        engine = f" [{r['engine']}]" if r.get("engine") else ""
        lines.append(f"- {r.get('title')}{engine} ({r.get('url')})\n  {_clean(r.get('content', ''))[:400]}")
    return "\n".join(lines) or "No results found."


async def _search_provider_native(client: httpx.AsyncClient, q: str, provider: str, cfg: dict) -> str:
    """Use native live web search built into Gemini, OpenAI, or Claude when one of those keys is present."""
    api_key = ai_config.get_api_key(provider, cfg)
    if not api_key:
        raise RuntimeError(f"No API key for {provider}")

    if provider == "gemini":
        active = cfg.get("active") or {}
        model = (active.get("model") if active.get("provider") == "gemini" else "gemini-2.5-flash").removeprefix("models/")
        base = ai_config.PROVIDERS["gemini"]["base_url"]
        resp = await client.post(
            f"{base}/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": f"Search the web and give key facts with sources for: {q}"}]}],
                "tools": [{"google_search": {}}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        summary = "\n".join(p.get("text", "") for p in parts if not p.get("thought") and p.get("text")).strip()
        chunks = ((cand.get("groundingMetadata") or {}).get("groundingChunks") or [])
        links = [
            f"- {c['web'].get('title') or c['web'].get('uri')} ({c['web'].get('uri')})"
            for c in chunks
            if isinstance(c.get("web"), dict) and c["web"].get("uri")
        ]
        out = [summary] if summary else []
        if links:
            out.append("Sources:\n" + "\n".join(dict.fromkeys(links)))
        return "\n\n".join(out) or "No results found."

    if provider == "openai":
        base = ai_config.PROVIDERS["openai"]["base_url"]
        resp = await client.post(
            f"{base}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4.1-mini",
                "tools": [{"type": "web_search"}],
                "input": f"Search the web and summarize key facts with links for: {q}",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        texts, links = [], []
        for item in data.get("output") or []:
            for c in item.get("content") or []:
                if c.get("text"):
                    texts.append(c["text"])
                for ann in c.get("annotations") or []:
                    if ann.get("type") == "url_citation" and ann.get("url"):
                        links.append(f"- {ann.get('title') or ann['url']} ({ann['url']})")
        out = ["\n".join(texts).strip()] if texts else []
        if links:
            out.append("Sources:\n" + "\n".join(dict.fromkeys(links)))
        return "\n\n".join(out) or "No results found."

    if provider == "claude":
        base = ai_config.PROVIDERS["claude"]["base_url"]
        resp = await client.post(
            f"{base}/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1200,
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
                "messages": [{"role": "user", "content": f"Search the web and summarize key facts with links for: {q}"}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        texts, links = [], []
        for blk in data.get("content") or []:
            if blk.get("type") == "text" and blk.get("text"):
                texts.append(blk["text"])
            elif blk.get("type") == "web_search_tool_result":
                for item in blk.get("content") or []:
                    if item.get("url"):
                        links.append(f"- {item.get('title') or item['url']} ({item['url']})")
        out = ["\n".join(texts).strip()] if texts else []
        if links:
            out.append("Sources:\n" + "\n".join(dict.fromkeys(links)))
        return "\n\n".join(out) or "No results found."

    raise RuntimeError(f"Unsupported native search provider: {provider}")


async def _search_fallback(client: httpx.AsyncClient, q: str) -> str:
    lines = []
    try:
        ddg = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": q, "format": "json", "no_html": "1", "skip_disambig": "1"},
        )
        if ddg.status_code == 200:
            d = ddg.json()
            if d.get("AbstractText"):
                lines.append(f"Summary ({d.get('AbstractURL', '')}): {d['AbstractText']}")
            for t in (d.get("RelatedTopics") or [])[:5]:
                if isinstance(t, dict) and t.get("Text"):
                    lines.append(f"- {t['Text']} ({t.get('FirstURL', '')})")
    except Exception:
        pass

    try:
        wiki = await client.get(
            "https://en.wikipedia.org/w/api.php",
            headers={"User-Agent": "innovation-agents/1.0 (research tool)"},
            params={"action": "query", "list": "search", "srsearch": q, "format": "json", "srlimit": 6},
        )
        if wiki.status_code == 200:
            for item in wiki.json().get("query", {}).get("search", []):
                title = item.get("title", "")
                slug = urllib.parse.quote(title.replace(" ", "_"))
                snippet = _clean(item.get("snippet", ""))
                lines.append(f"- {title} (https://en.wikipedia.org/wiki/{slug})\n  {snippet}")
    except Exception:
        pass

    if not lines:
        return (
            "No open web search results returned from basic fallback (Wikipedia/DDG Instant Answer). "
            "Configure Tavily, Serper, Brave, local SearXNG, or a Gemini/OpenAI/Claude key in AI Settings for full web search."
        )
    return "\n".join(lines)


@tool(
    "WebSearch",
    "Search the web for public information, news, companies, products, patents, and funding programmes.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
async def web_search(args):
    q = (args.get("query") or "").strip()
    if not q:
        return _reply("Empty search query.")

    cfg = ai_config.load_config()
    backend = ai_config.resolve_search_backend(cfg)
    cache_key = f"search:{backend['mode']}:{backend.get('provider', '')}:{q.lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return _reply(cached)

    async with httpx.AsyncClient(timeout=35.0, follow_redirects=True) as client:
        mode = backend["mode"]
        text = None
        errors = []

        if mode == "tavily":
            try:
                text = await _search_tavily(client, q, ai_config.get_search_key("tavily", cfg))
            except Exception as err:
                errors.append(f"tavily: {err}")
        elif mode == "serper":
            try:
                text = await _search_serper(client, q, ai_config.get_search_key("serper", cfg))
            except Exception as err:
                errors.append(f"serper: {err}")
        elif mode == "brave":
            try:
                text = await _search_brave(client, q, ai_config.get_search_key("brave", cfg))
            except Exception as err:
                errors.append(f"brave: {err}")
        elif mode == "searxng":
            try:
                text = await _search_searxng(client, q, ai_config.get_searxng_url(cfg))
            except Exception as err:
                errors.append(f"searxng: {err}")
        elif mode == "provider_native":
            try:
                text = await _search_provider_native(client, q, backend["provider"], cfg)
            except Exception as err:
                errors.append(f"{backend['provider']} native search: {err}")

        if text is None:
            text = await _search_fallback(client, q)
            if errors:
                text = f"[Note: primary search failed ({'; '.join(errors)}); used basic fallback]\n\n" + text

    _cache_set(cache_key, text)
    return _reply(text)


@tool(
    "WebFetch",
    "Download a web page or PDF URL and return clean readable text (with optional 'focus' to extract specific facts).",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "focus": {"type": "string", "description": "Optional question or topic to extract from long pages"},
        },
        "required": ["url"],
    },
)
async def web_fetch(args):
    url = (args.get("url") or "").strip()
    focus = (args.get("focus") or "").strip()
    if not url.startswith(("http://", "https://")):
        return {"content": [{"type": "text", "text": "URL must start with http:// or https://"}], "is_error": True}

    cache_key = f"fetch:{url}:{focus}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return _reply(cached)

    cfg = ai_config.load_config()
    jina_fallback = bool((cfg.get("search") or {}).get("jina_fallback", True))
    text = ""
    blocked_or_sparse = False

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (innovation-agents)"})
            if resp.status_code in (403, 429):
                blocked_or_sparse = True
            else:
                resp.raise_for_status()
                ctype = (resp.headers.get("content-type") or "").lower()
                is_pdf = "application/pdf" in ctype or url.lower().split("?", 1)[0].endswith(".pdf") or resp.content[:5] == b"%PDF-"
                if is_pdf:
                    def parse_pdf(raw: bytes) -> str:
                        reader = PdfReader(io.BytesIO(raw))
                        return unicodedata.normalize("NFKC", "\n\n".join(p.extract_text() or "" for p in reader.pages))
                    text = await asyncio.to_thread(parse_pdf, resp.content)
                elif "application/json" in ctype or "text/plain" in ctype:
                    text = resp.text
                else:
                    extracted = await asyncio.to_thread(
                        trafilatura.extract,
                        resp.text,
                        include_links=True,
                        include_tables=True,
                        output_format="markdown",
                    )
                    text = (extracted or "").strip() or _html_to_text(resp.text)
                    if len(text) < 300:
                        blocked_or_sparse = True
        except httpx.HTTPError:
            blocked_or_sparse = True

        if blocked_or_sparse and jina_fallback:
            try:
                jina_key = ai_config.get_search_key("jina", cfg)
                j_headers = {"Accept": "text/plain"}
                if jina_key:
                    j_headers["Authorization"] = f"Bearer {jina_key}"
                j_resp = await client.get(f"https://r.jina.ai/{url}", headers=j_headers)
                if j_resp.status_code == 200 and len(j_resp.text.strip()) > len(text):
                    text = j_resp.text.strip()
            except Exception:
                pass

    if not text:
        return {"content": [{"type": "text", "text": "Could not fetch readable content from this URL."}], "is_error": True}

    if focus and len(text) > 6000:
        try:
            import llm
            distill = await llm.run_agent(
                f"URL: {url}\nFocus: {focus}\n\nExtract the relevant facts, numbers, dates, and URLs verbatim from the text below:\n\n{text[:30000]}",
                llm.AgentOptions(
                    system_prompt="You extract accurate facts, numbers, dates, and links from fetched web pages. Never invent details.",
                    role="extractor",
                    max_turns=1,
                    max_tokens=1500,
                ),
            )
            if distill.get("text"):
                text = distill["text"].strip()
        except Exception:
            text = text[:15000]
    else:
        text = text[:15000]

    _cache_set(cache_key, text)
    return _reply(text)


# ---------- Scoped File Tools (Read, Edit, Write) ----------

def _is_within(path: Path, roots: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == r.resolve() or r.resolve() in resolved.parents for r in roots)


def make_file_tools(
    root: Path,
    read_roots: list[Path] | None = None,
    write_roots: list[Path] | None = None,
    allow_read: bool = True,
    allow_write: bool = False,
) -> list[ToolDef]:
    """Create sandboxed Read, Edit, and Write tools scoped to allowed directories."""
    r_roots = [p.resolve() for p in (read_roots or [root])]
    w_roots = [p.resolve() for p in (write_roots or [])]
    built: list[ToolDef] = []

    def resolve_path(raw: str) -> Path:
        p = Path(raw)
        return (p if p.is_absolute() else (root / p)).resolve()

    if allow_read:
        @tool(
            "Read",
            "Read a file from disk (supports Markdown, text, JSON, CSV, PDF, and images).",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "file_path": {"type": "string", "description": "Alias for path"},
                },
            },
        )
        async def read_file_tool(args):
            raw_path = args.get("path") or args.get("file_path") or ""
            if not raw_path:
                return {"content": [{"type": "text", "text": "Missing path"}], "is_error": True}
            target = resolve_path(raw_path)
            if not _is_within(target, r_roots):
                return {"content": [{"type": "text", "text": f"Access denied: {raw_path} is outside allowed read directories."}], "is_error": True}
            if not target.is_file():
                return {"content": [{"type": "text", "text": f"File not found: {raw_path}"}], "is_error": True}

            suffix = target.suffix.lower()
            if suffix == ".pdf":
                def extract_pdf():
                    reader = PdfReader(target)
                    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
                    return unicodedata.normalize("NFKC", text)
                text = await asyncio.to_thread(extract_pdf)
                return _reply(text[:60000] or "(PDF contained no extractable text)")
            if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                mime = mimetypes.guess_type(target.name)[0] or "image/png"
                b64 = base64.b64encode(target.read_bytes()).decode("ascii")
                return {
                    "content": [{"type": "text", "text": f"Loaded image {target.name} ({mime})."}],
                    "images": [{"mime": mime, "data_b64": b64}],
                }
            return _reply(target.read_text(errors="replace")[:60000])

        built.append(read_file_tool)

    if allow_write and w_roots:
        @tool(
            "Edit",
            "Replace an exact string in an existing file inside the allowed output folder.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["old_string", "new_string"],
            },
            read_only=False,
        )
        async def edit_file_tool(args):
            raw_path = args.get("path") or args.get("file_path") or ""
            target = resolve_path(raw_path)
            if not _is_within(target, w_roots):
                return {"content": [{"type": "text", "text": f"Edit denied: {raw_path} is outside allowed folder."}], "is_error": True}
            if not target.is_file():
                return {"content": [{"type": "text", "text": f"File not found: {raw_path}"}], "is_error": True}
            content = target.read_text()
            old, new = args["old_string"], args["new_string"]
            if old not in content:
                return {"content": [{"type": "text", "text": "Target text not found in file."}], "is_error": True}
            target.write_text(content.replace(old, new, 1))
            return _reply(f"Edited {target.name}.")

        @tool(
            "Write",
            "Write or overwrite a file inside the allowed output folder.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["content"],
            },
            read_only=False,
        )
        async def write_file_tool(args):
            raw_path = args.get("path") or args.get("file_path") or ""
            target = resolve_path(raw_path)
            if not _is_within(target, w_roots):
                return {"content": [{"type": "text", "text": f"Write denied: {raw_path} is outside allowed folder."}], "is_error": True}
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args["content"])
            return _reply(f"Wrote {target.name}.")

        built.extend([edit_file_tool, write_file_tool])

    return built


# ---------- Patents: Google Patents ----------

@tool(
    "google_patent",
    "Read one patent from Google Patents by publication number, e.g. US20200411873A1 or EP3544099A1. "
    "Returns title, owner, inventors, dates, legal status, countries it was filed in, CPC classes, timeline, "
    "abstract, claims, and the patents it cites, that cite it, and that are similar to it. "
    "Find starting patent numbers with web search, then follow the cited, citing and similar lists.",
    {"number": str},
)
async def google_patent(args):
    number = re.sub(r"[^A-Za-z0-9]", "", args["number"]).upper()
    url = f"https://patents.google.com/patent/{number}/en"
    page = await _fetch(url)

    def first(pattern):
        match = re.search(pattern, page, re.S)
        return _clean(match.group(1)) if match else "not given"

    def unique(pattern):
        return ", ".join(dict.fromkeys(_clean(m) for m in re.findall(pattern, page, re.S))) or "not given"

    def rows(prop):
        found = re.findall(rf'<tr itemprop="{prop}".*?</tr>', page, re.S)
        return "\n".join("  " + _clean(row) for row in found[:15]) or "  none"

    cpc = re.findall(r'itemprop="Code">([^<]+)</span>&mdash;<span itemprop="Description">([^<]+)</span>\s*<meta itemprop="Leaf"', page)
    events = re.findall(r'<dd itemprop="events".*?datetime="([^"]+)".*?itemprop="title">(.*?)</span>', page, re.S)
    return _reply(f"""{number}: {first(r'<span itemprop="title">(.*?)</span>')}
Link: {url}
Status: {first(r'itemprop="status">(.*?)<')} | priority date {first(r'itemprop="priorityDate" datetime="([^"]+)"')} | filed {first(r'itemprop="filingDate" datetime="([^"]+)"')} | published {first(r'itemprop="publicationDate" datetime="([^"]+)"')}
Owner: {unique(r'itemprop="assigneeCurrent"[^>]*>(.*?)</dd>')} | Inventors: {unique(r'itemprop="inventor"[^>]*>(.*?)</dd>')}
Filed in: {", ".join(dict.fromkeys(re.findall(r'itemprop="countryCode">(\w+)<', page)))}
CPC classes: {"; ".join(f"{code} {_clean(text)}" for code, text in dict.fromkeys(cpc))}
Timeline: {"; ".join(f"{day} {_clean(title)}" for day, title in events)}
Abstract: {first(r'itemprop="abstract".*?itemprop="content"[^>]*>(.*?)</section>')}
Claims: {first(r'itemprop="claims".*?itemprop="content"[^>]*>(.*?)</section>')[:5000]}
Cites:
{rows("backwardReferences")}
Cited by:
{rows("forwardReferences")}
Similar documents:
{rows("similarDocuments")}""")


# ---------- Scientific papers: OpenAlex ----------

@tool(
    "search_papers",
    "Search scientific papers in OpenAlex (over 250 million works). Returns title, year, citations, authors, "
    "institutions, funders, DOI and the start of the abstract for the 10 best matches. Use short keyword queries.",
    {"query": str},
)
async def search_papers(args):
    params = {"search": args["query"], "per_page": 10}
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    data = json.loads(await _fetch("https://api.openalex.org/works", params))
    lines = [f"{data['meta']['count']} papers found. Best matches:"]
    for work in data["results"]:
        words = {pos: word for word, positions in (work.get("abstract_inverted_index") or {}).items() for pos in positions}
        abstract = " ".join(words[pos] for pos in sorted(words))[:500]
        authors = ", ".join(a["author"]["display_name"] for a in work["authorships"][:4])
        institutions = ", ".join(dict.fromkeys(i["display_name"] for a in work["authorships"] for i in a["institutions"]))
        funders = ", ".join(f["display_name"] for f in work.get("funders") or []) or "not given"
        lines.append(
            f"- {work['display_name']} ({work['publication_year']}, cited {work['cited_by_count']} times) {work['doi'] or work['id']}\n"
            f"  Authors: {authors}. Institutions: {institutions[:300]}. Funders: {funders}.\n  Abstract: {abstract}"
        )
    return _reply("\n".join(lines))


AALTO = "https://ror.org/020hwjq30"


@tool(
    "research_trend",
    "How active a research topic is, from OpenAlex: papers per year for the last 10 years, the institutions that "
    "publish most on it, and Aalto University's own paper count with its most cited papers. Use short keyword queries. "
    "Good evidence for whether a field is growing and how strong Aalto is in it.",
    {"query": str},
)
async def research_trend(args):
    key = {"api_key": os.environ["OPENALEX_API_KEY"]} if os.environ.get("OPENALEX_API_KEY") else {}

    async def works(**params):
        return json.loads(await _fetch("https://api.openalex.org/works", {"search": args["query"], **params, **key}))

    years, places, aalto = await asyncio.gather(
        works(group_by="publication_year"), works(group_by="authorships.institutions.id"),
        works(filter=f"authorships.institutions.ror:{AALTO}", sort="cited_by_count:desc", per_page=5))
    per_year = sorted((g["key"], g["count"]) for g in years["group_by"])[-10:]
    lines = [f"{years['meta']['count']} papers match. Per year: " + ", ".join(f"{y}: {n}" for y, n in per_year),
             "Most active institutions: " + ", ".join(f"{g['key_display_name']} ({g['count']})" for g in places["group_by"][:10]),
             f"Aalto University: {aalto['meta']['count']} papers. Most cited:"]
    lines += [f"- {w['display_name']} ({w['publication_year']}, cited {w['cited_by_count']} times) {w['doi'] or w['id']}" for w in aalto["results"]]
    return _reply("\n".join(lines))


# ---------- EU funding calls: Funding & Tenders Portal ----------

@tool(
    "eu_funding_calls",
    "Search EU funding calls that are open or upcoming (Horizon Europe, EIC, Digital Europe, LIFE and others). "
    "Returns topic ID, title, type of action, opening date and deadline. Read a call in full with eu_call_details.",
    {"query": str},
)
async def eu_funding_calls(args):
    query = {"bool": {"must": [{"terms": {"type": ["1", "8"]}}, {"terms": {"status": ["31094501", "31094502"]}}]}}
    boundary = uuid.uuid4().hex
    body = "".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="blob"\r\n'
        f"Content-Type: application/json\r\n\r\n{json.dumps(value)}\r\n"
        for name, value in {"query": query, "languages": ["en"]}.items()
    ) + f"--{boundary}--\r\n"
    data = json.loads(await _fetch(
        "https://api.tech.ec.europa.eu/search-api/prod/rest/search",
        {"apiKey": "SEDIA", "text": args["query"], "pageSize": 100, "pageNumber": 1},
        data=body.encode(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    ))
    today = date.today().isoformat()
    calls = {}
    for result in data["results"]:
        meta = result["metadata"]
        deadline = max((d[:10] for d in meta.get("deadlineDate") or []), default="not set")
        opens = min((d[:10] for d in meta.get("startDate") or []), default="not set")
        if max(deadline, opens) < today:
            continue
        topic = meta.get("identifier", ["?"])[0]
        calls[topic] = f"- {topic}: {meta.get('title', ['?'])[0]} | {', '.join(meta.get('typesOfAction', []))} | opens {opens} | deadline {deadline}"
    return _reply("\n".join(list(calls.values())[:25]) or "No open or upcoming EU calls found for these words.")


@tool(
    "eu_call_details",
    "Read one EU funding call topic in full by its topic ID (from eu_funding_calls): expected outcomes, scope, "
    "budget, deadlines and conditions.",
    {"topic_id": str},
)
async def eu_call_details(args):
    topic = args["topic_id"].strip().lower()
    details = json.loads(await _fetch(f"https://ec.europa.eu/info/funding-tenders/opportunities/data/topicDetails/{topic}.json"))["TopicDetails"]

    def day(milliseconds):
        return date.fromtimestamp(int(milliseconds) / 1000).isoformat()

    actions = (details.get("budgetOverviewJSONItem") or {}).get("budgetTopicActionMap", {}).values()
    budget = "\n".join(
        f"- {b['action'].split(' - ', 1)[-1]}: total {sum(int(v) for v in b['budgetYearMap'].values()):,} EUR, "
        f"about {b.get('expectedGrants')} grants of {b.get('minContribution')} to {b.get('maxContribution')} EUR each, "
        f"opens {day(b['plannedOpeningDate'])}, deadline {', '.join(day(d) for d in b['deadlineDates'])}"
        for action in actions for b in action if b["action"].startswith(details["identifier"])
    )
    status = ", ".join(a["status"]["description"] for a in details.get("actions") or [])
    return _reply(f"""{details['identifier']}: {details['title']}
Call: {details.get('callTitle')} | Status: {status}
Link: https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/{topic}
Budget:
{budget or "not given"}

{_clean(details.get('description'))[:6000]}

Conditions: {_clean(details.get('conditions'))[:2000]}""")


def eu_call_facts(topic_id):
    """One EU call topic's deadline and budget, straight from the portal."""
    details = json.loads(_get(f"https://ec.europa.eu/info/funding-tenders/opportunities/data/topicDetails/{topic_id.lower()}.json"))["TopicDetails"]
    days = sorted(date.fromtimestamp(int(d) / 1000).isoformat() for a in details.get("actions") or [] for d in a.get("deadlineDates") or [])
    coming = [d for d in days if d >= date.today().isoformat()]
    budget = sum(int(v) for action in (details.get("budgetOverviewJSONItem") or {}).get("budgetTopicActionMap", {}).values()
                 for b in action if b["action"].startswith(details["identifier"]) for v in b["budgetYearMap"].values())
    return {"deadline": coming[0] if coming else days[-1] if days else "", "budget": f"EUR {budget:,}" if budget else ""}


# ---------- Finnish companies: PRH / YTJ trade register ----------

@tool(
    "finnish_companies",
    "Search the Finnish Trade Register (PRH/YTJ) by company name, main line of business (TOL 2008 code, "
    "e.g. 27200 = manufacture of batteries and accumulators), city, or a mix. Returns business ID, name, company form, "
    "industry, website, registration date, address and register status for up to 20 companies. No financial figures.",
    {
        "type": "object",
        "properties": {"name": {"type": "string"}, "industry_code": {"type": "string"}, "city": {"type": "string"}},
    },
)
async def finnish_companies(args):
    params = {"name": args.get("name"), "mainBusinessLine": args.get("industry_code"), "location": args.get("city")}
    data = json.loads(await _fetch(
        "https://avoindata.prh.fi/opendata-ytj-api/v3/companies", {k: v for k, v in params.items() if v}
    ))

    def english(item):
        return next((d["description"] for d in (item or {}).get("descriptions", []) if d["languageCode"] == "3"), "")

    def current(items):
        return [item for item in items or [] if not item.get("endDate")]

    lines = [f"{data['totalResults']} companies found."]
    for company in data["companies"][:20]:
        name = next((n["name"] for n in current(company["names"])), company["names"][0]["name"])
        address = next((a for a in company.get("addresses") or [] if a["type"] == 1), None)
        city = next((p["city"] for p in (address or {}).get("postOffices", []) if p["languageCode"] == "1"), "")
        where = f"{address['street']} {address['buildingNumber']}, {address['postCode']} {city}" if address else "not given"
        status = ", ".join(dict.fromkeys(english(e) for e in current(company.get("registeredEntries")) + company.get("companySituations", [])))
        form = ", ".join(english(f) for f in current(company.get("companyForms"))) or "form not given"
        industry = company.get("mainBusinessLine") or {}
        lines.append(
            f"- {name} ({company['businessId']['value']}), {form}. "
            f"Industry: {industry.get('type', '?')} {english(industry)}. Website: {(company.get('website') or {}).get('url', 'not given')}. "
            f"Registered {company.get('registrationDate')}. Address: {where}. Status: {status}"
        )
    return _reply("\n".join(lines))


# ---------- Statistics: Eurostat ----------

@functools.cache
def _eurostat_catalogue():
    return [line.replace('"', "").split("\t") for line in _get(
        "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt", {"lang": "en"}
    ).splitlines()[1:]]


@tool(
    "eurostat_search",
    "Find Eurostat datasets whose title contains all the given words, e.g. 'enterprises turnover', "
    "'industrial production' or 'international trade'. Returns dataset code, title and years covered. "
    "Then get the numbers with eurostat_data.",
    {"words": str},
)
async def eurostat_search(args):
    words = args["words"].lower().split()
    rows = [r for r in await asyncio.to_thread(_eurostat_catalogue) if r[2] in ("dataset", "table") and all(w in r[0].lower() for w in words)]
    return _reply("\n".join(f"- {r[1]}: {r[0].strip()} ({r[5]} to {r[6]})" for r in rows[:30]) or "No datasets found. Try fewer or other words.")


@tool(
    "eurostat_data",
    "Get numbers from one Eurostat dataset. 'filters' maps dimension codes to values, e.g. "
    '{"geo": ["FI", "SE", "EU27_2020"], "sinceTimePeriod": "2020"}. Start with a narrow request such as '
    '{"geo": "FI", "lastTimePeriod": "1"} to see the dimensions, then ask for what you need. Returns up to 300 values. '
    "Two product-level datasets give real market sizes and always need narrow filters: DS-059358 is PRODCOM, sold "
    'production, exports and imports per product, e.g. {"reporter": "EU27_2020", "product": "27202300", "indicators": '
    '"PRODVAL", "time": "2023"} with an 8-digit PRODCOM code; DS-045409 is EU trade per product, e.g. {"reporter": "FI", '
    '"partner": "WORLD", "product": "850760", "flow": "2", "freq": "A", "time": "2025"} with an HS or CN code '
    "(flow 1 = import, 2 = export). Find product codes with web search.",
    {
        "type": "object",
        "properties": {"dataset": {"type": "string"}, "filters": {"type": "object"}},
        "required": ["dataset"],
    },
)
async def eurostat_data(args):
    data = json.loads(await _fetch(
        f"https://ec.europa.eu/eurostat/api/{'comext/' if args['dataset'].upper().startswith('DS-') else ''}"
        f"dissemination/statistics/1.0/data/{args['dataset']}",
        {"lang": "en", **(args.get("filters") or {})},
    ))
    dims = [(d, data["dimension"][d]["category"]) for d in data["id"]]
    codes = [sorted(cat["index"], key=cat["index"].get) for _, cat in dims]
    fixed = [f"{d}={cat['label'][c[0]]}" for (d, cat), c in zip(dims, codes) if len(c) == 1]
    lines = [f"{data['label']}. Same for all values: {', '.join(fixed)}"]
    for flat, value in sorted(data["value"].items(), key=lambda item: int(item[0]))[:300]:
        position, parts = int(flat), []
        for (d, cat), c, size in reversed(list(zip(dims, codes, data["size"]))):
            position, i = divmod(position, size)
            if size > 1:
                parts.append(f"{d}={cat['label'][c[i]]}")
        lines.append(f"{', '.join(reversed(parts))}: {value}")
    if not data["value"]:
        lines.append("No values. The figure may be confidential or not collected; check the codes and try other years.")
    return _reply("\n".join(lines))


# ---------- Research projects and funding: OpenAIRE ----------

@tool(
    "research_projects",
    "Search research projects funded by the EU, the Research Council of Finland and many other national funders "
    "(OpenAIRE). Returns title, project code, funder, dates, funded amount and summary. All words must match, so use "
    "1 to 3 broad keywords. Optional funder filter: EC for the EU, AKA for the Research Council of Finland.",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}, "funder": {"type": "string"}},
        "required": ["query"],
    },
)
async def research_projects(args):
    params = {"search": args["query"], "pageSize": 15}
    if args.get("funder"):
        params["fundingShortName"] = args["funder"]
    data = json.loads(await _fetch("https://api.openaire.eu/graph/v1/projects", params))
    lines = [f"{data['header']['numFound']} projects found."]
    for project in data["results"]:
        funders = ", ".join(f"{f['name']} ({f.get('jurisdiction') or '?'})" for f in project.get("fundings") or [])
        granted = project.get("granted") or {}
        amount = f"{granted['fundedAmount']:,.0f} {granted.get('currency', '')}" if granted.get("fundedAmount") else "amount not given"
        lines.append(
            f"- {project['title']} ({project.get('acronym') or project['code']}), {funders}, "
            f"{project.get('startDate')} to {project.get('endDate')}, {amount}\n  {_clean(project.get('summary'))[:400]}"
        )
    return _reply("\n".join(lines))


# ---------- Patents: European Patent Office (OPS) ----------

@tool(
    "epo_patents",
    "Search published patents worldwide in the European Patent Office database (needs EPO keys in .env). "
    "The query uses CQL, for example: ta=\"solid state battery\" and pd>=2024, cpc=H01M10 and pa=aalto, "
    "or ti all \"lignin binder\". Fields: ti title, ab abstract, ta title or abstract, pa applicant, in inventor, "
    "cpc class, pd publication date, pn publication number or country (pn=EP). Returns the hit count and up to "
    "25 patents with number, date, applicants, CPC classes, abstract and an Espacenet link.",
    {"query": str},
)
async def epo_patents(args):
    total, found = await asyncio.to_thread(epo.search, args["query"], 25)
    lines = [f"{total} patents match. Showing {len(found)}:"]
    lines += [f"- {p['number']} ({p['date']}) {p['title']}. Applicants: {', '.join(p['applicants']) or 'not given'}. "
              f"CPC: {', '.join(p['cpc'])}. {p['link']}\n  {p['abstract']}" for p in found]
    return _reply("\n".join(lines))


@tool(
    "epo_patent",
    "Look up one patent at the European Patent Office by publication number, e.g. EP3544099A1 or WO2019180077 (needs "
    "EPO keys in .env). Returns every publication in its patent family (which countries, and whether any was granted: "
    "kind B), the latest legal events for each (grant, lapse, withdrawal, fees) and the claims. Use it to check whether "
    "a similar patent was granted and is still in force.",
    {"number": str},
)
async def epo_patent(args):
    found = await asyncio.to_thread(epo.details, args["number"])
    if not found["members"]:
        return _reply(f"No patent family found for {found['number']}.")
    granted = [m["number"] for m in found["members"] if m["kind"].startswith("B")]
    lines = [f"{found['number']}: family of {len(found['members'])} publications. "
             f"Granted: {', '.join(granted) if granted else 'no granted publication in the family yet'}.",
             f"Link: https://worldwide.espacenet.com/patent/search?q=pn%3D{found['number']}"]
    for m in found["members"]:
        events = "; ".join(f"{day} {code} {text}" for day, code, text in m["events"][-6:]) or "none recorded"
        lines.append(f"- {m['number']} ({m['date']}). Latest legal events: {events}")
    return _reply("\n".join(lines) + f"\n\nClaims:\n{found['claims'][:6000] or 'not available from the EPO for this publication'}")


def images_on(url):
    """The pictures on a page as (alt text, absolute link), with the page's own preview image (og:image) first."""
    page = _get(url)
    found = [("page preview image", m) for m in re.findall(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', page)]
    found += [("site icon", m) for m in re.findall(r'<link[^>]+rel=["\']apple-touch-icon[^"\']*["\'][^>]+href=["\']([^"\']+)', page)]
    for tag in re.findall(r"<img\b[^>]*>", page, re.I):
        src = re.search(r'\s(?:data-src|data-lazy-src|src)=["\']([^"\']+)', tag) or re.search(r'srcset=["\']([^\s"\']+)', tag)
        alt = re.search(r'\salt=["\']([^"\']*)', tag)
        if src and not src.group(1).startswith("data:"):
            found.append((html.unescape(alt.group(1)) if alt else "", urllib.parse.urljoin(url, html.unescape(src.group(1)))))
    return list(dict.fromkeys((alt, urllib.parse.urljoin(url, html.unescape(src))) for alt, src in found))


@tool(
    "page_images",
    "List the pictures on a web page with their alt text and full links, for finding photos of people (on team, about "
    "or press pages, or Wikipedia) or a company logo. Match a photo to a person by the alt text or the file name.",
    {"url": str},
)
async def page_images(args):
    found = await asyncio.to_thread(images_on, args["url"])
    return _reply("\n".join(f"- {alt or '(no alt text)'}: {src}" for alt, src in found[:80]) or "No pictures found on this page.")


TOOLS = {t.name: t for t in [
    web_search, web_fetch,
    google_patent, search_papers, research_trend, eu_funding_calls, eu_call_details,
    finnish_companies, eurostat_search, eurostat_data, research_projects, epo_patents, epo_patent, page_images,
]}
