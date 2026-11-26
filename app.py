"""Web page for analysing invention disclosures: drop a file to run the agents, read the reports,
choose/switch AI providers & models, and chat with an assistant about them.

Start it with:  .venv/bin/python app.py   then open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

import ai_config
import ai_routes
import llm
from run import AGENTS_DIR, OUTPUT_DIR, ROOT, agent_options, start_run

DISCLOSURES_DIR = ROOT / "disclosures"
FILE_TYPES = {".pdf", ".md", ".txt", ".png", ".jpg", ".jpeg", ".gif", ".webp"}

CHAT_PROMPT = """You help an innovation advisor at Aalto Innovation work with the results of one analysis run.

The run's folder is {folder}. It holds one Markdown report per agent, and run_info.txt, which lists the input files.

- Answer questions from the reports and input files. Read them when you need them with the Read tool; never guess what they say.
- When asked to change a report, edit the file in the run folder using Edit or Write, then say in one or two sentences what you changed. \
You can only edit files in {folder}.
- When asked to check something, verify it with WebSearch or the data tools and say what you found, with links. \
Point out anything in the reports that looks wrong.
- Keep answers short and in plain language.

The invention disclosure is confidential and unpublished. When searching the web or using data tools, use general \
terms and never paste distinctive details of the invention."""

app = FastAPI()
app.include_router(ai_routes.router)
sessions: dict[str, list[dict]] = {}  # run id -> chat history
analyses: set[asyncio.Task] = set()


class ChatRequest(BaseModel):
    run: str
    message: str
    viewing: str | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    endpoint_id: str | None = None


def run_folder(run: str) -> Path:
    folder = (OUTPUT_DIR / run).resolve()
    if folder.parent != OUTPUT_DIR.resolve() or not folder.is_dir():
        raise HTTPException(404, "No such run")
    return folder


@app.get("/")
def page():
    return FileResponse(ROOT / "ui.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/runs")
def runs():
    """All runs, newest first, with their report names and run_info.txt."""
    found = []
    OUTPUT_DIR.mkdir(exist_ok=True)
    for folder in sorted((f for f in OUTPUT_DIR.iterdir() if f.is_dir()), reverse=True):
        info = folder / "run_info.txt"
        found.append({
            "id": folder.name,
            "reports": sorted(p.stem for p in folder.glob("*.md")),
            "info": info.read_text() if info.exists() else "",
        })
    return found


@app.post("/api/runs")
async def new_run(
    request: Request,
    name: str,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    endpoint_id: str | None = None,
):
    """Save an uploaded disclosure in disclosures/ and start all agents on it in the background."""
    filename = re.sub(r"[^\w.-]", "_", Path(name).name)
    suffix = Path(filename).suffix.lower()
    if suffix not in FILE_TYPES:
        raise HTTPException(400, "Use a PDF, text, Markdown or image file. Save Word files as PDF first.")
    data = await request.body()
    if not data:
        raise HTTPException(400, "The file is empty.")

    resolved = ai_config.resolve(provider=provider, model=model, base_url=base_url, endpoint_id=endpoint_id, role="disclosure")
    if ai_config.PROVIDERS[resolved["provider"]]["requires_key"] and not resolved["api_key"]:
        raise HTTPException(
            503,
            f"Missing API key for {ai_config.PROVIDERS[resolved['provider']]['label']}. "
            "Open AI Settings in the top bar to add your key or choose another provider.",
        )

    DISCLOSURES_DIR.mkdir(exist_ok=True)
    path, n = DISCLOSURES_DIR / filename, 2
    while path.exists() and path.read_bytes() != data:
        path, n = DISCLOSURES_DIR / f"{Path(filename).stem}-{n}{suffix}", n + 1
    path.write_bytes(data)

    out_dir, work = start_run(
        [str(path)],
        sorted(p.stem for p in AGENTS_DIR.glob("*.md")),
        provider=resolved["provider"],
        model=resolved["model"],
        base_url=resolved["base_url"],
    )
    task = asyncio.create_task(work)
    analyses.add(task)
    task.add_done_callback(analyses.discard)
    return {"id": out_dir.name, "provider": resolved["provider"], "model": resolved["model"]}


@app.get("/api/runs/{run}/{name}", response_class=PlainTextResponse)
def report(run: str, name: str):
    folder = run_folder(run)
    path = (folder / f"{name}.md").resolve()
    if path.parent != folder or not path.exists():
        raise HTTPException(404, "No such report")
    return path.read_text()


@app.post("/api/chat/{run}/reset")
def reset(run: str):
    sessions.pop(run, None)
    return {"ok": True}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    folder = run_folder(request.run)
    resolved = ai_config.resolve(
        provider=request.provider,
        model=request.model,
        base_url=request.base_url,
        endpoint_id=request.endpoint_id,
        role="chat",
    )
    if ai_config.PROVIDERS[resolved["provider"]]["requires_key"] and not resolved["api_key"]:
        raise HTTPException(
            503,
            f"Missing API key for {ai_config.PROVIDERS[resolved['provider']]['label']}. Configure it in AI Settings.",
        )

    all_tools = [
        "WebSearch", "WebFetch", "google_patent", "search_papers", "research_trend",
        "eu_funding_calls", "eu_call_details", "finnish_companies", "eurostat_search",
        "eurostat_data", "research_projects", "epo_patents", "epo_patent",
    ]
    options = agent_options(
        CHAT_PROMPT.format(folder=folder),
        all_tools,
        allow_read=True,
        allow_write=True,
        read_roots=[folder, DISCLOSURES_DIR],
        write_roots=[folder],
        provider=resolved["provider"],
        model=resolved["model"],
        base_url=resolved["base_url"],
        history=sessions.get(request.run),
    )
    prompt = f"[Report open on my screen: {request.viewing}.md]\n\n{request.message}" if request.viewing else request.message

    def event(**fields):
        return json.dumps(fields) + "\n"

    async def stream():
        try:
            async for evt in llm.run_agent_stream(prompt, options):
                etype = evt["type"]
                if etype == "text":
                    yield event(type="text", text=evt["text"])
                elif etype == "tool":
                    yield event(type="tool", id=evt["id"], name=evt["name"], detail=evt["detail"])
                elif etype == "tool_failed":
                    yield event(type="tool_failed", id=evt["id"])
                elif etype == "result":
                    sessions[request.run] = evt["history"]
                elif etype == "error":
                    yield event(type="error", text=evt["text"])
        except Exception as error:
            yield event(type="error", text=str(error))

    return StreamingResponse(stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
