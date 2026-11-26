"""Run specialist innovation agents on invention disclosures using any configured AI provider/model.

Supports OpenAI, Google Gemini, Anthropic Claude, and OpenAI-compatible endpoints (Ollama, vLLM,
OpenRouter, LM Studio, Groq, etc.).

Each file in agents/ is one agent. Each run gets its own folder, outputs/<date>_<time>_<first file name>/,
with one report per agent and a run_info.txt listing the inputs and how each agent is doing.

Usage:
    python run.py disclosures/example.md
    python run.py disclosures/example.md -a market nabc
    python run.py disclosures/example.md --provider openai_compatible --model qwen2.5:32b --base-url http://localhost:11434/v1
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import ai_config
import llm
from tools import TOOLS, make_file_tools

ROOT = Path(__file__).resolve().parent
AGENTS_DIR = ROOT / "agents"
OUTPUT_DIR = ROOT / "outputs"

TASK = """Input files (read them all first with the Read tool):
{files}

The invention disclosure is confidential and unpublished. When searching the web or using data tools, \
use general terms (field, product type, company names) and never paste distinctive details of the invention.

Reply with your finished report in Markdown as your final message."""


def agent_options(
    system_prompt: str,
    tool_names: list[str],
    *,
    allow_read: bool = True,
    allow_write: bool = False,
    read_roots: list[Path] | None = None,
    write_roots: list[Path] | None = None,
    extra_tools: list[llm.ToolDef] | None = None,
    output_schema: dict | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    endpoint_id: str | None = None,
    role: str | None = None,
    history: list[dict] | None = None,
    pre_tool_hook=None,
    max_turns: int = 30,
) -> llm.AgentOptions:
    """Build provider-agnostic AgentOptions with scoped file tools and data tools."""
    file_tools = make_file_tools(
        ROOT,
        read_roots=read_roots,
        write_roots=write_roots,
        allow_read=allow_read,
        allow_write=allow_write,
    )
    selected_tools = [TOOLS[name] for name in tool_names if name in TOOLS]
    all_tools = [*file_tools, *selected_tools, *(extra_tools or [])]
    return llm.AgentOptions(
        system_prompt=system_prompt,
        tools=all_tools,
        output_schema=output_schema,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        endpoint_id=endpoint_id,
        role=role,
        history=history,
        pre_tool_hook=pre_tool_hook,
        max_turns=max_turns,
    )


def read_agent(name: str) -> tuple[dict[str, str], str]:
    """Read agents/<name>.md: an optional 'key: value' header between --- lines, then the system prompt."""
    text = (AGENTS_DIR / f"{name}.md").read_text()
    config: dict[str, str] = {}
    if text.startswith("---"):
        header, text = text[3:].split("---", 1)
        for line in header.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()
    return config, text.strip()


def load_agent(
    name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> llm.AgentOptions:
    config, prompt = read_agent(name)
    names = [t.strip() for t in config.get("tools", "").split(",") if t.strip()]
    return agent_options(
        prompt,
        names,
        allow_read=True,
        provider=provider or config.get("provider"),
        model=model or config.get("model"),
        base_url=base_url or config.get("base_url"),
        api_key=api_key,
        role="disclosure",
    )


async def run_agent(
    name: str,
    files: list[str],
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Run one agent on the input files and return its final Markdown reply."""
    prompt = TASK.format(files="\n".join(f"- {Path(f).resolve()}" for f in files))
    opts = load_agent(name, provider=provider, model=model, base_url=base_url, api_key=api_key)
    res = await llm.run_agent(prompt, opts)
    return res["text"]


def start_run(
    files: list[str],
    agents: list[str],
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
):
    """Create the run folder and return it with the coroutine that runs the agents."""
    started = datetime.now()
    out_dir = OUTPUT_DIR / f"{started:%Y-%m-%d_%H-%M-%S}_{Path(files[0]).stem}"
    out_dir.mkdir(parents=True)
    resolved = ai_config.resolve(provider=provider, model=model, base_url=base_url, api_key=api_key, role="disclosure")

    later = [a for a in agents if read_agent(a)[0].get("after") == "reports"]
    first = [a for a in agents if a not in later]
    status = {a: f"[{a}] {'waiting' if a in later else 'running'}" for a in agents}
    inputs = "\n".join(f"  {Path(f).resolve()}" for f in files)

    def save_status():
        (out_dir / "run_info.txt").write_text(
            f"Started: {started:%Y-%m-%d %H:%M:%S}\n"
            f"Model: {resolved['provider']} / {resolved['model']}\n"
            f"Inputs:\n{inputs}\n\n"
            + "\n".join(status.values())
            + "\n"
        )

    async def run_and_save(name, agent_files):
        status[name] = f"[{name}] running"
        save_status()
        start = time.monotonic()
        try:
            reply = await run_agent(name, agent_files, provider=provider, model=model, base_url=base_url, api_key=api_key)
            (out_dir / f"{name}.md").write_text(reply)
            result = f"saved {name}.md"
        except Exception as error:
            result = f"failed: {error}"
        status[name] = f"[{name}] {result} ({time.monotonic() - start:.0f} s)"
        print(status[name])
        save_status()

    async def run_all():
        await asyncio.gather(*(run_and_save(a, files) for a in first))
        reports = [str(out_dir / f"{a}.md") for a in first if (out_dir / f"{a}.md").exists()]
        await asyncio.gather(*(run_and_save(a, files + reports) for a in later))

    save_status()
    return out_dir, run_all()


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="the disclosure first, then any extra data files")
    parser.add_argument("-a", "--agents", nargs="+", default=sorted(p.stem for p in AGENTS_DIR.glob("*.md")))
    parser.add_argument("--provider", help="openai, gemini, claude, or openai_compatible")
    parser.add_argument("--model", help="Model ID to use for this run")
    parser.add_argument("--base-url", help="Custom base URL (for OpenAI-compatible endpoints)")
    parser.add_argument("--api-key", help="Override API key for this run")
    args = parser.parse_args()

    resolved = ai_config.resolve(
        provider=args.provider, model=args.model, base_url=args.base_url, api_key=args.api_key, role="disclosure"
    )
    if ai_config.PROVIDERS[resolved["provider"]]["requires_key"] and not resolved["api_key"]:
        sys.exit(f"Missing API key for provider '{resolved['provider']}'. Set it in .env, ai_config.json, or --api-key.")

    out_dir, work = start_run(
        args.files, args.agents, provider=args.provider, model=args.model, base_url=args.base_url, api_key=args.api_key
    )
    print(f"Using {resolved['provider']} ({resolved['model']})")
    print(f"Running {', '.join(args.agents)} on {', '.join(args.files)}\nSaving to {out_dir.relative_to(ROOT)}/")
    await work


if __name__ == "__main__":
    asyncio.run(main())
