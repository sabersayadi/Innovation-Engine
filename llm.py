"""Provider-agnostic async LLM client, streaming agent loop, and model discovery.

Supports:
- OpenAI (/v1/chat/completions)
- OpenAI-compatible endpoints (Ollama, LM Studio, vLLM, OpenRouter, Groq, DeepSeek, etc.)
- Google Gemini (/v1beta/models/{model}:streamGenerateContent)
- Anthropic Claude (/v1/messages)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

import ai_config


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], Awaitable[Any]]
    read_only: bool = True


@dataclass
class AgentOptions:
    system_prompt: str
    tools: list[ToolDef] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    endpoint_id: str | None = None
    role: str | None = None
    history: list[dict[str, Any]] | None = None
    pre_tool_hook: Callable[[str, dict[str, Any]], Awaitable[str | None]] | None = None
    max_turns: int = 30
    max_tokens: int = 8192
    max_context_chars: int = 240_000


def _compact_history(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """Trim older tool outputs when total conversation chars exceed max_chars, keeping recent turns intact."""
    total = sum(len(str(m.get("content") or "")) for m in messages)
    if total <= max_chars or len(messages) <= 8:
        return messages
    compacted = [dict(m) for m in messages]
    keep_recent = 6
    for i in range(1, len(compacted) - keep_recent):
        if compacted[i].get("role") == "tool":
            txt = compacted[i].get("content") or ""
            if len(txt) > 800:
                compacted[i]["content"] = txt[:600] + "\n...[earlier tool output compacted to fit context]..."
                total -= (len(txt) - len(compacted[i]["content"]))
                if total <= max_chars:
                    break
    return compacted


def _normalize_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure a tool parameter schema is a valid JSON Schema object."""
    if not schema:
        return {"type": "object", "properties": {}}
    if "type" in schema:
        return schema
    # Shorthand dict like {"query": str} or {"number": str}
    props = {}
    req = []
    for k, v in schema.items():
        if v is str:
            props[k] = {"type": "string"}
        elif v is int:
            props[k] = {"type": "integer"}
        elif v is float:
            props[k] = {"type": "number"}
        elif v is bool:
            props[k] = {"type": "boolean"}
        elif isinstance(v, dict):
            props[k] = v
        else:
            props[k] = {"type": "string"}
        req.append(k)
    return {"type": "object", "properties": props, "required": req}


def _strip_gemini_unsupported(obj: Any) -> Any:
    """Gemini rejects 'additionalProperties' and '$schema' in function/response schemas."""
    if isinstance(obj, dict):
        return {
            k: _strip_gemini_unsupported(v)
            for k, v in obj.items()
            if k not in ("additionalProperties", "$schema")
        }
    if isinstance(obj, list):
        return [_strip_gemini_unsupported(x) for x in obj]
    return obj


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from raw text, tolerating Markdown code fences or surrounding prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.S)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Model did not return valid JSON: {text[:300]}")


def _normalize_tool_output(raw: Any) -> tuple[str, list[dict[str, str]], bool]:
    """Convert tool return value into (text, images, is_error)."""
    if isinstance(raw, dict) and "content" in raw:
        texts = [b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text"]
        images = raw.get("images", [])
        return "\n".join(texts), images, bool(raw.get("is_error", False))
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], list):
        return str(raw[0]), raw[1], False
    return str(raw), [], False


def _tool_detail(args: dict[str, Any]) -> str:
    val = next((v for v in args.values() if isinstance(v, str) and v.strip()), "")
    if not val:
        return ""
    if "/" in val and val.startswith("/"):
        return val.rsplit("/", 1)[-1][:80]
    return val[:80]


# ---------- Provider adapters (single turn, streaming) ----------

async def _call_openai_stream(
    client: httpx.AsyncClient,
    resolved: dict[str, str],
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[ToolDef],
    output_schema: dict[str, Any] | None,
    on_text: Callable[[str], Awaitable[None]] | None,
) -> dict[str, Any]:
    url = f"{resolved['base_url']}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if resolved["api_key"]:
        headers["Authorization"] = f"Bearer {resolved['api_key']}"

    oai_messages: list[dict[str, Any]] = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = msg["role"]
        if role == "user":
            images = msg.get("images") or []
            if images:
                parts: list[dict[str, Any]] = [{"type": "text", "text": msg.get("content") or ""}]
                for img in images:
                    parts.append({"type": "image_url", "image_url": {"url": f"data:{img['mime']};base64,{img['data_b64']}"}})
                oai_messages.append({"role": "user", "content": parts})
            else:
                oai_messages.append({"role": "user", "content": msg.get("content") or ""})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or None}
            if msg.get("tool_calls"):
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"], ensure_ascii=False)},
                    }
                    for tc in msg["tool_calls"]
                ]
            oai_messages.append(entry)
        elif role == "tool":
            oai_messages.append({"role": "tool", "tool_call_id": msg["tool_call_id"], "content": msg.get("content") or ""})

    payload: dict[str, Any] = {"model": resolved["model"], "messages": oai_messages, "stream": True}
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": _normalize_schema(t.parameters)},
            }
            for t in tools
        ]
    if output_schema and not tools:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "structured_output", "strict": True, "schema": output_schema},
        }

    # Try streaming with fallback for OpenAI-compatible servers that reject strict json_schema.
    attempts = [payload]
    if "response_format" in payload:
        p2 = {**payload, "response_format": {"type": "json_object"}}
        p3 = {k: v for k, v in payload.items() if k != "response_format"}
        attempts = [payload, p2, p3]

    last_err = ""
    for attempt_payload in attempts:
        text_parts: list[str] = []
        tc_acc: dict[int, dict[str, str]] = {}
        async with client.stream("POST", url, headers=headers, json=attempt_payload) as resp:
            if resp.status_code == 400 and len(attempts) > 1:
                last_err = (await resp.aread()).decode("utf-8", "ignore")
                continue
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "ignore")
                raise RuntimeError(f"{resolved['provider']} HTTP {resp.status_code}: {body[:500]}")

            ctype = resp.headers.get("content-type", "")
            if "text/event-stream" not in ctype and "application/json" in ctype:
                raw = json.loads(await resp.aread())
                choice = raw["choices"][0]["message"]
                text = choice.get("content") or ""
                if text and on_text:
                    await on_text(text)
                calls = []
                for tc in choice.get("tool_calls") or []:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                    calls.append({"id": tc.get("id") or uuid.uuid4().hex[:10], "name": tc["function"]["name"], "arguments": args})
                return {"content": text, "tool_calls": calls}

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    piece = delta["content"]
                    text_parts.append(piece)
                    if on_text:
                        await on_text(piece)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tc_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

        calls = []
        for idx in sorted(tc_acc):
            slot = tc_acc[idx]
            try:
                args = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": slot["id"] or uuid.uuid4().hex[:10], "name": slot["name"], "arguments": args})
        return {"content": "".join(text_parts), "tool_calls": calls}

    raise RuntimeError(f"{resolved['provider']} HTTP 400: {last_err[:500]}")


async def _call_claude_stream(
    client: httpx.AsyncClient,
    resolved: dict[str, str],
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[ToolDef],
    output_schema: dict[str, Any] | None,
    on_text: Callable[[str], Awaitable[None]] | None,
    max_tokens: int,
) -> dict[str, Any]:
    if not resolved["api_key"]:
        raise RuntimeError("Anthropic Claude API key is missing. Add ANTHROPIC_API_KEY in AI Settings or .env.")
    url = f"{resolved['base_url']}/messages"
    headers = {
        "x-api-key": resolved["api_key"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    claude_messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results():
        if pending_tool_results:
            claude_messages.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for msg in messages:
        role = msg["role"]
        if role == "tool":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": msg.get("content") or "",
                **({"is_error": True} if msg.get("is_error") else {}),
            })
            continue
        flush_tool_results()
        if role == "user":
            blocks: list[dict[str, Any]] = []
            for img in msg.get("images") or []:
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": img["mime"], "data": img["data_b64"]},
                })
            if msg.get("content"):
                blocks.append({"type": "text", "text": msg["content"]})
            claude_messages.append({"role": "user", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == "assistant":
            blocks = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls") or []:
                blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
            claude_messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
    flush_tool_results()

    payload: dict[str, Any] = {
        "model": resolved["model"],
        "max_tokens": max_tokens,
        "messages": claude_messages,
        "stream": True,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if tools:
        payload["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": _normalize_schema(t.parameters)}
            for t in tools
        ]
    elif output_schema:
        # Force a StructuredOutput tool call when structured output is requested without other tools.
        payload["tools"] = [{"name": "StructuredOutput", "description": "Return the final structured JSON result.", "input_schema": output_schema}]
        payload["tool_choice"] = {"type": "tool", "name": "StructuredOutput"}

    text_parts: list[str] = []
    blocks_acc: dict[int, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []

    async with client.stream("POST", url, headers=headers, json=payload) as resp:
        if resp.status_code >= 400:
            body = (await resp.aread()).decode("utf-8", "ignore")
            raise RuntimeError(f"Claude HTTP {resp.status_code}: {body[:500]}")
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            etype = evt.get("type")
            if etype == "content_block_start":
                idx = evt["index"]
                blk = evt["content_block"]
                blocks_acc[idx] = {**blk, "_json": ""}
            elif etype == "content_block_delta":
                idx = evt["index"]
                delta = evt.get("delta") or {}
                if delta.get("type") == "text_delta":
                    piece = delta.get("text", "")
                    text_parts.append(piece)
                    if on_text:
                        await on_text(piece)
                elif delta.get("type") == "input_json_delta" and idx in blocks_acc:
                    blocks_acc[idx]["_json"] += delta.get("partial_json", "")
            elif etype == "content_block_stop":
                idx = evt["index"]
                blk = blocks_acc.get(idx)
                if blk and blk.get("type") == "tool_use":
                    raw_json = blk.get("_json", "").strip()
                    try:
                        args = json.loads(raw_json) if raw_json else (blk.get("input") or {})
                    except json.JSONDecodeError:
                        args = {}
                    calls.append({"id": blk["id"], "name": blk["name"], "arguments": args})

    return {"content": "".join(text_parts), "tool_calls": calls}


async def _call_gemini_stream(
    client: httpx.AsyncClient,
    resolved: dict[str, str],
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[ToolDef],
    output_schema: dict[str, Any] | None,
    on_text: Callable[[str], Awaitable[None]] | None,
) -> dict[str, Any]:
    if not resolved["api_key"]:
        raise RuntimeError("Google Gemini API key is missing. Add GEMINI_API_KEY in AI Settings or .env.")
    model = resolved["model"].removeprefix("models/")
    url = f"{resolved['base_url']}/models/{model}:streamGenerateContent?alt=sse"
    headers = {"x-goog-api-key": resolved["api_key"], "Content-Type": "application/json"}

    contents: list[dict[str, Any]] = []
    pending_fn_responses: list[dict[str, Any]] = []

    def flush_fn_responses():
        if pending_fn_responses:
            contents.append({"role": "user", "parts": list(pending_fn_responses)})
            pending_fn_responses.clear()

    for msg in messages:
        role = msg["role"]
        if role == "tool":
            pending_fn_responses.append({
                "functionResponse": {
                    "name": msg["name"],
                    "response": {"result": msg.get("content") or "", "is_error": bool(msg.get("is_error"))},
                }
            })
            continue
        flush_fn_responses()
        if role == "user":
            parts: list[dict[str, Any]] = []
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for img in msg.get("images") or []:
                parts.append({"inlineData": {"mimeType": img["mime"], "data": img["data_b64"]}})
            contents.append({"role": "user", "parts": parts or [{"text": ""}]})
        elif role == "assistant":
            if msg.get("raw_gemini_parts"):
                contents.append({"role": "model", "parts": msg["raw_gemini_parts"]})
            else:
                parts = []
                if msg.get("content"):
                    parts.append({"text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    parts.append({"functionCall": {"name": tc["name"], "args": tc["arguments"]}})
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
    flush_fn_responses()

    payload: dict[str, Any] = {"contents": contents}
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if tools:
        payload["tools"] = [{
            "functionDeclarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": _strip_gemini_unsupported(_normalize_schema(t.parameters)),
                }
                for t in tools
            ]
        }]
    if output_schema and not tools:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": _strip_gemini_unsupported(output_schema),
        }

    text_parts: list[str] = []
    raw_parts: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    async with client.stream("POST", url, headers=headers, json=payload) as resp:
        if resp.status_code >= 400:
            body = (await resp.aread()).decode("utf-8", "ignore")
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {body[:500]}")
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            for cand in chunk.get("candidates") or []:
                for part in (cand.get("content") or {}).get("parts") or []:
                    raw_parts.append(part)
                    if part.get("thought"):
                        continue
                    if "text" in part and part["text"]:
                        piece = part["text"]
                        text_parts.append(piece)
                        if on_text:
                            await on_text(piece)
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        calls.append({
                            "id": f"gem_{uuid.uuid4().hex[:10]}",
                            "name": fc["name"],
                            "arguments": fc.get("args") or {},
                        })

    return {"content": "".join(text_parts), "tool_calls": calls, "raw_gemini_parts": raw_parts}


# ---------- Public agent runner ----------

async def run_agent_stream(prompt: str, options: AgentOptions) -> AsyncIterator[dict[str, Any]]:
    """Run an agent (with multi-turn tool loop) and yield streaming events.

    Yields dicts:
      {"type": "text", "text": str}
      {"type": "tool", "id": str, "name": str, "detail": str}
      {"type": "tool_failed", "id": str, "error": str}
      {"type": "result", "text": str, "structured_output": dict | None, "history": list}
      {"type": "error", "text": str}
    """
    resolved = ai_config.resolve(
        provider=options.provider,
        model=options.model,
        base_url=options.base_url,
        api_key=options.api_key,
        endpoint_id=options.endpoint_id,
        role=options.role,
    )

    tools = list(options.tools)
    system_prompt = options.system_prompt
    structured_via_tool = bool(options.output_schema and tools)

    if structured_via_tool:
        async def _submit_structured(args):
            return {"content": [{"type": "text", "text": "Structured output accepted."}]}

        tools.append(ToolDef(
            name="StructuredOutput",
            description="Submit the final structured JSON response after finishing all required research.",
            parameters=options.output_schema,  # type: ignore[arg-type]
            fn=_submit_structured,
        ))
        system_prompt += (
            "\n\nIMPORTANT: When you have finished your research, call the `StructuredOutput` tool with your complete "
            "JSON answer matching its schema."
        )
    elif options.output_schema and not tools:
        schema_hint = json.dumps(options.output_schema, ensure_ascii=False)
        system_prompt += f"\n\nReturn ONLY valid JSON matching this schema:\n{schema_hint}"

    tool_map = {t.name: t for t in tools}
    messages: list[dict[str, Any]] = list(options.history or [])
    if prompt:
        messages.append({"role": "user", "content": prompt})

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    structured_result: dict[str, Any] | None = None
    final_text = ""

    async def push_text(delta: str):
        await queue.put({"type": "text", "text": delta})

    async def drive():
        nonlocal structured_result, final_text
        async with httpx.AsyncClient(timeout=httpx.Timeout(240.0, connect=20.0)) as client:
            async def call_once(use_tools: list[ToolDef], use_schema: dict[str, Any] | None, emit_text: bool):
                bounded = _compact_history(messages, options.max_context_chars)
                if resolved["provider"] in ("openai", "openai_compatible"):
                    return await _call_openai_stream(
                        client, resolved, system_prompt, bounded, use_tools,
                        use_schema, push_text if emit_text else None,
                    )
                if resolved["provider"] == "claude":
                    return await _call_claude_stream(
                        client, resolved, system_prompt, bounded, use_tools,
                        use_schema, push_text if emit_text else None, options.max_tokens,
                    )
                if resolved["provider"] == "gemini":
                    return await _call_gemini_stream(
                        client, resolved, system_prompt, bounded, use_tools,
                        use_schema, push_text if emit_text else None,
                    )
                raise RuntimeError(f"Unsupported provider: {resolved['provider']}")

            for turn in range(options.max_turns):
                reply = await call_once(tools, options.output_schema if not tools else None, not structured_via_tool)

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": reply.get("content") or "",
                }
                if reply.get("tool_calls"):
                    assistant_msg["tool_calls"] = reply["tool_calls"]
                if reply.get("raw_gemini_parts"):
                    assistant_msg["raw_gemini_parts"] = reply["raw_gemini_parts"]
                messages.append(assistant_msg)

                tool_calls = reply.get("tool_calls") or []
                if not tool_calls:
                    final_text = reply.get("content") or ""
                    if options.output_schema and structured_result is None:
                        try:
                            structured_result = _extract_json(final_text)
                        except ValueError:
                            # One-shot JSON repair pass for smaller/OpenAI-compatible models that emitted prose.
                            messages.append({
                                "role": "user",
                                "content": (
                                    "Your previous response was not valid JSON matching the required schema. "
                                    "Return ONLY a single valid JSON object matching the schema, with no markdown fences or extra prose.\n\n"
                                    f"Schema: {json.dumps(options.output_schema, ensure_ascii=False)}"
                                ),
                            })
                            repaired = await call_once([], options.output_schema, False)
                            final_text = repaired.get("content") or ""
                            if repaired.get("tool_calls"):
                                structured_result = repaired["tool_calls"][0].get("arguments") or {}
                            else:
                                structured_result = _extract_json(final_text)
                    break

                # Execute tool calls (checking pre_tool_hook if provided).
                follow_up_images: list[dict[str, str]] = []
                done_structured = False

                for tc in tool_calls:
                    tname, targs, tid = tc["name"], tc.get("arguments") or {}, tc["id"]
                    await queue.put({"type": "tool", "id": tid, "name": tname, "detail": _tool_detail(targs)})

                    if options.pre_tool_hook:
                        deny_reason = await options.pre_tool_hook(tname, targs)
                        if deny_reason:
                            await queue.put({"type": "tool_failed", "id": tid, "error": deny_reason})
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tid,
                                "name": tname,
                                "content": deny_reason,
                                "is_error": True,
                            })
                            continue

                    if tname == "StructuredOutput":
                        structured_result = targs
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tid,
                            "name": tname,
                            "content": "Accepted.",
                        })
                        done_structured = True
                        continue

                    tool_def = tool_map.get(tname)
                    if not tool_def:
                        err = f"Tool '{tname}' is not available."
                        await queue.put({"type": "tool_failed", "id": tid, "error": err})
                        messages.append({"role": "tool", "tool_call_id": tid, "name": tname, "content": err, "is_error": True})
                        continue

                    try:
                        raw_out = await tool_def.fn(targs)
                        out_text, out_images, is_err = _normalize_tool_output(raw_out)
                        if is_err:
                            await queue.put({"type": "tool_failed", "id": tid, "error": out_text})
                        if out_images:
                            follow_up_images.extend(out_images)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tid,
                            "name": tname,
                            "content": out_text,
                            "is_error": is_err,
                        })
                    except Exception as exc:
                        err = f"Tool {tname} failed: {exc}"
                        await queue.put({"type": "tool_failed", "id": tid, "error": err})
                        messages.append({"role": "tool", "tool_call_id": tid, "name": tname, "content": err, "is_error": True})

                if follow_up_images:
                    messages.append({
                        "role": "user",
                        "content": "Here are the images loaded by the Read tool:",
                        "images": follow_up_images,
                    })

                if done_structured and structured_result is not None:
                    final_text = json.dumps(structured_result, ensure_ascii=False)
                    break
            else:
                raise RuntimeError(f"Agent exceeded max turns ({options.max_turns})")

    task = asyncio.create_task(drive())
    try:
        while True:
            get_item = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({task, get_item}, return_when=asyncio.FIRST_COMPLETED)
            if get_item in done:
                item = get_item.result()
                if item is not None:
                    yield item
            else:
                get_item.cancel()
            if task in done:
                while not queue.empty():
                    item = queue.get_nowait()
                    if item is not None:
                        yield item
                exc = task.exception()
                if exc:
                    yield {"type": "error", "text": str(exc)}
                else:
                    yield {
                        "type": "result",
                        "text": final_text,
                        "structured_output": structured_result,
                        "history": messages,
                        "provider": resolved["provider"],
                        "model": resolved["model"],
                    }
                break
    finally:
        if not task.done():
            task.cancel()


async def run_agent(prompt: str, options: AgentOptions) -> dict[str, Any]:
    """Run an agent to completion and return {'text', 'structured_output', 'history', 'provider', 'model'}."""
    result = None
    error = None
    async for evt in run_agent_stream(prompt, options):
        if evt["type"] == "result":
            result = evt
        elif evt["type"] == "error":
            error = evt["text"]
    if error:
        raise RuntimeError(error)
    if result is None:
        raise RuntimeError("Agent finished with no result")
    return result


# ---------- Model discovery & connection testing ----------

async def fetch_models(provider: str, base_url: str | None = None, api_key: str | None = None, endpoint_id: str | None = None) -> list[str]:
    """Query the provider's /models endpoint to list available model IDs."""
    resolved = ai_config.resolve(provider=provider, base_url=base_url, api_key=api_key, endpoint_id=endpoint_id)
    prov = resolved["provider"]
    async with httpx.AsyncClient(timeout=20.0) as client:
        if prov in ("openai", "openai_compatible"):
            headers = {"Authorization": f"Bearer {resolved['api_key']}"} if resolved["api_key"] else {}
            resp = await client.get(f"{resolved['base_url']}/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            ids = [m["id"] for m in data.get("data", []) if m.get("id")]
            return sorted(ids)
        if prov == "gemini":
            if not resolved["api_key"]:
                raise RuntimeError("GEMINI_API_KEY is required to fetch Gemini models.")
            resp = await client.get(f"{resolved['base_url']}/models", headers={"x-goog-api-key": resolved["api_key"]})
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return sorted(
                m["name"].removeprefix("models/")
                for m in models
                if "generateContent" in (m.get("supportedGenerationMethods") or [])
            )
        if prov == "claude":
            if not resolved["api_key"]:
                raise RuntimeError("ANTHROPIC_API_KEY is required to fetch Claude models.")
            resp = await client.get(
                f"{resolved['base_url']}/models",
                headers={"x-api-key": resolved["api_key"], "anthropic-version": "2023-06-01"},
            )
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []) if m.get("id"))
    return ai_config.PROVIDERS.get(prov, {}).get("models", [])


async def test_connection(
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    endpoint_id: str | None = None,
) -> dict[str, Any]:
    """Verify credentials and model responsiveness with a minimal prompt."""
    start = time.monotonic()
    res = await run_agent(
        "Reply with exactly: OK",
        AgentOptions(
            system_prompt="You are a concise connection check.",
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            endpoint_id=endpoint_id,
            max_turns=1,
            max_tokens=32,
        ),
    )
    return {
        "ok": True,
        "provider": res["provider"],
        "model": res["model"],
        "reply": res["text"].strip()[:120],
        "latency_ms": round((time.monotonic() - start) * 1000),
    }
