"""Verification tests for provider-agnostic AI configuration, streaming adapters, tools, and web endpoints."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ai_config  # noqa: E402
import app as disclosure_app  # noqa: E402
import llm  # noqa: E402
import terminal as terminal_app  # noqa: E402
import tools  # noqa: E402


class ProviderAgnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.orig_cfg = ai_config.CONFIG_FILE
        ai_config.CONFIG_FILE = Path(self.tmp.name) / "ai_config.json"

    def tearDown(self):
        ai_config.CONFIG_FILE = self.orig_cfg
        self.tmp.cleanup()

    def test_config_switch_and_masking(self):
        updated = ai_config.update_config({
            "active": {"provider": "openai", "model": "gpt-4.1"},
            "keys": {"openai": "sk-test-1234567890abcdef", "gemini": "AIzaSyTestKey12345"},
            "custom_endpoints": [
                {
                    "id": "local-ollama",
                    "name": "Local Ollama",
                    "base_url": "http://localhost:11434/v1",
                    "model": "qwen2.5:32b",
                    "api_key": "ollama-secret-9999",
                }
            ],
            "overrides": {
                "firewall_secure": {
                    "provider": "openai_compatible",
                    "model": "qwen2.5:32b",
                    "base_url": "http://localhost:11434/v1",
                }
            },
        })
        self.assertTrue(updated["keys"]["openai"]["configured"])
        self.assertNotIn("1234567890", updated["keys"]["openai"]["masked"])
        self.assertEqual(updated["custom_endpoints"][0]["masked_key"], "olla...9999")

        # Switching to custom OpenAI-compatible endpoint
        ai_config.update_config({
            "active": {
                "provider": "openai_compatible",
                "endpoint_id": "local-ollama",
                "model": "qwen2.5:32b",
                "base_url": "http://localhost:11434/v1",
            }
        })
        resolved = ai_config.resolve()
        self.assertEqual(resolved["provider"], "openai_compatible")
        self.assertEqual(resolved["base_url"], "http://localhost:11434/v1")
        self.assertEqual(resolved["api_key"], "ollama-secret-9999")

        # Per-role override
        ov = ai_config.resolve(role="firewall_secure")
        self.assertEqual(ov["provider"], "openai_compatible")
        self.assertEqual(ov["model"], "qwen2.5:32b")

    def test_web_apps_and_ai_routes(self):
        c1 = TestClient(disclosure_app.app)
        self.assertEqual(c1.get("/").status_code, 200)
        self.assertIn("ai-switcher", c1.get("/").text)
        self.assertEqual(c1.get("/api/runs").status_code, 200)
        self.assertEqual(c1.get("/api/ai/config").status_code, 200)
        self.assertEqual(c1.get("/api/ai/widget.js").status_code, 200)

        r = c1.post("/api/ai/config", json={"active": {"provider": "gemini", "model": "gemini-2.5-flash"}})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["active"]["provider"], "gemini")

        c2 = TestClient(terminal_app.app)
        self.assertEqual(c2.get("/").status_code, 200)
        self.assertIn("ai-switcher", c2.get("/").text)
        self.assertEqual(c2.get("/api/sectors").status_code, 200)
        self.assertEqual(c2.get("/api/ai/config").json()["active"]["provider"], "gemini")

    def test_scoped_file_tools(self):
        ftools = {t.name: t for t in tools.make_file_tools(
            ROOT, read_roots=[ROOT / "disclosures"], write_roots=[Path(self.tmp.name)], allow_read=True, allow_write=True
        )}
        ok = asyncio.run(ftools["Read"].fn({"path": str(ROOT / "disclosures" / "example.md")}))
        self.assertFalse(ok.get("is_error"))
        denied = asyncio.run(ftools["Read"].fn({"path": str(ROOT / "app.py")}))
        self.assertTrue(denied.get("is_error"))

        out_file = Path(self.tmp.name) / "report.md"
        wrote = asyncio.run(ftools["Write"].fn({"path": str(out_file), "content": "hello world"}))
        self.assertFalse(wrote.get("is_error"))
        edited = asyncio.run(ftools["Edit"].fn({"path": str(out_file), "old_string": "world", "new_string": "Aalto"}))
        self.assertFalse(edited.get("is_error"))
        self.assertEqual(out_file.read_text(), "hello Aalto")

        write_denied = asyncio.run(ftools["Write"].fn({"path": str(ROOT / "hack.txt"), "content": "x"}))
        self.assertTrue(write_denied.get("is_error"))

    def test_all_four_provider_adapters_with_tools(self):
        ai_config.update_config({
            "keys": {"openai": "sk-oai", "gemini": "gem-key", "claude": "sk-ant"},
        })

        @tools.tool("ping", "Return pong", {"value": str})
        async def ping_tool(args):
            return {"content": [{"type": "text", "text": f"pong:{args['value']}"}]}

        # 1. OpenAI & OpenAI-compatible SSE tool call -> text response
        oai_turn = {"n": 0}

        def oai_handler(request: httpx.Request) -> httpx.Response:
            oai_turn["n"] += 1
            if oai_turn["n"] == 1:
                body = (
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"ping","arguments":"{\\"value\\":\\"hi\\"}"}}]}}]}\n\n'
                    "data: [DONE]\n\n"
                )
            else:
                body = (
                    'data: {"choices":[{"delta":{"content":"Done "}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":"openAI"}}]}\n\n'
                    "data: [DONE]\n\n"
                )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        # 2. Claude SSE tool call -> text response
        cla_turn = {"n": 0}

        def claude_handler(request: httpx.Request) -> httpx.Response:
            cla_turn["n"] += 1
            if cla_turn["n"] == 1:
                body = (
                    'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"ping","input":{}}}\n\n'
                    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"value\\":\\"hi\\"}"}}\n\n'
                    'data: {"type":"content_block_stop","index":0}\n\n'
                )
            else:
                body = (
                    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
                    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Done claude"}}\n\n'
                    'data: {"type":"content_block_stop","index":0}\n\n'
                )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        # 3. Gemini SSE tool call -> text response
        gem_turn = {"n": 0}

        def gemini_handler(request: httpx.Request) -> httpx.Response:
            gem_turn["n"] += 1
            if gem_turn["n"] == 1:
                chunk = {"candidates": [{"content": {"parts": [{"functionCall": {"name": "ping", "args": {"value": "hi"}}}]}}]}
            else:
                chunk = {"candidates": [{"content": {"parts": [{"text": "Done gemini"}]}}]}
            body = f"data: {json.dumps(chunk)}\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        orig_client = httpx.AsyncClient

        def make_patched(handler):
            return lambda *a, **kw: orig_client(transport=httpx.MockTransport(handler))

        for prov, model, base_url, handler, expected in [
            ("openai", "gpt-4.1", None, oai_handler, "Done openAI"),
            ("openai_compatible", "qwen2.5:32b", "http://localhost:11434/v1", oai_handler, "Done openAI"),
            ("claude", "claude-sonnet-4-5-20250929", None, claude_handler, "Done claude"),
            ("gemini", "gemini-2.5-flash", None, gemini_handler, "Done gemini"),
        ]:
            oai_turn["n"] = 0
            with patch.object(llm.httpx, "AsyncClient", side_effect=make_patched(handler)):
                res = asyncio.run(llm.run_agent(
                    "Say hi",
                    llm.AgentOptions(
                        system_prompt="test",
                        tools=[ping_tool],
                        provider=prov,
                        model=model,
                        base_url=base_url,
                    ),
                ))
                self.assertEqual(res["text"], expected)
                self.assertEqual(res["provider"], prov)

    def test_structured_output_with_hook_and_chat_streaming(self):
        ai_config.update_config({
            "active": {"provider": "openai", "model": "gpt-4.1"},
            "keys": {"openai": "sk-test-key"},
        })

        # Mock OpenAI returning StructuredOutput -> refused once by hook -> accepted on turn 2
        turn = {"n": 0}

        def hook_handler(request: httpx.Request) -> httpx.Response:
            turn["n"] += 1
            if turn["n"] == 1:
                args_json = json.dumps({"verdict": "ok"})
            else:
                args_json = json.dumps({"verdict": "confirmed"})
            tc = json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": f"s{turn['n']}", "function": {"name": "StructuredOutput", "arguments": args_json}}
            ]}}]})
            body = f"data: {tc}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        calls = {"refused": 0}

        async def pre_hook(name, args):
            if name == "StructuredOutput" and calls["refused"] == 0:
                calls["refused"] += 1
                return "Need more checks first"
            return None

        @tools.tool("noop", "No-op tool", {"q": str})
        async def noop_tool(args):
            return {"content": [{"type": "text", "text": "ok"}]}

        orig_client = httpx.AsyncClient
        with patch.object(llm.httpx, "AsyncClient", side_effect=lambda *a, **kw: orig_client(transport=httpx.MockTransport(hook_handler))):
            res = asyncio.run(llm.run_agent(
                "Evaluate",
                llm.AgentOptions(
                    system_prompt="sys",
                    tools=[noop_tool],
                    output_schema={"type": "object", "properties": {"verdict": {"type": "string"}}, "required": ["verdict"]},
                    pre_tool_hook=pre_hook,
                ),
            ))
            self.assertEqual(res["structured_output"], {"verdict": "confirmed"})
            self.assertEqual(calls["refused"], 1)

        # Mock /api/chat NDJSON streaming on app.py and terminal.py
        def chat_handler(request: httpx.Request) -> httpx.Response:
            body = 'data: {"choices":[{"delta":{"content":"Hello from mocked AI"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        with patch.object(llm.httpx, "AsyncClient", side_effect=lambda *a, **kw: orig_client(transport=httpx.MockTransport(chat_handler))):
            c1 = TestClient(disclosure_app.app)
            r1 = c1.post("/api/chat", json={"run": "2026-10-24_10-37-37_example", "message": "Hi"})
            self.assertEqual(r1.status_code, 200)
            events1 = [json.loads(line) for line in r1.text.splitlines() if line.strip()]
            self.assertTrue(any(e.get("type") == "text" and "Hello" in e.get("text", "") for e in events1))

            c2 = TestClient(terminal_app.app)
            r2 = c2.post("/api/chat", json={"message": "Hi"})
            self.assertEqual(r2.status_code, 200)
            events2 = [json.loads(line) for line in r2.text.splitlines() if line.strip()]
            self.assertTrue(any("Hello" in e.get("text", "") for e in events2))
            self.assertTrue(any("session" in e for e in events2))

    def test_search_cascade_webfetch_and_json_repair(self):
        tools._WEB_CACHE.clear()

        # 1. Search backend cascade resolution
        ai_config.update_config({"keys": {"gemini": "gem-search-key"}})
        self.assertEqual(ai_config.resolve_search_backend()["mode"], "provider_native")
        self.assertEqual(ai_config.resolve_search_backend()["provider"], "gemini")

        ai_config.update_config({"search": {"searxng_url": "http://localhost:8080"}})
        self.assertEqual(ai_config.resolve_search_backend()["mode"], "searxng")

        ai_config.update_config({"keys": {"tavily": "tvly-test"}})
        self.assertEqual(ai_config.resolve_search_backend()["mode"], "tavily")

        # 2. WebSearch mocked SearXNG + caching
        ai_config.update_config({"search": {"backend": "searxng", "searxng_url": "http://localhost:8080"}})
        hits = {"count": 0}

        def searx_handler(request: httpx.Request) -> httpx.Response:
            hits["count"] += 1
            return httpx.Response(200, json={
                "results": [{"title": "Solid-state battery news", "url": "https://example.com/ssb", "engine": "google", "content": "Details"}]
            })

        orig_client = httpx.AsyncClient
        with patch.object(tools.httpx, "AsyncClient", side_effect=lambda *a, **kw: orig_client(transport=httpx.MockTransport(searx_handler))):
            r1 = asyncio.run(tools.web_search.fn({"query": "solid state battery"}))
            r2 = asyncio.run(tools.web_search.fn({"query": "solid state battery"}))
            self.assertIn("Solid-state battery news", r1["content"][0]["text"])
            self.assertEqual(r1, r2)
            self.assertEqual(hits["count"], 1)  # served from short-TTL cache on second call

        # 3. WebSearch Gemini native search bridge
        ai_config.update_config({"search": {"backend": "provider_native"}})

        def gem_search_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "candidates": [{
                    "content": {"parts": [{"text": "Finland R2B funding is active."}]},
                    "groundingMetadata": {"groundingChunks": [{"web": {"title": "Business Finland", "uri": "https://www.businessfinland.fi"}}]},
                }]
            })

        with patch.object(tools.httpx, "AsyncClient", side_effect=lambda *a, **kw: orig_client(transport=httpx.MockTransport(gem_search_handler))):
            res = asyncio.run(tools.web_search.fn({"query": "Business Finland Research to Business"}))
            self.assertIn("https://www.businessfinland.fi", res["content"][0]["text"])

        # 4. WebFetch HTML trafilatura + Jina Reader SPA fallback
        def fetch_handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).startswith("https://r.jina.ai/"):
                return httpx.Response(200, text="Rendered SPA content about Nordic energy storage companies and funding.")
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html><body><div id='root'></div></body></html>")

        with patch.object(tools.httpx, "AsyncClient", side_effect=lambda *a, **kw: orig_client(transport=httpx.MockTransport(fetch_handler))):
            fetched = asyncio.run(tools.web_fetch.fn({"url": "https://example.com/spa"}))
            self.assertIn("Rendered SPA content", fetched["content"][0]["text"])

        # 5. JSON repair pass in llm.py when model returns invalid JSON first
        repair_turn = {"n": 0}

        def repair_handler(request: httpx.Request) -> httpx.Response:
            repair_turn["n"] += 1
            msg = "Sorry, here is some prose instead of JSON." if repair_turn["n"] == 1 else '{"answer":"fixed"}'
            body = f"data: {json.dumps({'choices': [{'delta': {'content': msg}}]})}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

        ai_config.update_config({"keys": {"openai": "sk-oai-repair"}})
        with patch.object(llm.httpx, "AsyncClient", side_effect=lambda *a, **kw: orig_client(transport=httpx.MockTransport(repair_handler))):
            out = asyncio.run(llm.run_agent(
                "Give JSON",
                llm.AgentOptions(
                    system_prompt="sys",
                    output_schema={"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                    provider="openai",
                    model="gpt-4.1",
                ),
            ))
            self.assertEqual(out["structured_output"], {"answer": "fixed"})
            self.assertEqual(repair_turn["n"], 2)


if __name__ == "__main__":
    unittest.main()
