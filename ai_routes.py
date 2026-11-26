"""Shared FastAPI routes and reusable UI widget for choosing AI providers, models, and endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import ai_config
import llm

router = APIRouter(prefix="/api/ai", tags=["ai"])


class ModelsRequest(BaseModel):
    provider: str
    base_url: str | None = None
    api_key: str | None = None
    endpoint_id: str | None = None


class TestRequest(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    endpoint_id: str | None = None


@router.get("/config")
def get_ai_config():
    return ai_config.public_config()


@router.post("/config")
def set_ai_config(payload: dict):
    try:
        return ai_config.update_config(payload)
    except ValueError as err:
        raise HTTPException(400, str(err))


@router.post("/models")
async def list_provider_models(req: ModelsRequest):
    try:
        raw_key = None if (not req.api_key or "..." in req.api_key or set(req.api_key) == {"*"}) else req.api_key
        models = await llm.fetch_models(req.provider, req.base_url, raw_key, req.endpoint_id)
        return {"models": models}
    except Exception as err:
        raise HTTPException(400, str(err))


@router.post("/test")
async def test_ai_connection(req: TestRequest):
    try:
        raw_key = None if (not req.api_key or "..." in req.api_key or set(req.api_key) == {"*"}) else req.api_key
        return await llm.test_connection(req.provider, req.model, req.base_url, raw_key, req.endpoint_id)
    except Exception as err:
        raise HTTPException(400, str(err))


WIDGET_JS = r"""
(() => {
  const STYLE = `
    .ai-pill-wrap { display: inline-flex; align-items: center; gap: 6px; position: relative; }
    .ai-pill {
      display: inline-flex; align-items: center; gap: 7px; padding: 6px 11px;
      border: 1px solid var(--line, #d9d9d9); border-radius: 999px;
      background: var(--panel, #fff); color: var(--text, #1b2230);
      font-size: 12.5px; font-weight: 600; cursor: pointer; max-width: 260px;
    }
    .ai-pill:hover { border-color: var(--accent, #3b5bdb); }
    .ai-dot { width: 8px; height: 8px; border-radius: 50%; background: #12805c; flex: none; }
    .ai-dot.warn { background: #d97706; }
    .ai-pill-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .ai-modal-backdrop {
      position: fixed; inset: 0; z-index: 1000; display: none; place-items: center; padding: 16px;
      background: rgba(15, 18, 24, 0.58); backdrop-filter: blur(2px);
    }
    .ai-modal-backdrop.open { display: grid; }
    .ai-modal {
      width: min(720px, 100%); max-height: 90vh; overflow-y: auto;
      background: var(--panel, #fff); color: var(--text, #1b2230);
      border: 1px solid var(--line, #d9d9d9); border-radius: 12px;
      padding: 22px 24px; box-shadow: 0 20px 50px rgba(0,0,0,0.28);
      font: 14px/1.5 Inter, system-ui, sans-serif;
    }
    .ai-modal h2 { margin: 0 0 4px; font-size: 18px; }
    .ai-modal p.sub { margin: 0 0 16px; color: var(--muted, #667085); font-size: 13px; }
    .ai-tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--line, #d9d9d9); margin-bottom: 16px; }
    .ai-tab {
      border: 0; background: none; padding: 8px 12px; font-weight: 600; font-size: 13px;
      color: var(--muted, #667085); border-bottom: 2px solid transparent; cursor: pointer;
    }
    .ai-tab.active { color: var(--text, #1b2230); border-bottom-color: var(--accent, #3b5bdb); }
    .ai-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .ai-field { display: flex; flex-direction: column; gap: 4px; margin-bottom: 12px; }
    .ai-field.full { grid-column: 1 / -1; }
    .ai-field label { font-size: 12px; font-weight: 600; color: var(--muted, #667085); }
    .ai-field input, .ai-field select {
      padding: 8px 10px; border: 1px solid var(--line, #d9d9d9); border-radius: 8px;
      background: var(--bg, #fff); color: var(--text, #1b2230); font: inherit; font-size: 13.5px;
    }
    .ai-row { display: flex; gap: 8px; align-items: center; }
    .ai-row > :first-child { flex: 1; }
    .ai-btn {
      padding: 8px 13px; border-radius: 8px; border: 1px solid var(--line, #d9d9d9);
      background: var(--bg, #f4f5f7); color: var(--text, #1b2230); font-weight: 600; font-size: 13px; cursor: pointer;
    }
    .ai-btn.primary { background: var(--accent, #3b5bdb); color: #fff; border-color: transparent; }
    .ai-btn:disabled { opacity: 0.5; cursor: default; }
    .ai-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .ai-chip {
      padding: 4px 9px; border-radius: 999px; border: 1px solid var(--line, #d9d9d9);
      background: none; font-size: 12px; cursor: pointer; color: var(--text, #1b2230);
    }
    .ai-chip:hover { border-color: var(--accent, #3b5bdb); }
    .ai-status { margin-top: 10px; padding: 9px 12px; border-radius: 8px; font-size: 13px; background: var(--bg, #f4f5f7); }
    .ai-status.ok { color: #0a7d38; }
    .ai-status.err { color: #c4320a; }
    .ai-ep-card {
      display: flex; justify-content: space-between; align-items: center; gap: 10px;
      padding: 9px 12px; border: 1px solid var(--line, #d9d9d9); border-radius: 8px; margin-bottom: 8px;
    }
    .ai-ep-card b { display: block; font-size: 13.5px; }
    .ai-ep-card small { color: var(--muted, #667085); font-size: 12px; }
    .ai-foot { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line, #d9d9d9); }
    @media (max-width: 640px) { .ai-grid { grid-template-columns: 1fr; } }
  `;

  let cfg = null;
  let tab = "model";
  let fetchedModels = [];

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  async function loadConfig() {
    const r = await fetch("/api/ai/config");
    if (!r.ok) return;
    cfg = await r.json();
    renderPill();
    window.dispatchEvent(new CustomEvent("ai-config-changed", { detail: cfg }));
  }

  function providerLabel(p) {
    return cfg?.providers?.[p]?.label || p;
  }

  function hasKeyForActive() {
    if (!cfg) return false;
    const p = cfg.active.provider;
    if (p === "openai_compatible") {
      if (cfg.active.endpoint_id) {
        const ep = cfg.custom_endpoints.find((e) => e.id === cfg.active.endpoint_id);
        return Boolean(ep && (ep.base_url || cfg.active.base_url));
      }
      return Boolean(cfg.active.base_url);
    }
    return Boolean(cfg.keys?.[p]?.configured);
  }

  function renderPill() {
    const mount = document.getElementById("ai-switcher");
    if (!mount || !cfg) return;
    const ok = hasKeyForActive();
    const shortProv = cfg.active.provider === "openai_compatible"
      ? (cfg.custom_endpoints.find((e) => e.id === cfg.active.endpoint_id)?.name || "Compatible")
      : providerLabel(cfg.active.provider).replace(/^(Google|Anthropic)\s+/, "");
    mount.innerHTML = `<div class="ai-pill-wrap">
      <button type="button" class="ai-pill" id="ai-open-btn" title="Change AI provider, model, or API key">
        <span class="ai-dot ${ok ? "" : "warn"}"></span>
        <span class="ai-pill-label">${esc(shortProv)} · ${esc(cfg.active.model)}</span>
      </button>
    </div>`;
    document.getElementById("ai-open-btn").onclick = openModal;
  }

  function ensureModal() {
    if (document.getElementById("ai-modal-backdrop")) return;
    const style = document.createElement("style");
    style.textContent = STYLE;
    document.head.appendChild(style);
    const bd = document.createElement("div");
    bd.id = "ai-modal-backdrop";
    bd.className = "ai-modal-backdrop";
    bd.innerHTML = `<div class="ai-modal" id="ai-modal-box"></div>`;
    bd.onclick = (e) => { if (e.target === bd) closeModal(); };
    document.body.appendChild(bd);
  }

  function openModal() {
    ensureModal();
    renderModal();
    document.getElementById("ai-modal-backdrop").classList.add("open");
  }

  function closeModal() {
    document.getElementById("ai-modal-backdrop")?.classList.remove("open");
  }

  function currentSuggestions(provider, endpointId) {
    const base = cfg?.providers?.[provider]?.models || [];
    const ep = endpointId ? cfg.custom_endpoints.find((e) => e.id === endpointId) : null;
    const combined = [...new Set([...fetchedModels, ...(ep?.models || []), ...(ep?.model ? [ep.model] : []), ...base])];
    return combined;
  }

  function renderModal() {
    const box = document.getElementById("ai-modal-box");
    if (!box || !cfg) return;
    const p = cfg.active.provider;
    const spec = cfg.providers[p];
    const suggestions = currentSuggestions(p, cfg.active.endpoint_id);

    let body = "";
    if (tab === "model") {
      const epOptions = cfg.custom_endpoints.map((ep) =>
        `<option value="${esc(ep.id)}" ${cfg.active.endpoint_id === ep.id ? "selected" : ""}>${esc(ep.name)} (${esc(ep.base_url)})</option>`
      ).join("");
      body = `
        <div class="ai-grid">
          <div class="ai-field">
            <label>AI Provider</label>
            <select id="ai-prov">
              ${Object.entries(cfg.providers).map(([id, info]) =>
                `<option value="${id}" ${id === p ? "selected" : ""}>${esc(info.label)}</option>`
              ).join("")}
            </select>
          </div>
          <div class="ai-field">
            <label>Model ID (choose or type any model)</label>
            <div class="ai-row">
              <input id="ai-model" list="ai-model-list" value="${esc(cfg.active.model)}" placeholder="e.g. gpt-4.1, gemini-2.5-flash, claude-sonnet-4-5-20250929">
              <datalist id="ai-model-list">${suggestions.map((m) => `<option value="${esc(m)}">`).join("")}</datalist>
              <button type="button" class="ai-btn" id="ai-fetch-models">Fetch</button>
            </div>
          </div>
          ${p === "openai_compatible" ? `
            <div class="ai-field">
              <label>Saved Compatible Endpoint (optional)</label>
              <select id="ai-ep-select">
                <option value="">Custom URL below</option>
                ${epOptions}
              </select>
            </div>
            <div class="ai-field">
              <label>Endpoint Base URL (must include /v1 if required)</label>
              <input id="ai-base-url" value="${esc(cfg.active.base_url || spec.base_url)}" placeholder="http://localhost:11434/v1">
            </div>
            <div class="ai-field full">
              <label>Quick-fill OpenAI-compatible templates</label>
              <div class="ai-chips">
                ${cfg.compat_templates.map((t) =>
                  `<button type="button" class="ai-chip" data-tpl="${esc(t.id)}">${esc(t.name)}</button>`
                ).join("")}
              </div>
            </div>
          ` : ""}
        </div>
        <div class="ai-field">
          <label>Suggested / Fetched Models (click to select)</label>
          <div class="ai-chips">
            ${suggestions.slice(0, 14).map((m) =>
              `<button type="button" class="ai-chip" data-pick-model="${esc(m)}">${esc(m)}</button>`
            ).join("")}
          </div>
        </div>
        <div class="ai-row" style="margin-top:8px">
          <button type="button" class="ai-btn" id="ai-test-btn">Test connection</button>
          <div id="ai-test-out" class="ai-status" style="display:none;flex:1;margin:0"></div>
        </div>
      `;
    } else if (tab === "keys") {
      const sb = cfg.search_status || {};
      body = `
        <p class="sub" style="margin-bottom:12px">Keys are stored locally in <code>ai_config.json</code> (or loaded from <code>.env</code>). Leave a field blank to keep its existing key, or click Clear.</p>
        <div class="ai-grid">
          ${["openai", "gemini", "claude", "openai_compatible"].map((k) => {
            const st = cfg.keys[k];
            const label = cfg.providers[k].label + (k === "openai_compatible" ? " Key (optional for local)" : " API Key");
            return `<div class="ai-field">
              <label>${esc(label)} · <span style="color:${st.configured ? "#0a7d38" : "#667085"}">${st.configured ? `Configured (${esc(st.masked)}, ${st.source})` : "Not set"}</span></label>
              <div class="ai-row">
                <input type="password" id="ai-key-${k}" placeholder="${st.configured ? st.masked : "Paste API key..."}">
                ${st.source === "ui" ? `<button type="button" class="ai-btn" data-clear-key="${k}">Clear</button>` : ""}
              </div>
            </div>`;
          }).join("")}
        </div>
        <h3 style="font-size:13px;margin:14px 0 6px">Web Search & Fetch Backend · <span style="color:${sb.full_web ? "#0a7d38" : "#d97706"}">${esc(sb.label || "Basic Fallback")}</span></h3>
        <div class="ai-grid">
          <div class="ai-field">
            <label>Search Backend Mode</label>
            <select id="ai-search-backend">
              ${Object.entries(cfg.search_backends || {}).map(([id, label]) =>
                `<option value="${esc(id)}" ${cfg.search?.backend === id ? "selected" : ""}>${esc(label)}</option>`
              ).join("")}
            </select>
          </div>
          <div class="ai-field">
            <label>Local SearXNG URL (optional, e.g. http://localhost:8080)</label>
            <input id="ai-searxng-url" value="${esc(cfg.search?.searxng_url || "")}" placeholder="http://localhost:8080">
          </div>
          ${["tavily", "serper", "brave", "jina"].map((k) => {
            const st = cfg.keys[k];
            const hint = k === "jina" ? "Optional (for WebFetch SPA fallback)" : "Not set";
            return `<div class="ai-field">
              <label>${esc(k.toUpperCase())} API Key · ${st.configured ? esc(st.masked) : hint}</label>
              <div class="ai-row">
                <input type="password" id="ai-key-${k}" placeholder="${st.configured ? st.masked : "Optional"}">
                ${st.source === "ui" ? `<button type="button" class="ai-btn" data-clear-key="${k}">Clear</button>` : ""}
              </div>
            </div>`;
          }).join("")}
          <div class="ai-field full" style="margin-bottom:0">
            <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
              <input type="checkbox" id="ai-jina-fallback" ${cfg.search?.jina_fallback !== false ? "checked" : ""}>
              Enable Jina Reader fallback in WebFetch when sites block plain HTTP or require JavaScript
            </label>
          </div>
        </div>
      `;
    } else if (tab === "endpoints") {
      body = `
        <p class="sub" style="margin-bottom:12px">Save multiple OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, OpenRouter, Groq, SECDATA) for 1-click switching.</p>
        <div id="ai-ep-list">
          ${cfg.custom_endpoints.length ? cfg.custom_endpoints.map((ep, idx) => `
            <div class="ai-ep-card">
              <div>
                <b>${esc(ep.name)}</b>
                <small>${esc(ep.base_url)} · model: ${esc(ep.model || "(none)")}${ep.has_key ? " · key saved" : ""}</small>
              </div>
              <div class="ai-row" style="flex:none">
                <button type="button" class="ai-btn" data-use-ep="${idx}">Use now</button>
                <button type="button" class="ai-btn" data-del-ep="${idx}">Delete</button>
              </div>
            </div>
          `).join("") : `<div class="ai-status">No custom endpoints saved yet. Add one below.</div>`}
        </div>
        <div class="ai-grid" style="margin-top:14px;padding-top:12px;border-top:1px solid var(--line, #d9d9d9)">
          <div class="ai-field"><label>Name</label><input id="ep-new-name" placeholder="e.g. Local Ollama"></div>
          <div class="ai-field"><label>Base URL</label><input id="ep-new-url" placeholder="http://localhost:11434/v1"></div>
          <div class="ai-field"><label>Default Model ID</label><input id="ep-new-model" placeholder="qwen2.5:32b"></div>
          <div class="ai-field"><label>API Key (optional)</label><input type="password" id="ep-new-key" placeholder="sk-... or blank"></div>
        </div>
        <button type="button" class="ai-btn" id="ep-add-btn">+ Save Endpoint</button>
      `;
    } else if (tab === "roles") {
      body = `
        <p class="sub" style="margin-bottom:12px">Optionally assign a different provider/model per task (e.g. local Ollama for the Firewall Secure Zone, Gemini/Claude/OpenAI for research). Leave blank to use the active model.</p>
        ${Object.entries(cfg.roles).map(([role, label]) => {
          const ov = cfg.overrides?.[role] || {};
          return `<div class="ai-grid" style="margin-bottom:8px;align-items:end">
            <div class="ai-field" style="margin:0">
              <label>${esc(label)} (${esc(role)})</label>
              <select data-ov-prov="${role}">
                <option value="">Use active default</option>
                ${Object.entries(cfg.providers).map(([id, info]) =>
                  `<option value="${id}" ${ov.provider === id ? "selected" : ""}>${esc(info.label)}</option>`
                ).join("")}
              </select>
            </div>
            <div class="ai-field" style="margin:0">
              <label>Model ID (and optional Base URL)</label>
              <div class="ai-row">
                <input data-ov-model="${role}" value="${esc(ov.model || "")}" placeholder="Default model">
                <input data-ov-url="${role}" value="${esc(ov.base_url || "")}" placeholder="Base URL (if compatible)">
              </div>
            </div>
          </div>`;
        }).join("")}
      `;
    }

    box.innerHTML = `
      <h2>AI Provider & Model Settings</h2>
      <p class="sub">Switch between OpenAI, Google Gemini, Anthropic Claude, and OpenAI-compatible endpoints.</p>
      <div class="ai-tabs">
        ${[["model", "Active Model"], ["keys", "API Keys"], ["endpoints", "Compatible Endpoints"], ["roles", "Per-Task Overrides"]].map(([id, label]) =>
          `<button type="button" class="ai-tab ${tab === id ? "active" : ""}" data-ai-tab="${id}">${label}</button>`
        ).join("")}
      </div>
      ${body}
      <div class="ai-foot">
        <span style="font-size:12px;color:var(--muted,#667085)">Active: <b>${esc(providerLabel(p))}</b> · <code>${esc(cfg.active.model)}</code> · Search: <b>${esc(cfg.search_status?.label || "Fallback")}</b></span>
        <div class="ai-row" style="flex:none">
          <button type="button" class="ai-btn" id="ai-close-btn">Cancel</button>
          <button type="button" class="ai-btn primary" id="ai-save-btn">Save & Apply</button>
        </div>
      </div>
    `;

    box.querySelectorAll("[data-ai-tab]").forEach((b) => b.onclick = () => { tab = b.dataset.aiTab; renderModal(); });
    document.getElementById("ai-close-btn").onclick = closeModal;
    document.getElementById("ai-save-btn").onclick = saveFromModal;

    if (tab === "model") {
      document.getElementById("ai-prov").onchange = (e) => {
        const nextP = e.target.value;
        cfg.active.provider = nextP;
        cfg.active.model = cfg.providers[nextP].models[0];
        if (nextP !== "openai_compatible") cfg.active.endpoint_id = "";
        else if (!cfg.active.base_url) cfg.active.base_url = cfg.providers[nextP].base_url;
        fetchedModels = [];
        renderModal();
      };
      if (document.getElementById("ai-ep-select")) {
        document.getElementById("ai-ep-select").onchange = (e) => {
          const epId = e.target.value;
          cfg.active.endpoint_id = epId;
          const ep = cfg.custom_endpoints.find((x) => x.id === epId);
          if (ep) {
            cfg.active.base_url = ep.base_url;
            if (ep.model) cfg.active.model = ep.model;
          }
          renderModal();
        };
      }
      box.querySelectorAll("[data-tpl]").forEach((b) => b.onclick = () => {
        const tpl = cfg.compat_templates.find((t) => t.id === b.dataset.tpl);
        if (!tpl) return;
        cfg.active.endpoint_id = "";
        cfg.active.base_url = tpl.base_url;
        cfg.active.model = tpl.model;
        renderModal();
      });
      box.querySelectorAll("[data-pick-model]").forEach((b) => b.onclick = () => {
        document.getElementById("ai-model").value = b.dataset.pickModel;
        cfg.active.model = b.dataset.pickModel;
      });
      document.getElementById("ai-fetch-models").onclick = async () => {
        const btn = document.getElementById("ai-fetch-models");
        btn.disabled = true; btn.textContent = "Loading...";
        try {
          const r = await fetch("/api/ai/models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              provider: document.getElementById("ai-prov").value,
              base_url: document.getElementById("ai-base-url")?.value || null,
              endpoint_id: document.getElementById("ai-ep-select")?.value || null,
            }),
          });
          const data = await r.json();
          if (!r.ok) throw new Error(data.detail || "Failed");
          fetchedModels = data.models || [];
          cfg.active.model = document.getElementById("ai-model").value;
          renderModal();
        } catch (err) {
          const out = document.getElementById("ai-test-out");
          out.style.display = "block"; out.className = "ai-status err";
          out.textContent = "Fetch models failed: " + err.message;
        } finally {
          btn.disabled = false; btn.textContent = "Fetch";
        }
      };
      document.getElementById("ai-test-btn").onclick = async () => {
        const out = document.getElementById("ai-test-out");
        out.style.display = "block"; out.className = "ai-status"; out.textContent = "Testing connection...";
        const r = await fetch("/api/ai/test", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            provider: document.getElementById("ai-prov").value,
            model: document.getElementById("ai-model").value,
            base_url: document.getElementById("ai-base-url")?.value || null,
            endpoint_id: document.getElementById("ai-ep-select")?.value || null,
          }),
        });
        const data = await r.json();
        if (r.ok && data.ok) {
          out.className = "ai-status ok";
          out.textContent = `Connected to ${data.provider} (${data.model}) in ${data.latency_ms} ms: "${data.reply}"`;
        } else {
          out.className = "ai-status err";
          out.textContent = data.detail || "Connection failed";
        }
      };
    }

    if (tab === "keys") {
      box.querySelectorAll("[data-clear-key]").forEach((b) => b.onclick = async () => {
        await fetch("/api/ai/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keys: { [b.dataset.clearKey]: "__CLEAR__" } }),
        });
        await loadConfig();
        renderModal();
      });
    }

    if (tab === "endpoints") {
      document.getElementById("ep-add-btn").onclick = async () => {
        const name = document.getElementById("ep-new-name").value.trim();
        const base_url = document.getElementById("ep-new-url").value.trim();
        const model = document.getElementById("ep-new-model").value.trim();
        const api_key = document.getElementById("ep-new-key").value.trim();
        if (!name || !base_url) return;
        const id = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || ("ep-" + Date.now());
        const next = [...cfg.custom_endpoints, { id, name, base_url, model, api_key, models: model ? [model] : [] }];
        const r = await fetch("/api/ai/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ custom_endpoints: next }),
        });
        cfg = await r.json();
        renderPill();
        renderModal();
      };
      box.querySelectorAll("[data-del-ep]").forEach((b) => b.onclick = async () => {
        const idx = Number(b.dataset.delEp);
        const next = cfg.custom_endpoints.filter((_, i) => i !== idx);
        const r = await fetch("/api/ai/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ custom_endpoints: next }),
        });
        cfg = await r.json();
        renderPill();
        renderModal();
      });
      box.querySelectorAll("[data-use-ep]").forEach((b) => b.onclick = async () => {
        const ep = cfg.custom_endpoints[Number(b.dataset.useEp)];
        if (!ep) return;
        const r = await fetch("/api/ai/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            active: { provider: "openai_compatible", endpoint_id: ep.id, base_url: ep.base_url, model: ep.model || cfg.active.model }
          }),
        });
        cfg = await r.json();
        tab = "model";
        renderPill();
        renderModal();
      });
    }
  }

  async function saveFromModal() {
    const payload = {};
    if (tab === "model") {
      payload.active = {
        provider: document.getElementById("ai-prov").value,
        model: document.getElementById("ai-model").value.trim(),
        base_url: document.getElementById("ai-base-url")?.value.trim() || "",
        endpoint_id: document.getElementById("ai-ep-select")?.value || "",
      };
    } else if (tab === "keys") {
      const keys = {};
      for (const k of ["openai", "gemini", "claude", "openai_compatible", "tavily", "serper", "brave", "jina"]) {
        const val = document.getElementById("ai-key-" + k)?.value.trim();
        if (val) keys[k] = val;
      }
      payload.keys = keys;
      payload.search = {
        backend: document.getElementById("ai-search-backend")?.value || "auto",
        searxng_url: document.getElementById("ai-searxng-url")?.value.trim() || "",
        jina_fallback: Boolean(document.getElementById("ai-jina-fallback")?.checked),
      };
    } else if (tab === "roles") {
      const overrides = {};
      for (const role of Object.keys(cfg.roles)) {
        const prov = document.querySelector(`[data-ov-prov="${role}"]`)?.value || "";
        const model = document.querySelector(`[data-ov-model="${role}"]`)?.value.trim() || "";
        const base_url = document.querySelector(`[data-ov-url="${role}"]`)?.value.trim() || "";
        if (prov || model) overrides[role] = { provider: prov, model, base_url };
      }
      payload.overrides = overrides;
    }
    const r = await fetch("/api/ai/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (r.ok) {
      cfg = await r.json();
      renderPill();
      window.dispatchEvent(new CustomEvent("ai-config-changed", { detail: cfg }));
      closeModal();
    }
  }

  window.aiWidget = { loadConfig, openModal, getConfig: () => cfg };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadConfig);
  } else {
    loadConfig();
  }
})();
"""


@router.get("/widget.js")
def widget_js():
    return Response(content=WIDGET_JS, media_type="application/javascript")
