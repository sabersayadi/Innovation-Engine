# Keyword firewall (prototype)

Research an invention without sending the invention to outside services. A model you control reads the disclosure and decides what may leave. The heavy research, meaning many web searches and long reports, runs at cloud providers but only ever sees a public description of the field. So no large in-house computer is needed: the secure zone only does two light tasks.

This is a standalone prototype, to be merged into the newer version of the app later. It reuses `run.py` (`agent_options`) and `tools.py` (data tools) through the provider-agnostic `llm.py` runtime.

## How it works

```
Disclosure (PDF or text)
   │  read on this computer
   ▼
1. SECURE ZONE   writes a public brief (field, problem, product type, customers,
                 search keywords, patent classes) and a private list of secret terms.
                 A local check flags secret terms, numbers and copied sentences in the brief.
   ▼
2. REVIEW        you see the brief, can edit public_brief.json, and press Enter (or n to stop)
   ▼
3. RESEARCH ZONE three cloud agents (patents, market, funding) research the field with web search
   (cloud)       and the data tools. They get only the brief and no file access, so they cannot
                 see the disclosure. Every outside request is logged and checked for secret terms.
   ▼
4. SECURE ZONE   compares the disclosure with their reports: novelty against the patents and papers
                 found, market fit, funding fit, risks, next steps, questions for the researchers
```

## Run it

From the project folder:

```bash
.venv/bin/python firewall/pipeline.py disclosures/test-leak-detector.pdf \
  --secure-provider openai_compatible --secure-model qwen2.5:32b --secure-base-url http://localhost:11434/v1 \
  --research-provider openai --research-model gpt-4.1
```

Each run gets a folder in `firewall/runs/<date>_<time>_<disclosure>/`:

| File | Contains confidential information? |
|---|---|
| `public_brief.json` | No. This is what left the secure zone |
| `outbound_log.md` | No. Every search and data request the research agents sent, and how many contain a secret term |
| `patents.md`, `market.md`, `funding.md` | No. Written by the research agents, who never saw the invention |
| `leak_check.txt` | Partly: it can quote secret terms the check found |
| `secret_terms.json` | **Yes** |
| `assessment.md` | **Yes**. It compares the invention with the findings |

## The secure-zone model

`secure_model()` in [pipeline.py](pipeline.py) is the only place the full disclosure goes. It uses the `firewall_secure` role override (configurable in AI Settings or via `--secure-provider`, `--secure-model`, `--secure-base-url`) with no tools or file access, so you can point it directly at a local OpenAI-compatible server (Ollama, vLLM, LM Studio, or Aalto's SECDATA environment) while the research zone runs on any cloud provider (`--research-provider`, `--research-model`). If `firewall_secure` is pointed at a cloud provider, use fictional or public disclosures only.

The local model's jobs are small. In the test run:
- Step 1 read about 1,000 tokens and wrote a short JSON brief.
- Step 4 read about 8,500 tokens (disclosure plus three reports) and wrote a 2-page assessment.

A model with a 16,000-token context window is enough. That is workstation-sized, not a data centre.

## Test results (fictional leak-detector disclosure, 28 Oct 2026)

- **The brief stayed at field level.** It covered district heating leak detection, correlators, noise loggers, thermal imaging, customer types and CPC classes, with nothing about the self-calibration (the inventive step), results, names or dates.
- **The secret terms list had 52 entries:** the inventive step, all numbers, names, the partner and the dates.
- **The research agents sent 21 outside requests, and none contained a secret term.** The whole run took 148 seconds.
- **The assessment separated known from new.** It found acoustic detection in district heating and electrical pulses on alarm wires already known, and judged self-calibrating acoustic pulses per pipe section as looking new. It also said this needs a confidential professional search.

## Limits

- **The secret terms check matches exact phrases only.** Generic secret terms (for example "energy company") cause harmless flags. A rephrased secret would not be caught, which is why the human review of the brief matters most.
- **The novelty search covers the field, not the invention's own features.** Searching for the features would reveal them. The assessment says so, and a confidential professional patent search is still needed.
- **Everything else still goes to the cloud.** The research reports and the brief go to Anthropic and to the search and data services. They contain no invention details, but they do show that someone is interested in this field.
- **Inputs are PDF and text files only,** read on this computer. Images are not supported.
