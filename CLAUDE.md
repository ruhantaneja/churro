# CLAUDE.md — instructions for Claude Code working in this repo

This file loads automatically at the start of every Claude Code session
here. Read it before touching anything else.

## What CHURRO is

CHURRO is a model-independent AI development workspace. Core principle:
**"The model can change. The workspace isn't reset."** Project state, task
state, workspace, tools, and continuity belong to CHURRO — not to any one
LLM. Long-term it should become a general AI workspace (coding, research,
writing, planning, business work); coding is the current engineering focus.

Product thesis: a task can start with Model A. Model A can fail, hit a
limit, or be swapped out. CHURRO saves task state. Model B inspects the
same workspace and a compact continuation handoff, and continues the SAME
task — the project does not reset when the model changes.

Persistence philosophy: the model's internal conversation is disposable.
CHURRO owns goal, current task, completed work, blockers, relevant files,
next action, constraints, decisions, and task continuity. The filesystem
is the strongest source of truth for work actually performed — we do not
solve continuity by dumping old conversations into a new model; we use
compact state + workspace inspection instead.

## Repo layout

- `churro/`
  - `core/` — `Session`, `SessionManager`, `AgentFrame`/handoff (`handoff.py`,
    `session.py`), parser, validator
  - `agent/runner.py` — `AgentRunner`, the bounded autonomous tool-calling loop
  - `providers/` — provider abstraction + `ProviderFactory`, OpenAI,
    OpenAI-compatible, Ollama providers
  - `tools/` — `read_file`, `write_file`, `list_files`, `search_files`,
    `run_command`, `git_diff`, `run_tests`, `ToolExecutor`, tool registry,
    workspace isolation
  - `prompts/system.txt` — the system prompt CHURRO feeds to the **models it
    orchestrates** (OpenAI/Ollama, inside CHURRO's own agent loop). This is
    product code, not an instruction to you. Don't confuse it with this file.
  - `main.py` — CLI entrypoint / REPL (`/status`, `/save`, `/switch`,
    `/agent`, `/handoff`, `/files`, `/help`, `/quit`)
  - `repo_index.py` — repository overview / structure intelligence
- `tests/` — pytest suite, roughly one file per module
- `benchmarks/step19/` — a benchmark harness with a baseline project used to
  score agent runs; excluded from pytest via `pytest.ini` (`norecursedirs`)
- `sessions/*.json` — persisted CHURRO sessions (runtime data, not source —
  don't hand-edit these as if they were config)
- `.env.example` — env var documentation (`OPENAI_API_KEY`, `OLLAMA_MODEL`,
  `OLLAMA_BASE_URL`, `CHURRO_PROVIDER`, `CHURRO_MODEL`). Never put a real
  key in code or commit a real `.env`.

Remote: `origin` → `https://github.com/ruhantaneja/churro.git`, branch `main`.

## Run it

```powershell
python -m pip install -r requirements.txt
python -m churro                 # starts the CHURRO REPL
python -m pytest                 # full suite (pytest.ini excludes benchmarks/)
```

Known pre-existing failure — do not "fix" it unless specifically asked:
`tests/test_provider_factory.py::test_factory_imports_no_provider_sdks`

## Roadmap

v0.1 Foundation → v0.2 Tool System → v0.3 Autonomous Agent → v0.4 Repository
Intelligence → **v0.5 Reliable Model Switching + Continuation (current)** →
v0.6 Multi-agent → v0.7 Model router → v0.8 Agent replay/audit → v0.9 Visual
canvas → v1.0 Fine-tuned local model. Full detail on shipped versions is in
`README.md`.

### Current status — verify before trusting

Latest commit on `main` as of the last recorded session: `e3ca0a6` — "add
/resume to continue an interrupted agent task".

v0.5 in progress:
- Step 1 (`AgentFrame`, `Session.pending_agent`), Step 2 (capture provider
  failures / iteration-limit hits / Ctrl+C interruptions, compact tool
  digest), and Step 3 (`/resume`) are implemented and committed.

This section can go stale the moment work happens outside a tracked
session. At the start of any new session: run `git status` and
`git log --oneline -5`, diff that against the paragraph above, and treat
the actual repo state as ground truth, not this file.

## Development rules — do not violate these

- Small incremental changes. Prefer minimal diffs over rewrites.
- Provider-neutral architecture: nothing in `core/` or `agent/` should
  assume OpenAI or Ollama specifically.
- Deterministic behavior; tests run offline (no live network calls).
- Compact persistent state — never treat a full conversation dump as
  "state."
- The workspace/filesystem is the strongest source of truth for what work
  was actually done — more than any conversation or state blob.
- Do **not** prematurely add: RAG, embeddings, vector databases, model
  routers, multi-agent orchestration, visual UI, fine-tuning, distributed
  systems — unless the roadmap above has explicitly reached that step.
- Do **not** start UI implementation. The engine comes first; UI is a
  dedicated later phase.
- Any change touching `core/`, `agent/`, `providers/`, or `tools/` needs a
  passing test.

## Product vision (not current scope — read only when asked)

The long-term product vision — the future UI/interaction phase and
brainstorm-level ideas beyond the current engineering roadmap — lives in
`docs/PRODUCT_VISION.md`. It is explicitly NOT a spec to build against yet.
Don't pull it into context on a normal engineering task; read it only when
the user asks about product direction, the UI phase, or long-term design,
by referencing `@docs/PRODUCT_VISION.md`.

## Working style for this repo

1. Before writing code, state your plan in a couple of sentences (files
   touched, approach) and check it against the rules above.
2. After writing code, run `python -m pytest`, report the pass/fail counts,
   and call out the known pre-existing failure separately from anything new
   you may have introduced.
3. Commit in small, clearly scoped commits. This repo's history uses short,
   lowercase, imperative subjects (e.g. "add repository structure
   intelligence", "build reliable autonomous agent core").
4. Never commit `.env`, real API keys, or `sessions/*.json` files produced
   by your own test/dev runs.
