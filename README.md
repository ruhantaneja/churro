# CHURRO v0.1

A model-agnostic AI coding environment.

**The AI model can change, but the work should never reset.**

CHURRO owns the persistent project state. The AI model only proposes
changes; CHURRO validates, merges, and stores them. Switching models uses a
compact, structured handoff generated from that state — not the previous
conversation.

## What v0.1 does

- Interactive terminal that talks to OpenAI (Chat Completions)
- Persistent, validated project state: goal, current task, completed work,
  status, blockers, relevant files, next action, notes
- The AI ends each reply with a `<state_update>` block that CHURRO parses,
  validates, and merges — only safe fields are persisted
- `completed_work` accumulates; `blockers` and `relevant_files` are replaced
  by the latest proposal
- Model switching (OpenAI model to another OpenAI model) with a compact
  handoff; the old conversation is archived locally but never sent to the
  new model
- `/handoff` shows exactly what another model would receive, with an
  approximate token count

## Install

```powershell
python -m pip install -r requirements.txt
```

## Configure the API key

Set the OpenAI key as an environment variable (see `.env.example`):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Never put a real key in code or commit one. CHURRO reads the key only from
the environment.

## Start CHURRO

```powershell
python -m churro
```

First launch asks for a goal, creates a session, and saves it to
`sessions/{id}.json`. You can resume a session by passing its file:

```powershell
python -m churro sessions/<session-id>.json
```

## Commands

| Command | Action |
|---|---|
| `<message>` | Send a message to the active model |
| `/status` | Show the persistent project state |
| `/save` | Save the session now |
| `/switch <provider[:model]>` | Switch provider/model (v0.1: `openai` only) |
| `/handoff` | Show the handoff another model would receive |
| `/files` | Show relevant files from state |
| `/help` | List commands |
| `/quit` | Save and exit |

## Example interaction

```
> The login fails for users with unicode email addresses.
AI: I found the bug. The regex at auth.py:42 uses [a-zA-Z0-9] and rejects unicode.

> /handoff
You are taking over an existing coding task in CHURRO.

PROJECT STATE

Goal:
Fix the unicode email login bug
...

Handoff size: ~121 tokens (483 chars)

> /switch openai:gpt-4o-mini
Switched openai/gpt-4o -> openai/gpt-4o-mini
The new model starts from a fresh handoff; the old conversation context was archived.
```

Only `openai` exists in v0.1. `/switch anthropic` reports a clear error and
leaves the session untouched. The architecture (a provider registry, a
shared provider interface, and handoffs built from state) is what makes
additional providers a later, additive step.

## Offline demo (no API key)

```powershell
python -m examples.demo_interactive
```

Simulates a full session, `/status`, `/handoff`, and `/switch` with fake AI
replies so you can explore the UX for free.

The tool-system demo registers and executes fake tools, showing controlled
failures, without any API key:

```powershell
python -m examples.demo_tools
```

A real-API smoke test is also available once a key is set:

```powershell
python examples/demo_openai_provider.py
```

## Tests

404 tests, all offline — no API key or network required:

```powershell
$env:PYTHONPATH = "D:\AI PROJECT"
python tests/test_parser_validator.py
python tests/test_session_manager.py
python tests/test_handoff.py
python tests/test_openai_provider.py
python tests/test_cli_app.py
python tests/test_tools.py
python tests/test_read_file.py
python tests/test_list_files.py
python tests/test_search_files.py
python tests/test_write_file.py
python tests/test_run_command.py
python tests/test_git_diff.py
python tests/test_run_tests.py
python tests/test_tool_calling.py
python tests/test_tool_executor.py
python tests/test_result_messages.py
python tests/test_agent_runner.py
python tests/test_openai_tool_messages.py
python tests/test_ollama_provider.py
```

## Directory layout

```
churro/
  main.py                 CLI loop and CHURROApp
  __main__.py             python -m churro entry point
  agent/
    runner.py             AgentRunner: bounded, provider-neutral agent loop
    __init__.py           agent package exports
  core/
    session.py            Session, SessionState, persistence
    session_manager.py    parse -> validate -> merge for AI responses
    parser.py             split AI reply into text + <state_update>
    validator.py          field-by-field validation of proposals
    handoff.py            state -> handoff string + size estimates
  providers/
    provider.py           Provider interface + error hierarchy
    responses.py          ProviderResponse, ToolCall, ToolResultMessage, Usage
    tool_definition.py    ToolDefinition (LLM-facing tool spec)
    openai_provider.py    OpenAI Chat Completions provider (openai SDK)
    ollama_provider.py    Ollama provider (stdlib HTTP only, no SDK)
    openai_compat.py      shared OpenAI-compatible wire format translation
  tools/
    tool.py               Tool, ToolArgs, ToolResult, and tool error types
    registry.py           ToolRegistry: register / get / list / execute
    workspace.py          workspace containment, relative display, file walk
    read_file.py          read_file: workspace-confined UTF-8 file reader
    list_files.py         list_files: deterministic workspace tree listing
    search_files.py       search_files: literal text search with line info
    write_file.py         write_file: atomic workspace-confined file writer
    run_command.py        run_command: workspace-cwd shell execution
    git_diff.py           git_diff: read-only Git diff inspection
    run_tests.py          run_tests: runs the project test suite
    executor.py           ToolExecutor: normalized ToolCall -> ToolResult
    result_messages.py    ToolResult -> provider-neutral ToolResultMessage
    subprocess_runner.py  shared subprocess capture/timeout/termination
  prompts/system.txt      instructions given to every model
examples/                 demos (offline sim, parser/validator, handoff, provider)
tests/                    all automated tests
sessions/                 session JSON files (created at runtime)
```

## Design principle

`conversation_history` is ephemeral — useful for the current model, archived
(not forwarded) on switch. `state` is persistent, owned by CHURRO, and is
the only thing a new model receives. The session file is the single source
of truth.

## Tool system (v0.2)

CHURRO is gaining a generic tool system. All v0.2 tools are implemented;
wiring them to the LLM is the next step. Nothing is wired yet.

```
churro/tools/tool.py
  Tool       base class; a tool owns its name, description, an argument
             model, and its implementation
  ToolArgs   required base for every tool's argument model; unknown
             arguments are rejected (extra="forbid")
  ToolResult success / output / error returned by every execution
  ToolError  base error type (+ ToolDefinitionError, UnknownToolError,
             DuplicateToolError)

churro/tools/registry.py
  ToolRegistry  register / get / list_tools / execute(name, args)

churro/tools/workspace.py
  workspace containment (resolve_within_workspace / relative_display),
  plus a deterministic file walker (iter_files), shared by every
  filesystem tool

churro/tools/read_file.py, list_files.py, search_files.py, write_file.py,
run_command.py, git_diff.py, run_tests.py
  ReadFileTool / ListFilesTool / SearchFilesTool / WriteFileTool /
  RunCommandTool / GitDiffTool / RunTestsTool, added without touching the
  registry

churro/tools/subprocess_runner.py
  run_captured / terminate_tree, the shared no-raise subprocess launch,
  capture, timeout, and process-tree termination used by both run_command
  and run_tests
```

The registry knows how to dispatch, never the implementation — each tool
is an independent `Tool` subclass, and adding one means registering a new
class without touching the registry:

```
ToolRegistry  ->  read_file tool  ->  filesystem
```

`execute(name, args)` always returns a `ToolResult`. Invalid arguments,
unknown tools, and tools that raise mid-run are captured as `success=False`
results, so CHURRO never crashes on a bad tool call. Registering a
duplicate name raises `DuplicateToolError`. Execution is isolated from the
session system and the LLM for now.

### read_file

Reads a UTF-8 text file and returns its contents. The workspace root is
passed to the tool at construction:

```python
from churro.tools import ReadFileTool, ToolRegistry

registry = ToolRegistry()
registry.register(ReadFileTool(r"D:\AI PROJECT"))

result = registry.execute("read_file", {"path": "churro/main.py"})
```

Security: every request is resolved as a real path and must stay inside the
workspace root. `../` traversal, absolute paths elsewhere, and symlinks
pointing outside are all rejected with a controlled `success=False` result.
The tool only reads — it never modifies the file.

### list_files

Lists files and directories as workspace-relative paths. `FILE`/`DIR` rows
are fixed-width (`FILE  path`, `DIR   dir/`) and ordered deterministically
by name. `recursive` is optional (default `false`):

```python
result = registry.execute("list_files", {"path": ".", "recursive": True})
# FILE  churro/main.py
# DIR   tests/
```

All paths in the output are workspace-relative and forward-slash separated;
no timestamps or permissions are included. Missing directories and requests
pointing at files fail with a controlled `success=False` result. Recursion
never descends symlinked directories or anything outside the workspace.

### search_files

Searches UTF-8 text files for a literal, case-sensitive string and reports
workspace-relative file, line number, and matching line:

```python
result = registry.execute("search_files", {
    "query": "SessionState", "path": ".", "recursive": True,
})
# churro/core/session.py:15:class SessionState(BaseModel):
```

`max_results` (default 50, must be a positive integer) caps the matches
returned; if more exist, a final `... results truncated at N matches` line
is appended. Traversal is deterministic (sorted, depth-first), files that
cannot be decoded as UTF-8 or that vanish mid-search are skipped, and
symlinked directories are never followed. Empty queries and missing or
non-directory start paths return controlled `success=False` results.

### write_file

Creates a new file or replaces an existing one with the given UTF-8
content. The write is atomic — content goes to a hidden temp file in the
same directory, is flushed and `fsync`ed, then `os.replace`d over the
target, so a crash never leaves a half-written file:

```python
result = registry.execute("write_file", {
    "path": "notes.txt", "content": "hello",
})
# Created notes.txt   (or "Updated notes.txt" when replacing)
```

The target and every parent must already exist inside the workspace;
parent directories are never auto-created and a missing parent fails with
a controlled `success=False` result. The final path element must not be a
directory. Content is written byte-for-byte, so newlines (including CRLF)
and unicode are preserved exactly. When replacing an existing file, its
permission mode is preserved; a failed write never touches an unrelated
file and leaves the original target intact.

Security: paths are resolved with the same containment rules as
`read_file` — `../` traversal, absolute paths elsewhere, and symlinks
pointing outside the workspace (including symlinked parent directories)
are all rejected, and temp-file cleanup happens even on failure. Output
never includes the absolute workspace path.

### run_command

Executes a shell command with the workspace as its working directory and
returns the captured output. `command` is passed exactly as supplied to the
platform shell; `timeout_seconds` (default 30, max 300, positive integer)
caps runtime:

```python
result = registry.execute("run_command", {
    "command": "python tests/test_tools.py",
    "timeout_seconds": 120,
})
# Exit code: 0
#
# STDOUT:
# All 18 tests passed.
#
# STDERR:
# (empty)
```

`success=True` only for exit code 0; non-zero exits, command-not-found,
launch failures, and timeouts all return a controlled `success=False`
result with the exit code (or `killed on timeout`) and any partial output.
stdout and stderr are captured separately and labelled. Captured output is
capped at 20,000 characters total by default (configurable at construction
via `max_output_chars`); a truncated stream is marked with
`... [output truncated at N characters]`. On timeout the process tree is
terminated (`taskkill /T /F` on Windows) so the tool never hangs CHURRO.

> DANGER: this tool runs arbitrary, trusted commands with real shell access
> and is NOT sandboxed. Workspace confinement only sets the initial working
> directory; a command can read or write any file or reach any network
> resource the process can. Do not call it with untrusted input. Command
> allowlists, Docker, containers, and OS-level sandboxing are deliberately
> out of scope for v0.2.

### git_diff

Read-only Git inspection of uncommitted changes. `staged` (default false)
selects the working-tree diff or the staged (`--cached`) diff; the optional
`path` (default `.`) restricts the diff to a workspace-relative path:

```python
result = registry.execute("git_diff", {"path": "churro/core/session.py"})
# Git diff:
# --- a/churro/core/session.py
# +++ b/churro/core/session.py
# ...
```

Git is invoked as a subprocess argument list (`git --no-pager diff
[--cached] -- <path>`) with `shell=False`, always with the workspace as its
cwd; the caller can never choose a cwd. The path is validated through the
same `resolve_within_workspace` containment rules — `../` traversal,
absolute paths elsewhere, and symlink escapes are rejected, and only
workspace-relative paths reach Git. A clean repository returns
`success=True` with `Git diff:\nNo changes.`; a path that does not exist
returns a controlled failure. Diff output is capped at 20,000 characters by
default (constructor-configurable via `max_output_chars`) and marked with
`... [diff truncated ...]` when cut. Git's error text and normal output are
redacted so no absolute host paths appear.

Strictly read-only: it never stages, commits, resets, or writes anything
(`GIT_OPTIONAL_LOCKS=0`, `GIT_PAGER=cat`). Note that Git ignores untracked
files by design, so brand-new files do not appear in the diff — only
changes to tracked files (including staged new files) are visible.

### run_tests

Runs the project's test suite from the workspace and returns structured
results. With no `command`, it detects pytest (via `find_spec` on the
current interpreter) and runs `python -m pytest` as an argument list; if
pytest is unavailable it fails cleanly with `No test runner detected`,
rather than inventing a framework. Only Python/pytest detection is
supported:

```python
result = registry.execute("run_tests", {"timeout_seconds": 300})
# Test command:
# python -m pytest
#
# Exit code: 0
#
# STDOUT:
# 1 passed in 0.01s
#
# STDERR:
# (empty)
#
# Tests passed.
```

A provided `command` is executed exactly as given (shell execution, like
`run_command` — trusted, NOT sandboxed — so project-specific commands work
unchanged). `timeout_seconds` defaults to 120, must be a positive integer,
and is capped at 300; on timeout the process tree is terminated and any
partial output is preserved. `success=True` only for exit code 0. stdout
and stderr are capped separately to ~20,000 characters total
(constructor-configurable via `max_output_chars`, plus a 200-character cap
on the echoed command) with explicit truncation markers. The workspace is
always the cwd and there is no `cwd` argument.

### Tool calling normalization (v0.2)

Before wiring tools to an LLM, the provider layer needed a normalized shape
that CHURRO's core can consume without importing provider-specific SDK
types. This lives in `churro/providers/responses.py` and
`churro/providers/tool_definition.py`:

```
churro/providers/responses.py
  ToolCall          id / name / arguments (parsed dict, never raw JSON)
  Usage             prompt_tokens / completion_tokens / total_tokens (all optional)
  ProviderResponse  text / tool_calls / finish_reason / usage
  normalize_finish_reason  lowercases and aliases provider-specific
                           reasons ("function_call" -> "tool_calls", etc.)

churro/providers/tool_definition.py
  ToolDefinition    name / description / input_schema
  from_tool()       builds a definition from a churro Tool (reuses
                    tool.input_schema — Tool depends on no OpenAI types)
```

`Provider.send()` remains abstract and returns `str` for backwards
compatibility. A new `Provider.send_normalized()` method (non-abstract,
default wrapping `send()`) returns `ProviderResponse` so every existing
provider keeps working with no changes. `OpenAIProvider` overrides it
to send tool definitions and translate `message.tool_calls` into parsed
`ToolCall` objects; malformed JSON arguments raise a controlled
`ToolArgumentsError` rather than surfacing a raw parsing exception.

Providers never execute tools, never call the ToolRegistry, and never
parse `<state_update>` blocks — those remain the core's responsibility.
The OpenAI adapter translates only at the boundary:

```
ToolDefinition  ->  {"type":"function", "function":{...}}  ->  OpenAI API
OpenAI response ->  message.tool_calls                      ->  ToolCall list
```

### Tool execution coordinator (v0.2)

`churro/tools/executor.py` performs one controlled cycle that connects a
normalized provider response to the tool system:

```
ProviderResponse -> ToolCall -> ToolExecutor -> ToolRegistry.execute
                                                     -> ToolResult
                                                     -> ToolExecutionResult
                  ... (one per ToolCall, in order) -> ToolExecutionBatch
```

```
churro/tools/executor.py
  ToolExecutionResult  call_id / tool_name / result (a ToolResult)
  ToolExecutionBatch   executions (ordered); has_executions / all_succeeded
  ToolExecutor         execute(response) -> ToolExecutionBatch
```

A response with no tool calls returns an empty batch. Each `ToolCall` is
dispatched by name through the registry with its structured argument
dictionary passed through unchanged; the `call_id`, `tool_name`, and full
`ToolResult` (success / output / error) are preserved, and ordering is
kept. Unknown tools, invalid arguments, tools that return a failure, and
tools that raise mid-run all become controlled `success=False` results —
one failing call never stops the calls after it.

The coordinator is deliberately narrow and provider-neutral: it imports
only the normalized `ToolCall`/`ProviderResponse` types and the existing
`ToolRegistry` — no OpenAI/Anthropic/Gemini/Ollama SDK, no LLM call, no
SessionState or conversation-history mutation, and no autonomous loop.
Sending these results back to an LLM is the next (agent-loop) step.

`ToolDefinition.from_tool(...)` still describes the tools available to a
provider; automatically discovering/registering every tool and forwarding
the definitions to the provider is intentionally not wired up yet.

### Tool-result messages (v0.2)

`churro/providers/responses.py` defines the result-side counterpart of
`ToolCall` — a provider-neutral `ToolResultMessage`:

```
ToolResultMessage  call_id / tool_name / content / success
```

`churro/tools/result_messages.py` performs the pure conversion:

```
ToolExecutionResult  ->  ToolResultMessage              (one call)
ToolExecutionBatch   ->  list[ToolResultMessage]        (ordered)
```

For a successful `ToolResult`, `content` is the tool's output; for a failed
one it is the error text. Empty output/error is replaced with a fixed,
deterministic placeholder. `call_id`, `tool_name`, and `success` are
preserved, ordering follows the batch exactly, and empty batches convert to
an empty list. The conversion never mutates the batch, the execution, or the
underlying `ToolResult`, and it imports only churro's normalized types.

This is deliberately the last provider-neutral hop:

```
ToolCall  ->  ToolExecutor  ->  ToolResult  ->  ToolResultMessage
                                                    ^
                              provider adapter translates this into its
                              native message format (a later step)
```

No provider-specific message objects are created here, and no OpenAI-style
tool message is invented. The provider abstraction needed no change for this
step.

### Bounded agent loop (v0.2)

`churro/agent/runner.py` adds the first autonomous-but-bounded cycle. It is
pure orchestration: nothing here knows which LLM is on the other side.

```
AgentRunner.run(messages)
    └─ for up to max_iterations:
         provider.send_normalized(conversation, tools=definitions,
                                  tool_results=tool_results)
              └─ ProviderResponse
         has tool calls?
            ├─ NO  -> return AgentResult(completed=True, final_text=...)
            └─ YES -> ToolExecutor -> ToolExecutionBatch
                       -> result_messages_from_batch -> ToolResultMessage[]
                       -> feed back to the provider, loop again
    └─ budget exhausted -> AgentResult(completed=False, ...)
```

Everything is injected — the provider, the `ToolRegistry`, and
`max_iterations` (positive integer, default 10):

```python
runner = AgentRunner(provider=provider, registry=registry, max_iterations=10)
result = runner.run(messages=[{"role": "user", "content": "..."}])
```

`AgentResult` is a small, normalized report: `final_text`, `completed`,
`iterations`, `error`, `tool_executions`, `tool_results`, and `messages`. A
run stops when the provider returns no tool calls, when a ProviderError is
surfaced as `error`, or when the iteration budget runs out (`completed=False`
— the runner never makes an extra provider call or tool run past the limit).

Notes:
- Tool definitions come from `registry.list_tools()` via
  `ToolDefinition.from_tool`, reusing the existing schema machinery — never
  duplicated.
- Tool results are fed back as provider-neutral `ToolResultMessage` objects;
  the interface extension is `send_normalized(..., tool_results=...)`, which
  text-only providers accept and ignore.
- After each tool round the runner appends the assistant's tool-call turn
  (role `assistant` + its normalized `ToolCall`s) to the conversation, so a
  provider adapter can reconstruct which calls the tool results answer.
- The runner copies its input messages, never mutates batches, responses, or
  tool results, touches no SessionState/history, holds no global state, and
  imports no provider SDK. Session integration belongs to a later step.

### Provider-native tool-result formatting (v0.2)

`ToolResultMessage` is provider-neutral, but a real provider needs native
messages. The normalized message layer carries just enough information for
an adapter to rebuild a complete tool-call turn:

- `ChatMessage` gained an optional `tool_calls: list[ToolCall]` (an
  assistant message may carry the normalized calls that produced results).
- `AgentRunner` preserves each assistant tool-call turn in the conversation
  log, right before the results that follow it.

`OpenAIProvider.send_normalized` is the only place OpenAI-shaped formatting
exists (all inside `churro/providers/openai_provider.py`):

```
normalized assistant message with tool_calls
        ↓
{ role: "assistant", tool_calls: [
    { id, type: "function",
      function: { name, arguments: json.dumps(...) } }, ... ] }

ToolResultMessage (by call_id)
        ↓
{ role: "tool", tool_call_id: <call id>, content: <result content> }
```

Built messages interleave each assistant tool-call turn with its matching
`tool` results by call ID (order follows the assistant's call order, so
results attach to the correct calls even if supplied out of order).
Failures are formatted exactly like successes. The runner and every other
core module still contain no OpenAI formatting or SDK imports — adding
another provider adapter later requires no changes outside that provider.

Existing behavior is untouched: `Provider.send(...) -> str` and
`send_normalized(...)` keep working, and text-only providers need no
changes.

### Local Ollama provider (v0.2)

`churro/providers/ollama_provider.py` is a free, local provider that proves
the provider layer is truly model-agnostic: the exact same normalized
tool-calling flow works against a cloud API (OpenAI) or a local server
(Ollama) with no changes to `AgentRunner`, `ToolExecutor`, or any core
module.

```
AgentRunner  ->  OllamaProvider  ->  http://localhost:11434/v1/chat/completions
                          ^ urllib only — no openai / ollama SDK
```

- Configurable model and base URL, via constructor arguments or the
  `OLLAMA_MODEL` / `OLLAMA_BASE_URL` environment variables (see
  `.env.example`). The base URL defaults to
  `http://localhost:11434/v1`, Ollama's OpenAI-compatible endpoint.
- Uses only the Python standard library (`urllib`). No third-party SDK and
  no API key. A missing model at construction raises `ProviderError` with a
  clear message instead of failing mid-run.
- Connection failures (server not running, HTTP errors, timeouts) map to
  the same controlled `APIRequestError`; malformed responses and tool calls
  surface as `UnexpectedResponseError` / `ToolArgumentsError`, exactly as
  the OpenAI adapter does.
- The `client` argument is an injectable callable `(url, payload, timeout)`
  so the whole provider is testable offline; the test suite never contacts
  Ollama.

All OpenAI wire-format translation (native tool-call arrays, `tool` result
messages, response parsing) now lives once in
`churro/providers/openai_compat.py`, shared by both adapters. Each provider
adapter keeps only its own transport: OpenAI uses the `openai` SDK,
Ollama uses `urllib`. No provider-specific formatting is duplicated, and no
core module imports either adapter's SDK.