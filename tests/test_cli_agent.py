import io
import json
import os
import tempfile
from pathlib import Path

from rich.console import Console

from churro.agent import DEFAULT_MAX_ITERATIONS
from churro.core.session import create_session, load_session
from churro.main import CHURROApp
from churro.providers.provider import APIRequestError, Provider
from churro.providers.responses import ProviderResponse, ToolCall
from churro.tools import (
    Tool,
    ToolArgs,
    ToolRegistry,
    ToolResult,
    default_tool_registry,
)


class PingArgs(ToolArgs):
    pass


class PingTool(Tool[PingArgs]):
    name = "ping"
    description = "Succeed and reply 'pong' (fake tool)."
    args_model = PingArgs

    def _run(self, args: PingArgs) -> ToolResult:
        return ToolResult(success=True, output="pong")


class BoomArgs(ToolArgs):
    pass


class BoomTool(Tool[BoomArgs]):
    name = "boom"
    description = "Always fails; demonstrates controlled tool errors (fake tool)."
    args_model = BoomArgs

    def _run(self, args: BoomArgs) -> ToolResult:
        raise RuntimeError("the fake tool exploded")


class AgentFakeProvider(Provider):
    name = "openai"

    def __init__(self, model="gpt-4o", responses=(), name=None, error=None, repeat_last=False):
        self.name = name or self.name
        self.model = model
        self.responses = list(responses)
        self.error = error
        self.repeat_last = repeat_last
        self.sent_messages = []
        self.agent_calls = []
        self.send_calls = 0

    def send(self, messages):
        self.send_calls += 1
        self.sent_messages.append(messages)
        return "No canned reply."

    def send_normalized(self, messages, tools=None, tool_results=None):
        self.agent_calls.append(
            {"messages": messages, "tools": tools, "tool_results": tool_results}
        )
        if self.error is not None:
            raise self.error
        if not self.responses:
            return ProviderResponse(text="fallback done.", finish_reason="stop")
        if not self.repeat_last:
            return self.responses.pop(0)
        item = self.responses[-1]
        while len(self.responses) > 1:
            self.responses.pop(0)
        return item


def tool_response(name="ping", text="", arguments=None, call_id="c1"):
    return ProviderResponse(
        text=text,
        tool_calls=[
            ToolCall(id=call_id, name=name, arguments=arguments or {})
        ],
        finish_reason="tool_calls",
    )


def final_response(text):
    return ProviderResponse(text=text, finish_reason="stop")


def final_with_state(text, update):
    block = "<state_update>\n" + json.dumps(update, indent=2) + "\n</state_update>"
    return ProviderResponse(text=text + "\n" + block, finish_reason="stop")


class SequenceProvider(Provider):
    """Pops canned items; exception instances are raised when reached.

    Lets a test script an interrupted run: a tool call first, then a
    ``KeyboardInterrupt`` or a ``ProviderError`` on the next iteration.
    """

    name = "openai"

    def __init__(self, responses, model="gpt-4o"):
        self.model = model
        self.responses = list(responses)
        self.requests = []

    def send(self, messages):
        return "unused"

    def send_normalized(self, messages, tools=None, tool_results=None):
        self.requests.append({"messages": messages})
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def default_registry():
    registry = ToolRegistry()
    registry.register(PingTool())
    registry.register(BoomTool())
    return registry


def new_session():
    session = create_session(goal="Fix the login bug", provider="openai", model="gpt-4o")
    session.state.status = "in_progress"
    session.state.relevant_files = ["auth.py"]
    return session


def make_agent_app(
    session,
    provider,
    lines,
    provider_registry=None,
    tool_registry=None,
    agent_max_iterations=None,
):
    outputs = io.StringIO()
    console = Console(file=outputs, force_terminal=False, width=120, markup=False)
    queued = list(lines)
    app = CHURROApp(
        session=session,
        provider=provider,
        system_prompt="SYSTEM_RULES",
        session_path=str(Path(tempfile.mkdtemp()) / f"{session.session_id}.json"),
        provider_registry=provider_registry,
        tool_registry=tool_registry,
        agent_max_iterations=agent_max_iterations,
        input_fn=lambda prompt="": queued.pop(0),
        console=console,
    )
    return app, outputs


def run(app, outputs):
    code = app.run()
    return code, outputs.getvalue()


def test_agent_command_recognized_and_answered():
    provider = AgentFakeProvider(responses=[final_response("All set.")])
    app, outputs = make_agent_app(new_session(), provider, ["/agent tidy up", "/quit"])
    code, text = run(app, outputs)

    assert code == 0
    assert "Unknown command" not in text
    assert "Starting agent" in text
    assert "All set." in text


def test_agent_empty_task_shows_usage():
    provider = AgentFakeProvider(responses=[final_response("nope")])
    app, outputs = make_agent_app(new_session(), provider, ["/agent", "/quit"])
    code, text = run(app, outputs)

    assert code == 0
    assert "Usage: /agent <task>" in text
    assert provider.agent_calls == []
    assert app.session.conversation_history == []


def test_agent_whitespace_task_shows_usage():
    provider = AgentFakeProvider(responses=[final_response("nope")])
    app, outputs = make_agent_app(new_session(), provider, ["/agent    ", "/quit"])
    run(app, outputs)
    assert "Usage: /agent <task>" in outputs.getvalue()
    assert provider.agent_calls == []


def test_agent_task_reaches_runner():
    provider = AgentFakeProvider(responses=[final_response("ok")])
    app, outputs = make_agent_app(new_session(), provider, ["/agent search auth.py", "/quit"])
    run(app, outputs)

    sent = provider.agent_calls[0]["messages"]
    assert sent[-1]["role"] == "user"
    assert sent[-1]["content"] == "search auth.py"


def test_agent_receives_system_prompt_and_state():
    provider = AgentFakeProvider(responses=[final_response("ok")])
    app, outputs = make_agent_app(new_session(), provider, ["/agent do it", "/quit"])
    run(app, outputs)

    sent = provider.agent_calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "SYSTEM_RULES" in sent[0]["content"]
    assert "CURRENT PROJECT STATE" in sent[0]["content"]


def test_agent_final_response_displayed():
    provider = AgentFakeProvider(responses=[final_response("Found regex in auth.py.")])
    app, outputs = make_agent_app(new_session(), provider, ["/agent find regex", "/quit"])
    run(app, outputs)

    text = outputs.getvalue()
    assert "Found regex in auth.py." in text
    assert "[tool]" not in text


def test_agent_tool_activity_displayed():
    provider = AgentFakeProvider(
        responses=[tool_response("ping"), final_response("Done.")]
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent run check", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    text = outputs.getvalue()
    assert "[tool] ping" in text
    assert "Done." in text


def test_agent_multiple_tool_calls_displayed():
    provider = AgentFakeProvider(
        responses=[
            ProviderResponse(
                text="",
                tool_calls=[
                    ToolCall(id="c1", name="ping", arguments={}),
                    ToolCall(id="c2", name="ping", arguments={}),
                ],
                finish_reason="tool_calls",
            ),
            final_response("finished"),
        ]
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent two calls", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    text = outputs.getvalue()
    assert text.count("[tool] ping") == 2
    assert "finished" in text


def test_agent_tool_failure_marked():
    provider = AgentFakeProvider(
        responses=[tool_response("boom"), final_response("recovered")]
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent risky", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    text = outputs.getvalue()
    assert "[tool] boom (failed)" in text
    assert "recovered" in text


def test_agent_tool_arguments_displayed():
    provider = AgentFakeProvider(
        responses=[
            tool_response("ping", arguments={"path": "auth.py"}),
            final_response("Done."),
        ]
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent run check", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    text = outputs.getvalue()
    assert "[tool] ping" in text
    assert 'args: {"path": "auth.py"}' in text


def test_agent_tool_failure_reason_displayed():
    provider = AgentFakeProvider(
        responses=[tool_response("boom"), final_response("recovered")]
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent risky", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    text = outputs.getvalue()
    assert "[tool] boom (failed)" in text
    assert (
        "error: Tool execution failed: RuntimeError: the fake tool exploded" in text
    )


def test_agent_provider_failure_displayed():
    provider = AgentFakeProvider(error=APIRequestError("network down"))
    app, outputs = make_agent_app(new_session(), provider, ["/agent risky", "/quit"])
    code, text = run(app, outputs)

    assert code == 0
    assert "network down" in text
    assert "Agent stopped" in text


def test_agent_iteration_limit_displayed():
    provider = AgentFakeProvider(
        responses=[tool_response("ping")], repeat_last=True
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent loop forever", "/quit"],
        tool_registry=default_registry(),
        agent_max_iterations=3,
    )
    run(app, outputs)

    text = outputs.getvalue()
    assert "iteration limit reached after 3 iterations" in text
    assert text.count("[tool] ping") == 3


def test_agent_iteration_budget_respected():
    provider = AgentFakeProvider(
        responses=[tool_response("ping")], repeat_last=True
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent loop", "/quit"],
        tool_registry=default_registry(),
        agent_max_iterations=3,
    )
    run(app, outputs)

    assert len(provider.agent_calls) == 3


def test_agent_uses_active_provider_after_switch():
    provider = AgentFakeProvider(
        name="openai", responses=[final_response("default")]
    )
    ollama_provider = AgentFakeProvider(
        name="ollama", model="qwen3:14b", responses=[final_response("ollama did it")]
    )
    registry = {"ollama": lambda model=None: ollama_provider}
    app, outputs = make_agent_app(
        new_session(), provider, ["/switch ollama:qwen3:14b", "/agent tidy", "/quit"],
        provider_registry=registry,
    )
    run(app, outputs)

    assert app.provider is ollama_provider
    assert len(ollama_provider.agent_calls) == 1
    assert "ollama did it" in outputs.getvalue()
    assert provider.agent_calls == []


def test_agent_uses_active_model_after_switch():
    provider = AgentFakeProvider(responses=[final_response("default")])
    ollama_provider = AgentFakeProvider(
        name="ollama", model="qwen3:14b", responses=[final_response("ok")]
    )
    registry = {"ollama": lambda model=None: ollama_provider}
    app, outputs = make_agent_app(
        new_session(), provider, ["/switch ollama:qwen3:14b", "/agent tidy", "/quit"],
        provider_registry=registry,
    )
    run(app, outputs)

    assert ollama_provider.model == "qwen3:14b"
    assert len(ollama_provider.agent_calls) == 1


def test_normal_chat_unchanged_by_agent():
    provider = AgentFakeProvider(responses=[final_response("ignored")])
    app, outputs = make_agent_app(new_session(), provider, ["hello", "/quit"])
    run(app, outputs)

    assert provider.send_calls == 1
    assert provider.agent_calls == []
    assert "-- Agent --" not in outputs.getvalue()


def test_status_unchanged_after_agent():
    provider = AgentFakeProvider(responses=[final_response("ok")])
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent tidy", "/status", "/quit"]
    )
    run(app, outputs)

    text = outputs.getvalue()
    assert "Provider:  openai" in text
    assert "Model:     gpt-4o" in text
    assert "Fix the login bug" in text


def test_session_valid_after_agent():
    provider = AgentFakeProvider(responses=[final_response("Answer body.")])
    app, outputs = make_agent_app(new_session(), provider, ["/agent do it", "/quit"])
    run(app, outputs)

    loaded = load_session(str(app.session_path))
    assert loaded.state.goal == "Fix the login bug"
    assert loaded.active_provider == "openai"
    assert loaded.active_model == "gpt-4o"
    history = loaded.conversation_history
    assert history[0].role == "user"
    assert history[0].content == "do it"
    assert history[-1].role == "assistant"
    assert history[-1].content == "Answer body."


def test_agent_session_records_only_task_and_answer():
    session = new_session()
    provider = AgentFakeProvider(
        responses=[tool_response("ping"), final_response("Answer body.")]
    )
    app, outputs = make_agent_app(
        session, provider, ["/agent do it", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    history = session.conversation_history
    assert len(history) == 2
    assert history[0].role == "user" and history[0].content == "do it"
    assert history[1].role == "assistant" and history[1].content == "Answer body."


def test_agent_failure_still_records_session():
    provider = AgentFakeProvider(error=APIRequestError("down"))
    app, outputs = make_agent_app(new_session(), provider, ["/agent risky", "/quit"])
    run(app, outputs)

    history = app.session.conversation_history
    assert len(history) == 2
    assert history[0].role == "user" and history[0].content == "risky"
    assert history[1].role == "assistant"


def test_agent_empty_final_text_handled():
    provider = AgentFakeProvider(responses=[final_response("")])
    app, outputs = make_agent_app(new_session(), provider, ["/agent quiet", "/quit"])
    run(app, outputs)

    text = outputs.getvalue()
    assert "Agent finished with no text response." in text
    assert app.session.conversation_history[-1].content == "(agent output)"


def test_no_retry_on_provider_failure():
    provider = AgentFakeProvider(error=APIRequestError("down"))
    app, outputs = make_agent_app(new_session(), provider, ["/agent risky", "/quit"])
    run(app, outputs)

    assert len(provider.agent_calls) == 1


def test_no_automatic_provider_switch_on_failure():
    provider = AgentFakeProvider(error=APIRequestError("down"))
    app, outputs = make_agent_app(new_session(), provider, ["/agent risky", "/quit"])
    run(app, outputs)

    assert app.provider is provider
    assert app.session.active_provider == "openai"
    assert app.session.active_model == "gpt-4o"
    assert app.session.archived_conversations == []


def test_default_registry_has_exact_seven_tools():
    with tempfile.TemporaryDirectory() as d:
        registry = default_tool_registry(d)
    names = [tool.name for tool in registry.list_tools()]

    assert len(names) == 7
    assert len(set(names)) == 7
    assert set(names) == {
        "read_file",
        "list_files",
        "search_files",
        "write_file",
        "run_command",
        "git_diff",
        "run_tests",
    }


def test_agent_uses_provided_registry_not_duplicate():
    registry = default_registry()
    provider = AgentFakeProvider(responses=[tool_response("ping"), final_response("ok")])
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent quick", "/quit"], tool_registry=registry
    )
    run(app, outputs)

    assert app.tool_registry is registry


def test_cli_orchestration_imports_no_provider_sdks():
    main_py = (Path(__file__).resolve().parents[1] / "churro" / "main.py").read_text(encoding="utf-8")

    for forbidden in [
        "import openai",
        "from openai ",
        "import ollama",
        "from ollama ",
        "openai_provider",
        "ollama_provider",
    ]:
        assert forbidden not in main_py, f"CLI orchestration leaks: {forbidden}"


def test_cli_uses_agent_runner_not_a_duplicate():
    main_py = (Path(__file__).resolve().parents[1] / "churro" / "main.py").read_text(encoding="utf-8")

    assert "from churro.agent import" in main_py
    assert "AgentRunner(" in main_py
    assert "runner.run(" in main_py


def test_config_rejects_non_positive_max_iterations():
    for bad in [0, -1, "3", True]:
        try:
            make_agent_app(
                new_session(), AgentFakeProvider(), [], agent_max_iterations=bad
            )
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for max_iterations={bad!r}")


def test_max_iterations_env_toggle():
    saved = os.environ.get("CHURRO_AGENT_MAX_ITERATIONS")
    os.environ["CHURRO_AGENT_MAX_ITERATIONS"] = "2"
    try:
        provider = AgentFakeProvider(
            responses=[tool_response("ping")], repeat_last=True
        )
        app, outputs = make_agent_app(
            new_session(), provider, ["/agent loop", "/quit"],
            tool_registry=default_registry(),
        )
        assert app.agent_max_iterations == 2
        run(app, outputs)
        assert "iteration limit reached after 2 iterations" in outputs.getvalue()
        assert len(provider.agent_calls) == 2
    finally:
        if saved is not None:
            os.environ["CHURRO_AGENT_MAX_ITERATIONS"] = saved
        else:
            os.environ.pop("CHURRO_AGENT_MAX_ITERATIONS", None)


def test_max_iterations_env_invalid_falls_back_to_default():
    saved = os.environ.get("CHURRO_AGENT_MAX_ITERATIONS")
    os.environ["CHURRO_AGENT_MAX_ITERATIONS"] = "abc"
    try:
        app, _ = make_agent_app(new_session(), AgentFakeProvider(), [])
        assert app.agent_max_iterations == DEFAULT_MAX_ITERATIONS
    finally:
        if saved is not None:
            os.environ["CHURRO_AGENT_MAX_ITERATIONS"] = saved
        else:
            os.environ.pop("CHURRO_AGENT_MAX_ITERATIONS", None)


def test_agent_no_transcript_dumped_to_state():
    session = new_session()
    session.state.notes = "keep me"
    provider = AgentFakeProvider(
        responses=[tool_response("ping"), final_response("Answer body.")]
    )
    app, outputs = make_agent_app(
        session, provider, ["/agent do it", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    assert session.state.notes == "keep me"
    assert session.state.status == "in_progress"
    assert session.state.next_action == ""
    assert len(session.conversation_history) == 2


# ---- state-update integration tests (Step 18) ----


def test_agent_valid_state_update_updates_session_state():
    session = new_session()
    provider = AgentFakeProvider(
        responses=[
            final_with_state(
                "Found the bug.",
                {
                    "status": "done",
                    "current_task": "Fix unicode in the regex",
                    "completed_work": ["Found the line"],
                    "blockers": [],
                    "relevant_files": ["auth.py"],
                    "next_action": "Run the test suite",
                    "notes": "Regex rejects non-ASCII characters",
                },
            )
        ]
    )
    app, outputs = make_agent_app(session, provider, ["/agent fix the bug", "/quit"])
    run(app, outputs)

    assert session.state.status == "done"
    assert session.state.current_task == "Fix unicode in the regex"
    assert session.state.next_action == "Run the test suite"
    assert session.state.notes == "Regex rejects non-ASCII characters"
    assert session.state.relevant_files == ["auth.py"]


def test_agent_state_update_merges_completed_work():
    session = new_session()
    session.state.completed_work = ["inspected auth.py"]
    provider = AgentFakeProvider(
        responses=[
            final_with_state(
                "Done.", {"completed_work": ["inspected auth.py", "fixed the bug"]}
            )
        ]
    )
    app, outputs = make_agent_app(session, provider, ["/agent work", "/quit"])
    run(app, outputs)
    assert session.state.completed_work == ["inspected auth.py", "fixed the bug"]


def test_agent_state_update_replaces_blockers_and_relevant_files():
    session = new_session()
    session.state.blockers = ["old blocker"]
    provider = AgentFakeProvider(
        responses=[
            final_with_state(
                "Done.", {"blockers": ["new blocker"], "relevant_files": ["main.py"]}
            )
        ]
    )
    app, outputs = make_agent_app(session, provider, ["/agent work", "/quit"])
    run(app, outputs)
    assert session.state.blockers == ["new blocker"]
    assert session.state.relevant_files == ["main.py"]


def test_agent_state_update_cannot_overwrite_goal():
    session = new_session()
    provider = AgentFakeProvider(
        responses=[final_with_state("Done.", {"goal": "replaced", "status": "done"})]
    )
    app, outputs = make_agent_app(session, provider, ["/agent work", "/quit"])
    run(app, outputs)
    assert session.state.goal == "Fix the login bug"
    assert session.state.status == "done"
    assert "protected" in outputs.getvalue()


def test_agent_without_state_update_still_works_and_state_unchanged():
    session = new_session()
    provider = AgentFakeProvider(responses=[final_response("All set.")])
    app, outputs = make_agent_app(session, provider, ["/agent tidy", "/quit"])
    run(app, outputs)
    assert "All set." in outputs.getvalue()
    assert session.state.status == "in_progress"
    assert session.state.next_action == ""
    assert session.state.goal == "Fix the login bug"


def test_agent_invalid_state_update_does_not_corrupt_state():
    session = new_session()
    provider = AgentFakeProvider(
        responses=[
            final_response("Working.\n<state_update>\n{\"status\": oops\n</state_update>")
        ]
    )
    app, outputs = make_agent_app(session, provider, ["/agent risky", "/quit"])
    run(app, outputs)

    assert session.state.status == "in_progress"
    assert session.state.next_action == ""
    assert "Working." in outputs.getvalue()


def test_agent_state_update_block_not_leaked_into_answer():
    provider = AgentFakeProvider(
        responses=[final_with_state("Trace complete.", {"status": "done"})]
    )
    app, outputs = make_agent_app(new_session(), provider, ["/agent trace", "/quit"])
    run(app, outputs)

    text = outputs.getvalue()
    assert "Trace complete." in text
    assert "<state_update>" not in text
    assert "</state_update>" not in text


def test_agent_tool_transcript_not_dumped_to_state():
    session = new_session()
    session.state.notes = "keep me"
    provider = AgentFakeProvider(
        responses=[
            tool_response("ping"),
            final_with_state("Done with it.", {"next_action": "check results"}),
        ]
    )
    app, outputs = make_agent_app(
        session, provider, ["/agent do it", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    assert session.state.next_action == "check results"
    assert session.state.notes == "keep me"
    assert "pong" not in str(session.state.model_dump())
    assert len(session.conversation_history) == 2


# ---- interrupted-run AgentFrame capture (Step 2) ----


def test_successful_agent_leaves_pending_agent_none():
    provider = AgentFakeProvider(responses=[final_response("All set.")])
    app, outputs = make_agent_app(new_session(), provider, ["/agent tidy", "/quit"])
    run(app, outputs)

    assert app.session.pending_agent is None
    assert "All set." in outputs.getvalue()


def test_provider_failure_creates_pending_agent():
    provider = AgentFakeProvider(error=APIRequestError("down"))
    app, outputs = make_agent_app(new_session(), provider, ["/agent risky", "/quit"])
    run(app, outputs)

    frame = app.session.pending_agent
    assert frame is not None
    assert frame.task == "risky"
    assert frame.stop_reason == "provider_error"
    assert frame.iterations_used == 1
    assert frame.max_iterations == DEFAULT_MAX_ITERATIONS
    assert frame.tool_digest == []


def test_provider_failure_records_task_iterations_and_digest():
    provider = SequenceProvider(
        [tool_response("ping"), APIRequestError("network down")]
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent fix it", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    frame = app.session.pending_agent
    assert frame is not None
    assert frame.task == "fix it"
    assert frame.stop_reason == "provider_error"
    assert frame.iterations_used == 2
    assert frame.max_iterations == DEFAULT_MAX_ITERATIONS
    assert frame.tool_digest == ["ping(c1) -> success"]


def test_iteration_limit_creates_pending_agent():
    provider = AgentFakeProvider(responses=[tool_response("ping")], repeat_last=True)
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent loop", "/quit"],
        tool_registry=default_registry(),
        agent_max_iterations=3,
    )
    run(app, outputs)

    frame = app.session.pending_agent
    assert frame is not None
    assert frame.stop_reason == "iteration_limit"
    assert frame.iterations_used == 3
    assert frame.max_iterations == 3
    assert frame.tool_digest == ["ping(c1) -> success"] * 3


def test_ctrl_c_preserves_partial_agent_frame():
    provider = SequenceProvider([tool_response("ping"), KeyboardInterrupt()])
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent do it", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    frame = app.session.pending_agent
    assert frame is not None
    assert frame.stop_reason == "interrupted"
    assert frame.iterations_used == 2
    assert frame.tool_digest == ["ping(c1) -> success"]
    assert "interrupted by Ctrl+C" in outputs.getvalue()
    assert frame.completed_at is not None


def test_tool_digest_contains_status_not_outputs():
    provider = SequenceProvider(
        [tool_response("boom"), APIRequestError("down")]
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent risky", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    frame = app.session.pending_agent
    assert frame.tool_digest == ["boom(c1) -> failure"]
    rendered = "\n".join(frame.tool_digest)
    assert "exploded" not in rendered
    assert "pong" not in rendered
    assert "output" not in frame.model_dump()


def test_pending_agent_survives_save_and_load():
    provider = SequenceProvider(
        [tool_response("ping"), APIRequestError("down")]
    )
    app, outputs = make_agent_app(
        new_session(), provider, ["/agent fix it", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    loaded = load_session(str(app.session_path))
    assert loaded.pending_agent is not None
    assert loaded.pending_agent == app.session.pending_agent
    assert loaded.pending_agent.stop_reason == "provider_error"
    assert loaded.pending_agent.task == "fix it"


def test_switch_preserves_pending_agent():
    ollama_provider = AgentFakeProvider(
        name="ollama", model="qwen3:14b", responses=[final_response("ok")]
    )
    registry = {"ollama": lambda model=None: ollama_provider}
    provider = SequenceProvider(
        [tool_response("ping"), APIRequestError("down")]
    )
    app, outputs = make_agent_app(
        new_session(), provider,
        ["/agent fix it", "/switch ollama:qwen3:14b", "/quit"],
        provider_registry=registry,
        tool_registry=default_registry(),
    )
    run(app, outputs)

    assert app.provider is ollama_provider
    assert app.session.active_model == "qwen3:14b"
    assert len(app.session.archived_conversations) == 1
    assert app.session.pending_agent is not None
    assert app.session.pending_agent.stop_reason == "provider_error"


TEST_FUNCTIONS = [
    test_agent_command_recognized_and_answered,
    test_agent_empty_task_shows_usage,
    test_agent_whitespace_task_shows_usage,
    test_agent_task_reaches_runner,
    test_agent_receives_system_prompt_and_state,
    test_agent_final_response_displayed,
    test_agent_tool_activity_displayed,
    test_agent_multiple_tool_calls_displayed,
    test_agent_tool_failure_marked,
    test_agent_tool_arguments_displayed,
    test_agent_tool_failure_reason_displayed,
    test_agent_provider_failure_displayed,
    test_agent_iteration_limit_displayed,
    test_agent_iteration_budget_respected,
    test_agent_uses_active_provider_after_switch,
    test_agent_uses_active_model_after_switch,
    test_normal_chat_unchanged_by_agent,
    test_status_unchanged_after_agent,
    test_session_valid_after_agent,
    test_agent_session_records_only_task_and_answer,
    test_agent_failure_still_records_session,
    test_agent_empty_final_text_handled,
    test_no_retry_on_provider_failure,
    test_no_automatic_provider_switch_on_failure,
    test_default_registry_has_exact_seven_tools,
    test_agent_uses_provided_registry_not_duplicate,
    test_cli_orchestration_imports_no_provider_sdks,
    test_cli_uses_agent_runner_not_a_duplicate,
    test_config_rejects_non_positive_max_iterations,
    test_max_iterations_env_toggle,
    test_max_iterations_env_invalid_falls_back_to_default,
    test_agent_no_transcript_dumped_to_state,
    test_agent_valid_state_update_updates_session_state,
    test_agent_state_update_merges_completed_work,
    test_agent_state_update_replaces_blockers_and_relevant_files,
    test_agent_state_update_cannot_overwrite_goal,
    test_agent_without_state_update_still_works_and_state_unchanged,
    test_agent_invalid_state_update_does_not_corrupt_state,
    test_agent_state_update_block_not_leaked_into_answer,
    test_agent_tool_transcript_not_dumped_to_state,
    test_successful_agent_leaves_pending_agent_none,
    test_provider_failure_creates_pending_agent,
    test_provider_failure_records_task_iterations_and_digest,
    test_iteration_limit_creates_pending_agent,
    test_ctrl_c_preserves_partial_agent_frame,
    test_tool_digest_contains_status_not_outputs,
    test_pending_agent_survives_save_and_load,
    test_switch_preserves_pending_agent,
]


def main() -> None:
    failures = 0
    for fn in TEST_FUNCTIONS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")
    if failures:
        raise SystemExit(f"\n{failures} test(s) failed")
    print(f"\nAll {len(TEST_FUNCTIONS)} tests passed.")


if __name__ == "__main__":
    main()