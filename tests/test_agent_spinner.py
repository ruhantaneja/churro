"""Offline tests for the /agent CLI activity indicator.

These verify the *lifecycle/state* of the Rich status spinner -- that it is
active while the provider call runs and fully torn down afterwards -- rather
than timing-sensitive animation frames.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from churro.agent import AgentResult, AgentRunner, DEFAULT_MAX_ITERATIONS
from churro.core.session import create_session, load_session
from churro.main import AGENT_STATUS_MESSAGE, CHURROApp
from churro.providers.provider import APIRequestError, Provider
from churro.providers.responses import ProviderResponse, ToolCall
from churro.tools import Tool, ToolArgs, ToolRegistry, ToolResult


class PingArgs(ToolArgs):
    pass


class PingTool(Tool[PingArgs]):
    name = "ping"
    description = "Succeed and reply 'pong' (fake tool)."
    args_model = PingArgs

    def _run(self, args: PingArgs) -> ToolResult:
        return ToolResult(success=True, output="pong")


class AgentFakeProvider(Provider):
    name = "openai"

    def __init__(self, model="gpt-4o", responses=(), name=None, error=None, repeat_last=False):
        self.name = name or self.name
        self.model = model
        self.responses = list(responses)
        self.error = error
        self.repeat_last = repeat_last
        self.agent_calls = []
        self.send_calls = 0

    def send(self, messages):
        self.send_calls += 1
        return "No canned reply."

    def send_normalized(self, messages, tools=None, tool_results=None):
        self.agent_calls.append(messages)
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


class ObservingProvider(AgentFakeProvider):
    """Records the console live-display state inside each provider call."""

    def __init__(self, console=None, **kwargs):
        super().__init__(**kwargs)
        self.console = console
        self.live_stack_sizes = []
        self.lives = []

    def send_normalized(self, messages, tools=None, tool_results=None):
        if self.console is not None:
            stack = getattr(self.console, "_live_stack", [])
            self.live_stack_sizes.append(len(stack))
            self.lives.append(stack[0] if stack else None)
        return super().send_normalized(messages, tools=tools, tool_results=tool_results)


def tool_response(name="ping", call_id="c1"):
    return ProviderResponse(
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments={})],
        finish_reason="tool_calls",
    )


def final_response(text):
    return ProviderResponse(text=text, finish_reason="stop")


def default_registry():
    registry = ToolRegistry()
    registry.register(PingTool())
    return registry


def new_session():
    session = create_session(goal="Fix the login bug", provider="openai", model="gpt-4o")
    session.state.status = "in_progress"
    session.state.relevant_files = ["auth.py"]
    return session


def make_app(
    session,
    provider,
    lines,
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
        tool_registry=tool_registry,
        agent_max_iterations=agent_max_iterations,
        input_fn=lambda prompt="": queued.pop(0),
        console=console,
    )
    if isinstance(provider, ObservingProvider):
        provider.console = app.console
    return app, outputs


def run(app, outputs):
    code = app.run()
    return code, outputs.getvalue()


def live_stack_size(app):
    return len(getattr(app.console, "_live_stack", []))


def test_status_active_during_provider_call():
    provider = ObservingProvider(responses=[final_response("All set.")])
    app, outputs = make_app(new_session(), provider, ["/agent tidy up", "/quit"])
    code, text = run(app, outputs)

    assert code == 0
    assert provider.live_stack_sizes == [1]
    assert live_stack_size(app) == 0
    assert "All set." in text


def test_status_removed_after_successful_response():
    provider = ObservingProvider(responses=[final_response("All set.")])
    app, _ = make_app(new_session(), provider, ["/agent tidy up", "/quit"])
    run(app, _)

    assert live_stack_size(app) == 0
    assert provider.lives[0]._refresh_thread is None


def test_status_removed_after_provider_error():
    provider = ObservingProvider(error=APIRequestError("model unreachable"))
    app, outputs = make_app(new_session(), provider, ["/agent risky", "/quit"])
    code, text = run(app, outputs)

    assert code == 0
    assert provider.live_stack_sizes == [1]
    assert live_stack_size(app) == 0
    assert provider.lives[0]._refresh_thread is None
    assert "model unreachable" in text
    assert "Agent stopped" in text


def test_status_removed_after_iteration_limit():
    provider = ObservingProvider(
        responses=[tool_response()], repeat_last=True
    )
    app, outputs = make_app(
        new_session(), provider, ["/agent loop", "/quit"],
        tool_registry=default_registry(),
        agent_max_iterations=3,
    )
    code, text = run(app, outputs)

    assert code == 0
    assert provider.live_stack_sizes == [1, 1, 1]
    assert live_stack_size(app) == 0
    assert provider.lives[-1]._refresh_thread is None
    assert "iteration limit reached after 3 iterations" in text


def test_tool_messages_still_appear():
    provider = ObservingProvider(
        responses=[tool_response(), final_response("Done.")]
    )
    app, outputs = make_app(
        new_session(), provider, ["/agent check", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    assert "[tool] ping" in outputs.getvalue()
    assert "Done." in outputs.getvalue()


def test_final_response_still_appears():
    provider = ObservingProvider(responses=[final_response("The bug is in auth.py.")])
    app, outputs = make_app(new_session(), provider, ["/agent inspect", "/quit"])
    run(app, outputs)

    assert "The bug is in auth.py." in outputs.getvalue()


def test_multiple_iterations_keep_status_active():
    provider = ObservingProvider(
        responses=[
            tool_response(call_id="c1"),
            tool_response(call_id="c2"),
            final_response("finished"),
        ]
    )
    app, outputs = make_app(
        new_session(), provider, ["/agent work", "/quit"],
        tool_registry=default_registry(),
    )
    run(app, outputs)

    assert provider.live_stack_sizes == [1, 1, 1]
    assert live_stack_size(app) == 0
    assert outputs.getvalue().count("[tool] ping") == 2


def test_ctrl_c_during_agent_cleans_up_spinner():
    provider = ObservingProvider(error=KeyboardInterrupt())
    app, outputs = make_app(new_session(), provider, ["/agent risky", "/quit"])
    code, text = run(app, outputs)

    assert code == 0
    assert "> Agent interrupted by Ctrl+C." in text
    assert live_stack_size(app) == 0
    assert provider.lives[0]._refresh_thread is None


def test_ctrl_c_does_not_corrupt_session():
    provider = ObservingProvider(error=KeyboardInterrupt())
    app, outputs = make_app(new_session(), provider, ["/agent risky", "/quit"])
    run(app, outputs)

    loaded = load_session(str(app.session_path))
    assert loaded.state.goal == "Fix the login bug"
    assert loaded.state.status == "in_progress"
    assert loaded.active_provider == "openai"
    assert loaded.active_model == "gpt-4o"
    assert [m.content for m in loaded.conversation_history] == ["risky"]


def test_no_background_spinner_after_completion():
    provider = ObservingProvider(responses=[final_response("done")])
    app, outputs = make_app(new_session(), provider, ["/agent one", "/quit"])
    run(app, outputs)

    assert live_stack_size(app) == 0
    assert all(live._refresh_thread is None for live in provider.lives)


def test_cli_stays_usable_after_agent_run():
    provider = ObservingProvider(responses=[final_response("first"), final_response("second")])
    app, outputs = make_app(
        new_session(), provider, ["/agent one", "hello", "/agent two", "/quit"]
    )
    code, text = run(app, outputs)

    assert code == 0
    assert live_stack_size(app) == 0
    assert len(provider.agent_calls) == 2
    assert "first" in text
    assert "second" in text


def test_normal_chat_intact_no_spinner():
    provider = ObservingProvider()
    app, outputs = make_app(new_session(), provider, ["hello", "/status", "/quit"])
    run(app, outputs)

    assert provider.send_calls == 1
    assert provider.agent_calls == []
    assert live_stack_size(app) == 0
    assert provider.live_stack_sizes == []


def test_status_message_is_application_level():
    assert AGENT_STATUS_MESSAGE == "CHURRO is thinking..."


def test_agent_runner_unchanged_by_spinner():
    runner_src = (
        Path(__file__).resolve().parents[1] / "churro" / "agent" / "runner.py"
    ).read_text(encoding="utf-8")

    for leaked in ["thinking", "status(", "console", "_live"]:
        assert leaked not in runner_src, f"spinner leaked into AgentRunner: {leaked!r}"

    assert "class AgentRunner:" in runner_src
    assert "def __init__(" in runner_src
    assert "max_iterations:" in runner_src
    assert "dishonest" not in runner_src


def test_agent_runner_api_still_used_by_cli():
    runner = AgentRunner(AgentFakeProvider(), ToolRegistry(), max_iterations=2)
    assert runner.max_iterations == 2

    with patch("churro.main.AgentRunner", wraps=AgentRunner) as wrapped:
        provider = AgentFakeProvider(responses=[final_response("ok")])
        app, outputs = make_app(new_session(), provider, ["/agent task", "/quit"])
        run(app, outputs)
        assert wrapped.called


TEST_FUNCTIONS = [
    test_status_active_during_provider_call,
    test_status_removed_after_successful_response,
    test_status_removed_after_provider_error,
    test_status_removed_after_iteration_limit,
    test_tool_messages_still_appear,
    test_final_response_still_appears,
    test_multiple_iterations_keep_status_active,
    test_ctrl_c_during_agent_cleans_up_spinner,
    test_ctrl_c_does_not_corrupt_session,
    test_no_background_spinner_after_completion,
    test_cli_stays_usable_after_agent_run,
    test_normal_chat_intact_no_spinner,
    test_status_message_is_application_level,
    test_agent_runner_unchanged_by_spinner,
    test_agent_runner_api_still_used_by_cli,
]


def main() -> None:
    failures = 0
    for fn in TEST_FUNCTIONS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit(f"\n{failures} test(s) failed")
    print(f"\nAll {len(TEST_FUNCTIONS)} tests passed.")


if __name__ == "__main__":
    main()