import io
import tempfile
from pathlib import Path

from rich.console import Console

from churro.agent.runner import AgentRunner
from churro.core.session import ConversationMessage, create_session
from churro.main import CHURROApp
from churro.prompts import load_system_prompt
from churro.providers.provider import Provider
from churro.providers.responses import ProviderResponse
from churro.tools import (
    Tool,
    ToolArgs,
    ToolRegistry,
    ToolResult,
)

DISCIPLINE_MARKERS = [
    "autonomous coding agent",
    "Stay focused on the exact task",
    "Do not invent a different task",
    "reassess it against the original user task",
    "Do not modify files unless",
    "stop using tools and provide a concise final answer",
]

FORBIDDEN_PROVIDER_TERMS = [
    "ollama",
    "openai",
    "qwen",
    "llama",
    "gpt-",
    "claude",
    "gemini",
    "chain-of-thought",
    "show your reasoning",
    "think step by step",
]


class EchoArgs(ToolArgs):
    pass


class EchoTool(Tool[EchoArgs]):
    name = "echo"
    description = "Echo back a marker (fake tool)."
    args_model = EchoArgs

    def _run(self, args: EchoArgs) -> ToolResult:
        return ToolResult(success=True, output="echoed")


class PromptFakeProvider(Provider):
    name = "openai"

    def __init__(self, model="gpt-4o"):
        self.model = model
        self.sent_messages = []
        self.agent_calls = []
        self.send_calls = 0

    def send(self, messages):
        self.send_calls += 1
        self.sent_messages.append(messages)
        return "chat reply."

    def send_normalized(self, messages, tools=None, tool_results=None):
        self.agent_calls.append(
            {"messages": messages, "tools": tools, "tool_results": tool_results}
        )
        return ProviderResponse(text="completed.", finish_reason="stop")


def new_session():
    session = create_session(goal="Example goal", provider="openai", model="gpt-4o")
    return session


def make_app(session, provider, lines, tool_registry=None):
    outputs = io.StringIO()
    console = Console(file=outputs, force_terminal=False, width=120, markup=False)
    queued = list(lines)
    app = CHURROApp(
        session=session,
        provider=provider,
        system_prompt="SYSTEM_RULES",
        session_path=str(Path(tempfile.mkdtemp()) / f"{session.session_id}.json"),
        tool_registry=tool_registry,
        input_fn=lambda prompt="": queued.pop(0),
        console=console,
    )
    return app, outputs


def run(app):
    app.run()


def echo_registry():
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


def test_system_prompt_contains_task_discipline_guidance():
    prompt = " ".join(load_system_prompt().split())
    for marker in DISCIPLINE_MARKERS:
        assert marker in prompt, f"missing discipline marker: {marker!r}"


def test_system_prompt_has_no_provider_specific_instructions():
    prompt = load_system_prompt().lower()
    for term in FORBIDDEN_PROVIDER_TERMS:
        assert term not in prompt, f"provider-specific term leaked: {term!r}"


def test_agent_original_user_task_stays_present_as_last_message():
    provider = PromptFakeProvider()
    app, _ = make_app(new_session(), provider, ["/agent run the tests", "/quit"])
    run(app)

    sent = provider.agent_calls[0]["messages"]
    assert len(sent) >= 2
    assert sent[0]["role"] == "system"
    assert sent[-1]["role"] == "user"
    assert sent[-1]["content"] == "run the tests"


def test_unrelated_history_does_not_displace_the_task():
    session = new_session()
    session.conversation_history.append(
        ConversationMessage(role="user", content="unrelated old question")
    )
    session.conversation_history.append(
        ConversationMessage(role="assistant", content="unrelated old answer")
    )
    provider = PromptFakeProvider()
    app, _ = make_app(session, provider, ["/agent run the tests", "/quit"])
    run(app)

    sent = provider.agent_calls[0]["messages"]
    roles = [m["role"] for m in sent]
    assert roles == ["system", "user", "assistant", "user"]
    assert sent[0]["content"].startswith("SYSTEM_RULES")
    assert sent[-1]["content"] == "run the tests"
    assert "unrelated old question" in sent[1]["content"]
    assert "unrelated old answer" in sent[2]["content"]


def test_tool_definitions_remain_available_to_agent():
    provider = PromptFakeProvider()
    app, _ = make_app(
        new_session(), provider, ["/agent run the tests", "/quit"],
        tool_registry=echo_registry(),
    )
    run(app)

    tools = provider.agent_calls[0]["tools"]
    assert tools is not None
    assert [t.name for t in tools] == ["echo"]


def test_normal_chat_behavior_unchanged():
    provider = PromptFakeProvider()
    app, _ = make_app(new_session(), provider, ["hello", "/quit"])
    run(app)

    assert provider.send_calls == 1
    assert provider.agent_calls == []
    assert len(provider.sent_messages[0]) == 2
    assert provider.sent_messages[0][0]["role"] == "system"
    assert provider.sent_messages[0][1] == {"role": "user", "content": "hello"}


def test_agent_runner_unchanged_by_prompt_fix():
    runner_src = (Path(__file__).resolve().parents[1] / "churro" / "agent" / "runner.py").read_text(encoding="utf-8")

    assert "churro.prompts" not in runner_src
    assert "from churro.prompts" not in runner_src
    assert "Stay focused on the exact task" not in runner_src
    assert "AgentRunner" in runner_src
    assert isinstance(AgentRunner.run, object)


def test_run_tests_description_identifies_primary_specialized_tool():
    from churro.tools.run_tests import RunTestsTool

    desc = RunTestsTool.description.lower()
    assert "primary" in desc
    assert "specialized" in desc
    assert "run_tests" in desc
    assert "discovery" in desc


def test_run_tests_description_discourages_manual_test_commands():
    from churro.tools.run_tests import RunTestsTool

    desc = RunTestsTool.description.lower()
    assert "run_command" in desc
    assert "pytest/python test commands" in desc
    assert "back to the agent" in desc


def test_system_prompt_prefers_specialized_tools_over_run_command():
    prompt = " ".join(load_system_prompt().split()).lower()
    assert "prefer specialized churro tools over run_command" in prompt
    assert "run_tests" in prompt
    assert "use run_command only when no specialized tool" in prompt


def test_system_prompt_directs_agents_to_run_tests_for_test_execution():
    prompt = " ".join(load_system_prompt().split()).lower()
    assert "run_tests is the specialized tool" in prompt
    assert "instead of run_command or manually constructed pytest/python" in prompt


def test_system_prompt_discourages_unix_shell_assumptions():
    prompt = " ".join(load_system_prompt().split()).lower()
    assert "do not assume unix commands" in prompt
    for term in ("find", "head", "python3"):
        assert term in prompt
    for tool in ("list_files", "search_files", "read_file"):
        assert tool in prompt


TEST_FUNCTIONS = [
    test_system_prompt_contains_task_discipline_guidance,
    test_system_prompt_has_no_provider_specific_instructions,
    test_run_tests_description_identifies_primary_specialized_tool,
    test_run_tests_description_discourages_manual_test_commands,
    test_system_prompt_prefers_specialized_tools_over_run_command,
    test_system_prompt_directs_agents_to_run_tests_for_test_execution,
    test_system_prompt_discourages_unix_shell_assumptions,
    test_agent_original_user_task_stays_present_as_last_message,
    test_unrelated_history_does_not_displace_the_task,
    test_tool_definitions_remain_available_to_agent,
    test_normal_chat_behavior_unchanged,
    test_agent_runner_unchanged_by_prompt_fix,
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