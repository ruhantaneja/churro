import contextlib
import io
import os
import tempfile
from pathlib import Path

from rich.console import Console

from churro.core.session import ConversationMessage, create_session, load_session
from churro.main import (
    CHURROApp,
    PROJECT_ROOT,
    SESSIONS_DIR,
    _start_new_session,
    _workspace_root_from_env,
    main as run_churro_main,
)
from churro.tools import default_tool_registry
from churro.providers.provider import APIRequestError, Provider


class FakeProvider(Provider):
    name = "openai"

    def __init__(self, model="gpt-4o", responses=(), name=None):
        self.name = name or self.name
        self.model = model
        self.responses = list(responses)
        self.sent_messages = []

    def send(self, messages):
        self.sent_messages.append(messages)
        return self.responses.pop(0) if self.responses else "No canned reply."


class BoomProvider(FakeProvider):
    def send(self, messages):
        self.sent_messages.append(messages)
        raise APIRequestError("network down")


def new_session():
    session = create_session(goal="Fix the login bug", provider="openai", model="gpt-4o")
    session.state.status = "in_progress"
    session.state.relevant_files = ["auth.py"]
    return session


def make_app(session, provider, lines, registry=None, session_path=None):
    outputs = io.StringIO()
    console = Console(file=outputs, force_terminal=False, width=120, markup=False)
    queued = list(lines)
    app = CHURROApp(
        session=session,
        provider=provider,
        system_prompt="SYSTEM_RULES",
        session_path=session_path
        or str(Path(tempfile.mkdtemp()) / f"{session.session_id}.json"),
        provider_registry=registry,
        input_fn=lambda prompt="": queued.pop(0),
        console=console,
    )
    return app, outputs


def run(app, outputs):
    code = app.run()
    return code, outputs.getvalue()


def test_session_starts_correctly():
    app, _ = make_app(new_session(), FakeProvider(responses=["ok"]), ["/quit"])
    code, _ = run(app, _)
    assert code == 0
    assert app.session.state.goal == "Fix the login bug"
    assert app.session.active_provider == "openai"
    assert app.session.active_model == "gpt-4o"


def test_start_new_session_requires_goal():
    outputs = io.StringIO()
    console = Console(file=outputs, force_terminal=False, width=120, markup=False)
    queued = ["", "Fix the unicode bug"]
    session, path = _start_new_session(
        console, lambda prompt="": queued.pop(0), "gpt-4o", Path(tempfile.mkdtemp())
    )
    assert session.state.goal == "Fix the unicode bug"
    assert path.exists()


def test_user_message_reaches_provider():
    provider = FakeProvider(responses=["Got it."])
    app, outputs = make_app(new_session(), provider, ["hello", "/quit"])
    run(app, outputs)

    sent = provider.sent_messages[0]
    assert sent[0]["role"] == "system"
    assert "SYSTEM_RULES" in sent[0]["content"]
    assert "CURRENT PROJECT STATE" in sent[0]["content"]
    assert sent[-1] == {"role": "user", "content": "hello"}


def test_ai_response_reaches_session_manager_state_persisted_clean_shown():
    raw = (
        "I found the regex.\n<state_update>\n"
        '{"next_action": "Patch it", "status": "in_progress"}\n'
        "</state_update>"
    )
    provider = FakeProvider(responses=[raw])
    app, outputs = make_app(new_session(), provider, ["go", "/quit"])
    run(app, outputs)

    assert app.session.state.next_action == "Patch it"

    text = outputs.getvalue()
    assert "I found the regex." in text
    assert "<state_update>" not in text

    loaded = load_session(str(app.session_path))
    assert loaded.state.next_action == "Patch it"


def test_status_command():
    app, outputs = make_app(new_session(), FakeProvider(), ["/status", "/quit"])
    run(app, outputs)
    text = outputs.getvalue()
    for expected in [
        "Session:", "Provider:", "Goal:", "Fix the login bug",
        "Status:", "in_progress", "Files:", "auth.py", "Next:",
    ]:
        assert expected in text


def test_save_command():
    app, outputs = make_app(new_session(), FakeProvider(), ["/save", "/quit"])
    run(app, outputs)
    assert Path(app.session_path).exists()
    assert "Saved to" in outputs.getvalue()


def test_handoff_command():
    app, outputs = make_app(new_session(), FakeProvider(), ["/handoff", "/quit"])
    run(app, outputs)
    text = outputs.getvalue()
    assert "PROJECT STATE" in text
    assert "Handoff size:" in text and "tokens" in text and "chars" in text


def test_files_command():
    app, outputs = make_app(new_session(), FakeProvider(), ["/files", "/quit"])
    run(app, outputs)
    assert "auth.py" in outputs.getvalue()


def test_quit_saves_and_exits():
    app, outputs = make_app(new_session(), FakeProvider(), ["/quit"])
    code, _ = run(app, outputs)
    assert code == 0
    assert Path(app.session_path).exists()
    assert "Goodbye" in outputs.getvalue()


def test_invalid_switch():
    session = new_session()
    app, outputs = make_app(session, FakeProvider(), ["/switch anthropic", "/quit"])
    run(app, outputs)

    text = outputs.getvalue()
    assert "not available" in text
    assert "openai" in text
    assert session.active_provider == "openai"
    assert session.active_model == "gpt-4o"


def test_successful_model_switch():
    session = new_session()
    session.conversation_history.append(
        ConversationMessage(role="user", content="old user message")
    )
    registry = {"openai": lambda model=None: FakeProvider(model=model or "gpt-4o")}
    provider = FakeProvider(model="gpt-4o", responses=["dummy"])
    app, outputs = make_app(
        session, provider, ["/switch openai:gpt-4o-mini", "/quit"], registry=registry
    )
    run(app, outputs)

    assert session.active_provider == "openai"
    assert session.active_model == "gpt-4o-mini"
    assert app.provider.model == "gpt-4o-mini"
    assert len(session.archived_conversations) == 1
    assert session.archived_conversations[0].messages[0].content == "old user message"
    assert session.conversation_history == []
    assert "Switched openai/gpt-4o -> openai/gpt-4o-mini" in outputs.getvalue()


def test_new_provider_starts_from_handoff_not_old_history():
    session = new_session()
    session.state.completed_work = ["Found auth.py"]
    session.conversation_history.append(
        ConversationMessage(role="user", content="SECRET_HISTORY")
    )
    registry = {
        "openai": lambda model=None: FakeProvider(model=model or "gpt-4o", responses=["continue"])
    }
    provider = FakeProvider(model="gpt-4o", responses=["continue"])
    app, outputs = make_app(
        session, provider, ["/switch openai:gpt-4o-mini", "continue", "/quit"],
        registry=registry,
    )
    run(app, outputs)

    sent = app.provider.sent_messages[0]
    assert sent[0]["role"] == "system"
    assert "You are taking over an existing coding task in CHURRO." in sent[0]["content"]
    assert "Found auth.py" in sent[0]["content"]
    joined = "\n".join(m["content"] for m in sent)
    assert "SECRET_HISTORY" not in joined
    assert "SECRET_HISTORY" not in outputs.getvalue()


def test_handoff_command_excludes_conversation_history():
    session = new_session()
    session.conversation_history.append(
        ConversationMessage(role="user", content="SECRET_HISTORY")
    )
    app, outputs = make_app(session, FakeProvider(), ["/handoff", "/quit"])
    run(app, outputs)
    assert "SECRET_HISTORY" not in outputs.getvalue()


def test_malformed_ai_response_no_crash():
    raw = "Hmm. <state_update>\n{\"status\": broken\n</state_update>"
    session = new_session()
    provider = FakeProvider(responses=[raw])
    app, outputs = make_app(session, provider, ["work", "/status", "/quit"])
    code, text = run(app, outputs)

    assert code == 0
    assert "Hmm." in text
    assert session.state.status == "in_progress"
    assert session.state.next_action == ""


def test_provider_error_does_not_crash():
    session = new_session()
    app, outputs = make_app(session, BoomProvider(), ["work", "/quit"])
    code, text = run(app, outputs)
    assert code == 0
    assert "network down" in text
    assert session.state.status == "in_progress"


def test_empty_input_ignored():
    provider = FakeProvider(responses=["ok"])
    app, outputs = make_app(new_session(), provider, ["", "   ", "hello", "/quit"])
    code, _ = run(app, outputs)
    assert code == 0
    assert len(provider.sent_messages) == 1


def test_unknown_command():
    app, outputs = make_app(new_session(), FakeProvider(), ["/bogus", "/quit"])
    run(app, outputs)
    assert "Unknown command" in outputs.getvalue()


def test_main_graceful_without_api_key():
    saved = os.environ.get("OPENAI_API_KEY")
    saved_provider = os.environ.get("CHURRO_PROVIDER")
    saved_model = os.environ.get("CHURRO_MODEL")
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("CHURRO_PROVIDER", None)
    os.environ.pop("CHURRO_MODEL", None)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run_churro_main([])
        assert code == 1
        assert "OPENAI_API_KEY" in buf.getvalue()
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved
        if saved_provider is not None:
            os.environ["CHURRO_PROVIDER"] = saved_provider
        else:
            os.environ.pop("CHURRO_PROVIDER", None)
        if saved_model is not None:
            os.environ["CHURRO_MODEL"] = saved_model
        else:
            os.environ.pop("CHURRO_MODEL", None)


def test_switch_ollama_through_default_factory():
    session = new_session()
    app, outputs = make_app(
        session, FakeProvider(), ["/switch ollama:qwen3:14b", "/status", "/quit"]
    )
    run(app, outputs)

    assert session.active_provider == "ollama"
    assert session.active_model == "qwen3:14b"
    assert app.provider.name == "ollama"
    assert app.provider.model == "qwen3:14b"

    text = outputs.getvalue()
    assert "Switched openai/gpt-4o -> ollama/qwen3:14b" in text
    assert "Provider:  ollama" in text
    assert "Model:     qwen3:14b" in text


def test_switch_ollama_with_registered_builder():
    session = new_session()
    session.conversation_history.append(
        ConversationMessage(role="user", content="old user message")
    )
    registry = {
        "ollama": lambda model=None: FakeProvider(name="ollama", model=model or "qwen3:14b")
    }
    app, outputs = make_app(
        session, FakeProvider(), ["/switch ollama:qwen3:14b", "/quit"], registry=registry
    )
    run(app, outputs)

    assert session.active_provider == "ollama"
    assert session.active_model == "qwen3:14b"
    assert app.provider.model == "qwen3:14b"
    assert session.state.goal == "Fix the login bug"
    assert len(session.archived_conversations) == 1
    assert session.conversation_history == []
    assert "Switched openai/gpt-4o -> ollama/qwen3:14b" in outputs.getvalue()


def test_status_after_openai_switch():
    registry = {
        "openai": lambda model=None: FakeProvider(name="openai", model=model or "gpt-4o")
    }
    app, outputs = make_app(
        new_session(), FakeProvider(), ["/switch openai:gpt-4o", "/status", "/quit"],
        registry=registry,
    )
    run(app, outputs)

    text = outputs.getvalue()
    assert "Provider:  openai" in text
    assert "Model:     gpt-4o" in text


def test_status_never_leaks_api_keys():
    app, outputs = make_app(new_session(), FakeProvider(), ["/status", "/quit"])
    run(app, outputs)

    text = outputs.getvalue()
    assert "sk-" not in text
    assert "OPENAI_API_KEY" not in text


def test_switch_usage_lists_providers():
    app, outputs = make_app(new_session(), FakeProvider(), ["/switch", "/quit"])
    run(app, outputs)

    text = outputs.getvalue()
    assert "Usage:" in text
    assert "ollama" in text
    assert "openai" in text


def test_invalid_switch_missing_model():
    session = new_session()
    app, outputs = make_app(session, FakeProvider(), ["/switch ollama:", "/quit"])
    run(app, outputs)

    text = outputs.getvalue()
    assert "missing model name" in text
    assert session.active_provider == "openai"
    assert session.active_model == "gpt-4o"
    assert len(session.archived_conversations) == 0


def test_switch_builder_error_leaves_session_intact():
    session = new_session()
    session.conversation_history.append(
        ConversationMessage(role="user", content="still here")
    )

    def broken(model=None):
        raise APIRequestError("ollama server unreachable")

    app, outputs = make_app(
        session, FakeProvider(), ["/switch ollama:qwen3:14b", "/quit"],
        registry={"ollama": broken},
    )
    run(app, outputs)

    assert "ollama server unreachable" in outputs.getvalue()
    assert session.active_provider == "openai"
    assert session.active_model == "gpt-4o"
    assert len(session.conversation_history) == 1
    assert len(session.archived_conversations) == 0


def test_workspace_defaults_to_cwd_when_env_unset():
    old = os.environ.pop("CHURRO_WORKSPACE", None)
    try:
        assert _workspace_root_from_env() == Path.cwd()
    finally:
        if old is not None:
            os.environ["CHURRO_WORKSPACE"] = old


def test_workspace_honors_churro_workspace_env():
    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.pop("CHURRO_WORKSPACE", None)
        try:
            os.environ["CHURRO_WORKSPACE"] = tmp
            assert _workspace_root_from_env() == Path(tmp)
        finally:
            os.environ.pop("CHURRO_WORKSPACE", None)
            if old is not None:
                os.environ["CHURRO_WORKSPACE"] = old


def test_sessions_dir_is_project_root_sessions():
    assert SESSIONS_DIR == PROJECT_ROOT / "sessions"


def test_default_tool_registry_binds_to_project_root():
    registry = default_tool_registry(PROJECT_ROOT)
    for name in ("read_file", "list_files", "run_tests"):
        assert registry.get(name)._workspace == PROJECT_ROOT


TEST_FUNCTIONS = [
    test_session_starts_correctly,
    test_start_new_session_requires_goal,
    test_user_message_reaches_provider,
    test_ai_response_reaches_session_manager_state_persisted_clean_shown,
    test_status_command,
    test_save_command,
    test_handoff_command,
    test_files_command,
    test_quit_saves_and_exits,
    test_invalid_switch,
    test_successful_model_switch,
    test_new_provider_starts_from_handoff_not_old_history,
    test_handoff_command_excludes_conversation_history,
    test_malformed_ai_response_no_crash,
    test_provider_error_does_not_crash,
    test_empty_input_ignored,
    test_unknown_command,
    test_main_graceful_without_api_key,
    test_switch_ollama_through_default_factory,
    test_switch_ollama_with_registered_builder,
    test_status_after_openai_switch,
    test_status_never_leaks_api_keys,
    test_switch_usage_lists_providers,
    test_invalid_switch_missing_model,
    test_workspace_defaults_to_cwd_when_env_unset,
    test_workspace_honors_churro_workspace_env,
    test_sessions_dir_is_project_root_sessions,
    test_default_tool_registry_binds_to_project_root,
    test_switch_builder_error_leaves_session_intact,
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