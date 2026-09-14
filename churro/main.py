"""CHURRO - the interactive terminal application.

Run with:  python -m churro   (requires OPENAI_API_KEY by default)

The startup provider is configurable with CHURRO_PROVIDER / CHURRO_MODEL
(see factory.startup_provider_spec), so local-Ollama-only setups start
without an OpenAI key.
"""

import json
import os
import sys
from pathlib import Path
from typing import Callable

from rich.console import Console

from churro.agent import AgentResult, AgentRunner, DEFAULT_MAX_ITERATIONS
from churro.core.handoff import build_handoff, character_count, estimate_tokens, format_state
from churro.core.session import (
    ArchivedConversation,
    ConversationMessage,
    Session,
    create_session,
    load_session,
    save_session,
)
from churro.core.session_manager import process_ai_response
from churro.prompts import load_system_prompt
from churro.providers.factory import (
    ProviderFactory,
    default_provider_factory,
    parse_provider_spec,
    startup_provider_spec,
)
from churro.providers.provider import Provider, ProviderError, MissingAPIKeyError
from churro.tools import ToolRegistry, default_tool_registry

CHURRO_WORKSPACE_ENV = "CHURRO_WORKSPACE"


def _workspace_root_from_env() -> Path:
    """Return the workspace root from ``CHURRO_WORKSPACE`` or ``Path.cwd()``."""
    raw = os.environ.get(CHURRO_WORKSPACE_ENV, "").strip()
    return Path(raw) if raw else Path.cwd()


PROJECT_ROOT = _workspace_root_from_env()
SESSIONS_DIR = PROJECT_ROOT / "sessions"

AGENT_MAX_ITERATIONS_ENV = "CHURRO_AGENT_MAX_ITERATIONS"

# Application-level activity status shown while the agent waits on the model.
AGENT_STATUS_MESSAGE = "CHURRO is thinking..."


def _agent_iteration_limit_from_env() -> int:
    raw = os.environ.get(AGENT_MAX_ITERATIONS_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_ITERATIONS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_ITERATIONS
    return value if value > 0 else DEFAULT_MAX_ITERATIONS

HELP_TEXT = """Commands:
  <message>                    Send a message to the active model
  /status                      Show the persistent project state
  /save                        Save the session to disk now
  /switch <provider[:model]>   Switch provider/model (openai, ollama)
  /agent <task>                Run a bounded AgentRunner over the task
  /handoff                     Show the handoff another model would receive
  /files                       Show relevant files from state
  /help                        Show this help
  /quit                        Save and exit"""


class CHURROApp:
    def __init__(
        self,
        session: Session,
        provider: Provider,
        system_prompt: str,
        session_path: str | Path,
        provider_registry: dict[str, Callable[[str | None], Provider]] | None = None,
        provider_factory: ProviderFactory | None = None,
        tool_registry: ToolRegistry | None = None,
        agent_max_iterations: int | None = None,
        input_fn: Callable[[str], str] | None = None,
        console: Console | None = None,
    ):
        self.session = session
        self.provider = provider
        self.system_prompt = system_prompt
        self.session_path = Path(session_path)

        if provider_factory is None:
            factory = ProviderFactory()
            if provider_registry is not None:
                for name, builder in provider_registry.items():
                    factory.register(name, builder)
            else:
                factory = default_provider_factory()
            provider_factory = factory
        self.provider_factory = provider_factory

        if tool_registry is None:
            tool_registry = default_tool_registry(PROJECT_ROOT)
        self.tool_registry = tool_registry

        if agent_max_iterations is None:
            agent_max_iterations = _agent_iteration_limit_from_env()
        if (
            isinstance(agent_max_iterations, bool)
            or not isinstance(agent_max_iterations, int)
            or agent_max_iterations <= 0
        ):
            raise ValueError("agent_max_iterations must be a positive integer")
        self.agent_max_iterations = agent_max_iterations

        self.input_fn = input_fn if input_fn is not None else input
        self.console = console if console is not None else Console()
        self._pending_handoff: str | None = None

    # ------------------------------------------------------------------ loop
    def run(self) -> int:
        self._print_banner()
        while True:
            try:
                raw = self.input_fn("> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                self.console.print("\nInterrupted. Saving...", markup=False)
                break

            text = raw.strip()
            if not text:
                continue
            if text.startswith("/"):
                if not self._process_command(text):
                    break
            else:
                self._handle_message(text)

        self._save()
        self.console.print("Session saved. Goodbye.", markup=False)
        return 0

    def _print_banner(self) -> None:
        self.console.print("CHURRO", markup=False)
        self.console.print(
            f"Model: {self.provider.name}/{self.provider.model}", markup=False
        )
        self.console.print(f"Session: {self.session.session_id}", markup=False)
        self.console.print("Type /help for commands.", markup=False)

    # --------------------------------------------------------------- messages
    def _handle_message(self, text: str) -> None:
        self.session.conversation_history.append(
            ConversationMessage(role="user", content=text)
        )

        messages = self._build_messages()
        try:
            raw_response = self.provider.send(messages)
        except ProviderError as exc:
            self.console.print(f"Provider error: {exc}", markup=False)
            self._save()
            return

        result = process_ai_response(self.session, raw_response)

        self.console.print("AI:", markup=False)
        self.console.print(result.clean_response, markup=False)
        for warning in result.warnings:
            self.console.print(f"note: {warning}", markup=False)
        self._save()

    def _build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self._pending_handoff is not None:
            messages.append({"role": "system", "content": self._pending_handoff})
            self._pending_handoff = None
        messages.append({"role": "system", "content": self._build_system_content()})
        for msg in self.session.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        return messages

    def _build_system_content(self) -> str:
        from churro.repo_index import build_summary

        return "\n\n".join(
            [
                self.system_prompt,
                "REPOSITORY OVERVIEW",
                build_summary(PROJECT_ROOT),
                "CURRENT PROJECT STATE",
                format_state(self.session.state),
            ]
        )

    # -------------------------------------------------------------- commands
    def _process_command(self, text: str) -> bool:
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        arg = arg.strip()

        if cmd == "/quit":
            return False
        if cmd == "/help":
            self.console.print(HELP_TEXT, markup=False)
            return True
        if cmd == "/save":
            try:
                self._save()
            except OSError as exc:
                self.console.print(f"Save failed: {exc}", markup=False)
            else:
                self.console.print(f"Saved to {self.session_path}.", markup=False)
            return True
        if cmd == "/status":
            self._command_status()
            return True
        if cmd == "/handoff":
            self._command_handoff()
            return True
        if cmd == "/files":
            self._command_files()
            return True
        if cmd == "/switch":
            self._command_switch(arg)
            return True
        if cmd == "/agent":
            self._command_agent(arg)
            return True

        self.console.print(f"Unknown command: {cmd}", markup=False)
        self.console.print("Type /help to list commands.", markup=False)
        return True

    def _command_status(self) -> None:
        s = self.session
        st = s.state
        lines = [
            f"Session:   {s.session_id}",
            f"Created:   {s.created_at}",
            f"Provider:  {s.active_provider}",
            f"Model:     {s.active_model}",
            f"Status:    {st.status}",
            f"Goal:      {st.goal}",
            f"Current:   {st.current_task}",
        ]
        if st.completed_work:
            lines.append("Completed:")
            lines.extend(f"  * {item}" for item in st.completed_work)
        lines.append(
            "Blockers:  " + (", ".join(st.blockers) if st.blockers else "None")
        )
        lines.append(
            "Files:     " + (", ".join(st.relevant_files) if st.relevant_files else "None")
        )
        lines.append(f"Next:      {st.next_action}")
        if st.notes:
            lines.append(f"Notes:     {st.notes}")
        lines.append(
            f"Turns:     {len(s.conversation_history)} active / "
            f"{len(s.archived_conversations)} archived"
        )
        self.console.print("\n".join(lines), markup=False)

    def _command_handoff(self) -> None:
        handoff = build_handoff(self.session)
        chars = character_count(handoff)
        tokens = estimate_tokens(handoff)
        self.console.print(handoff, markup=False)
        self.console.print(
            f"\nHandoff size: ~{tokens} tokens ({chars} chars)", markup=False
        )

    def _command_files(self) -> None:
        files = self.session.state.relevant_files
        if files:
            self.console.print("\n".join(files), markup=False)
        else:
            self.console.print("None", markup=False)

    def _command_agent(self, task: str) -> None:
        task = task.strip()
        if not task:
            self.console.print("Usage: /agent <task>", markup=False)
            self.console.print(
                "Starts a bounded AgentRunner that may use the project tools.",
                markup=False,
            )
            return

        self.session.conversation_history.append(
            ConversationMessage(role="user", content=task)
        )

        self.console.print("-- Agent --", markup=False)
        self.console.print(f"> Starting agent on task: {task}", markup=False)

        runner = AgentRunner(
            provider=self.provider,
            registry=self.tool_registry,
            max_iterations=self.agent_max_iterations,
        )
        try:
            with self.console.status(AGENT_STATUS_MESSAGE, spinner="dots"):
                result: AgentResult = runner.run(self._build_messages())
        except KeyboardInterrupt:
            self.console.print("> Agent interrupted by Ctrl+C.", markup=False)
            self._save()
            return

        arguments_by_id: dict[str, dict] = {}
        for message in result.messages:
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                call_id = getattr(call, "id", "")
                if call_id:
                    arguments_by_id[call_id] = getattr(call, "arguments", {}) or {}

        for execution in result.tool_executions:
            marker = "" if execution.result.success else " (failed)"
            self.console.print(f"[tool] {execution.tool_name}{marker}", markup=False)
            arguments = arguments_by_id.get(execution.call_id)
            if arguments is not None:
                self.console.print(
                    f"      args: {json.dumps(arguments, sort_keys=True)}",
                    markup=False,
                )
            if not execution.result.success:
                reason = (execution.result.error or "").strip()
                if not reason:
                    reason = "(no error details)"
                self.console.print(f"      error: {reason}", markup=False)

        raw_final = result.final_text.strip()
        processed = None
        clean_answer = ""
        if raw_final:
            processed = process_ai_response(self.session, raw_final)
            clean_answer = processed.clean_response
        else:
            self.session.conversation_history.append(
                ConversationMessage(role="assistant", content="(agent output)")
            )

        if result.completed:
            if not clean_answer:
                self.console.print(
                    "> Agent finished with no text response.", markup=False
                )
            else:
                self.console.print(">", markup=False)
                self.console.print(clean_answer, markup=False)
        else:
            reason = result.error or (
                f"iteration limit reached after {result.iterations} iterations"
            )
            self.console.print(f"> Agent stopped: {reason}", markup=False)
            if clean_answer:
                self.console.print(clean_answer, markup=False)

        if processed is not None:
            for warning in processed.warnings:
                self.console.print(f"note: {warning}", markup=False)

        self._save()

    def _command_switch(self, spec: str) -> None:
        available = ", ".join(self.provider_factory.supported_providers())
        if not spec:
            self.console.print(
                "Usage: /switch <provider>  or  /switch <provider:model>", markup=False
            )
            self.console.print(f"Available providers: {available}", markup=False)
            return

        try:
            parsed = parse_provider_spec(spec)
            new_provider = self.provider_factory.create(parsed)
        except ProviderError as exc:
            self.console.print(f"Could not build provider: {exc}", markup=False)
            return

        name = parsed.provider_name
        old = f"{self.session.active_provider}/{self.session.active_model}"
        self.session.archived_conversations.append(
            ArchivedConversation(
                provider=self.session.active_provider,
                model=self.session.active_model,
                messages=list(self.session.conversation_history),
            )
        )
        self.session.conversation_history = []
        self.session.active_provider = name
        self.session.active_model = new_provider.model
        self.provider = new_provider
        self._pending_handoff = build_handoff(self.session)

        self._save()
        new = f"{name}/{new_provider.model}"
        self.console.print(f"Switched {old} -> {new}", markup=False)
        self.console.print(
            "The new model starts from a fresh handoff; the old conversation "
            "context was archived.",
            markup=False,
        )

    # ------------------------------------------------------------ persistence
    def _save(self) -> None:
        save_session(self.session, str(self.session_path))


def _start_new_session(
    console: Console,
    input_fn: Callable[[str], str],
    model: str,
    sessions_dir: Path | None = None,
    provider_name: str = "openai",
) -> tuple[Session, Path]:
    sessions_dir = sessions_dir or SESSIONS_DIR
    sessions_dir.mkdir(parents=True, exist_ok=True)

    console.print("CHURRO", markup=False)
    console.print("What are we working on?", markup=False)
    while True:
        goal = input_fn("> ").strip()
        if goal:
            break
        console.print("A goal is required to start a session.", markup=False)

    session = create_session(goal=goal, provider=provider_name, model=model)
    path = sessions_dir / f"{session.session_id}.json"
    save_session(session, str(path))
    return session, path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    console = Console()

    try:
        system_prompt = load_system_prompt()
        factory = default_provider_factory()
        spec = startup_provider_spec()
        parsed = parse_provider_spec(spec)
        provider: Provider = factory.create(parsed)
    except MissingAPIKeyError as exc:
        console.print(str(exc), markup=False)
        console.print(
            "Set OPENAI_API_KEY in your environment, then run: python -m churro",
            markup=False,
        )
        console.print(
            "Or start with a local provider: set CHURRO_PROVIDER=ollama "
            "(and CHURRO_MODEL=<model>) in your environment.",
            markup=False,
        )
        return 1
    except ProviderError as exc:
        console.print(f"Startup error: {exc}", markup=False)
        return 1

    try:
        if argv:
            path = Path(argv[0])
            session = load_session(str(path))
            console.print(f"Loaded session {session.session_id}.", markup=False)
        else:
            session, path = _start_new_session(
                console,
                input,
                provider.model,
                provider_name=provider.name,
            )
    except (OSError, ValueError, KeyboardInterrupt, EOFError) as exc:
        console.print(f"Could not start session: {exc}", markup=False)
        return 1

    app = CHURROApp(
        session=session,
        provider=provider,
        system_prompt=system_prompt,
        session_path=path,
        provider_factory=factory,
        console=console,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())