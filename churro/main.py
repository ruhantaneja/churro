"""CHURRO v0.1 - the interactive terminal application.

Run with:  python -m churro   (requires OPENAI_API_KEY)
"""

import sys
from pathlib import Path
from typing import Callable

from rich.console import Console

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
from churro.providers.openai_provider import OpenAIProvider
from churro.providers.provider import Provider, ProviderError, MissingAPIKeyError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_ROOT / "sessions"

ProviderFactory = Callable[[str | None], Provider]

HELP_TEXT = """Commands:
  <message>                    Send a message to the active model
  /status                      Show the persistent project state
  /save                        Save the session to disk now
  /switch <provider[:model]>   Switch provider/model (v0.1: openai only)
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
        provider_registry: dict[str, ProviderFactory] | None = None,
        input_fn: Callable[[str], str] | None = None,
        console: Console | None = None,
    ):
        self.session = session
        self.provider = provider
        self.system_prompt = system_prompt
        self.session_path = Path(session_path)

        if provider_registry is None:
            def _make_openai(model: str | None = None) -> Provider:
                return OpenAIProvider(model=model) if model else OpenAIProvider()
            provider_registry = {"openai": _make_openai}
        self.provider_registry = provider_registry

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
        return "\n\n".join(
            [
                self.system_prompt,
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

        self.console.print(f"Unknown command: {cmd}", markup=False)
        self.console.print("Type /help to list commands.", markup=False)
        return True

    def _command_status(self) -> None:
        s = self.session
        st = s.state
        lines = [
            f"Session:   {s.session_id}",
            f"Created:   {s.created_at}",
            f"Provider:  {s.active_provider} / {s.active_model}",
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

    def _command_switch(self, spec: str) -> None:
        available = ", ".join(sorted(self.provider_registry))
        if not spec:
            self.console.print("Usage: /switch <provider>  or  /switch <provider:model>", markup=False)
            self.console.print(f"Available providers: {available}", markup=False)
            return

        name, _, model = spec.partition(":")
        name = name.strip().lower()
        model = model.strip() or None

        factory = self.provider_registry.get(name)
        if factory is None:
            self.console.print(
                f"Provider '{name}' is not available in CHURRO v0.1.", markup=False
            )
            self.console.print(f"Available providers: {available}", markup=False)
            return

        try:
            new_provider = factory(model)
        except ProviderError as exc:
            self.console.print(f"Could not build provider '{name}': {exc}", markup=False)
            return

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

    session = create_session(goal=goal, provider="openai", model=model)
    path = sessions_dir / f"{session.session_id}.json"
    save_session(session, str(path))
    return session, path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    console = Console()

    try:
        system_prompt = load_system_prompt()
        provider: Provider = OpenAIProvider()
    except MissingAPIKeyError as exc:
        console.print(str(exc), markup=False)
        console.print(
            "Set OPENAI_API_KEY in your environment, then run: python -m churro",
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
            session, path = _start_new_session(console, input, provider.model)
    except (OSError, ValueError, KeyboardInterrupt, EOFError) as exc:
        console.print(f"Could not start session: {exc}", markup=False)
        return 1

    def _make_openai(model: str | None = None) -> Provider:
        return OpenAIProvider(model=model) if model else OpenAIProvider()

    app = CHURROApp(
        session=session,
        provider=provider,
        system_prompt=system_prompt,
        session_path=path,
        provider_registry={"openai": _make_openai},
        console=console,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())