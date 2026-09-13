"""Offline simulation of a CHURRO session without an API key.

Drives the real CHURROApp with a fake provider that returns realistic AI
responses, so you can see the exact terminal UX and a /handoff for free.

Run with:  python -m examples.demo_interactive
"""

import io
import pathlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rich.console import Console

from churro.core.session import create_session
from churro.main import CHURROApp
from churro.prompts import load_system_prompt
from churro.providers.provider import Provider

FAKE_REPLIES = [
    (
        'I found the bug. The regex at auth.py:42 uses [a-zA-Z0-9] and '
        "rejects unicode characters.\n"
        "<state_update>\n"
        '{"status": "in_progress", "current_task": "Analyzing the login flow",'
        ' "completed_work": ["Located the validation regex at auth.py:42"],'
        ' "relevant_files": ["auth.py", "tests/test_auth.py"],'
        ' "blockers": [], "next_action": "Replace the regex with a unicode-aware pattern",'
        ' "notes": "Current pattern rejects unicode email addresses"}\n'
        "</state_update>"
    ),
    (
        "I updated the regex and added three tests for unicode emails. "
        "They pass locally.\n"
        "<state_update>\n"
        '{"status": "done", "current_task": "Verifying the fix",'
        ' "completed_work": ["Replaced the regex with a unicode-aware pattern",'
        ' "Added 3 tests for unicode email addresses"],'
        ' "blockers": [], "next_action": "Run the full test suite",'
        ' "notes": "Targeted tests pass locally"}\n'
        "</state_update>"
    ),
]


class SimulatedProvider(Provider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def send(self, messages):
        return FAKE_REPLIES.pop(0)


def main() -> None:
    outputs = io.StringIO()
    console = Console(file=outputs, force_terminal=False, width=120, markup=False)

    session = create_session(goal="Fix the unicode email login bug", provider="openai", model="gpt-4o")
    path = Path(tempfile.mkdtemp()) / f"{session.session_id}.json"

    lines = [
        "The login fails for users with unicode email addresses.",
        "/status",
        "/handoff",
        "/switch openai:gpt-4o-mini",
        "Run the full test suite now.",
        "/quit",
    ]
    provider = SimulatedProvider()

    app = CHURROApp(
        session=session,
        provider=provider,
        system_prompt=load_system_prompt(),
        session_path=path,
        provider_registry={"openai": lambda model=None: SimulatedProvider(model=model or "gpt-4o")},
        input_fn=lambda prompt="": lines.pop(0),
        console=console,
    )

    app.run()
    print(outputs.getvalue())


if __name__ == "__main__":
    main()