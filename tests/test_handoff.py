import json
import tempfile
from pathlib import Path

from churro.core.handoff import (
    CHARS_PER_TOKEN,
    build_handoff,
    character_count,
    estimate_tokens,
)
from churro.core.session import ConversationMessage, Session, SessionState, create_session

EXPECTED_HANDOFF = (
    "You are taking over an existing coding task in CHURRO.\n"
    "\n"
    "PROJECT STATE\n"
    "\n"
    "Goal:\n"
    "Fix the unicode email login bug.\n"
    "\n"
    "Status:\n"
    "in_progress\n"
    "\n"
    "Current task:\n"
    "Verify the authentication fix.\n"
    "\n"
    "Completed work:\n"
    "* Located the validation regex in auth.py\n"
    "* Updated the regex\n"
    "* Added tests for unicode email addresses\n"
    "\n"
    "Blockers:\n"
    "None\n"
    "\n"
    "Relevant files:\n"
    "* auth.py\n"
    "* tests/test_auth.py\n"
    "\n"
    "Next action:\n"
    "Run the full test suite.\n"
    "\n"
    "Notes:\n"
    "The previous model reported that the targeted tests pass.\n"
    "\n"
    "Continue from this state. Do not repeat completed work unnecessarily."
)


def full_session() -> Session:
    session = create_session(
        goal="Fix the unicode email login bug.",
        provider="openai",
        model="gpt-4o",
    )
    session.state.status = "in_progress"
    session.state.current_task = "Verify the authentication fix."
    session.state.completed_work = [
        "Located the validation regex in auth.py",
        "Updated the regex",
        "Added tests for unicode email addresses",
    ]
    session.state.blockers = []
    session.state.relevant_files = ["auth.py", "tests/test_auth.py"]
    session.state.next_action = "Run the full test suite."
    session.state.notes = "The previous model reported that the targeted tests pass."
    return session


def test_complete_state_produces_correct_handoff():
    handoff = build_handoff(full_session())
    assert handoff == EXPECTED_HANDOFF


def test_empty_fields_handled_cleanly():
    session = create_session(goal="Some goal", provider="openai", model="gpt-4o")
    handoff = build_handoff(session)

    assert "Status:\nnot_started" in handoff
    assert "Current task:\nNone" in handoff
    assert "Completed work:\nNone" in handoff
    assert "Blockers:\nNone" in handoff
    assert "Relevant files:\nNone" in handoff
    assert "Next action:\nNone" in handoff
    assert "Notes:\nNone" in handoff
    assert "Goal:\nSome goal" in handoff


def test_conversation_history_not_included():
    session = full_session()
    secret_text = "SNEAKY_HISTORY_LINE_12345"
    session.conversation_history.append(
        ConversationMessage(role="user", content=secret_text)
    )
    session.conversation_history.append(
        ConversationMessage(role="assistant", content="Another response")
    )

    handoff = build_handoff(session)
    assert secret_text not in handoff
    assert "Another response" not in handoff


def test_provider_information_not_included():
    session = full_session()
    handoff = build_handoff(session)

    assert "openai" not in handoff.lower()
    assert "gpt" not in handoff.lower()
    assert "anthropic" not in handoff.lower()
    assert "claude" not in handoff.lower()
    assert "provider" not in handoff.lower()


def test_all_state_fields_appear():
    session = full_session()
    handoff = build_handoff(session)

    assert "Goal:" in handoff and "Fix the unicode email login bug." in handoff
    assert "Status:" in handoff and "in_progress" in handoff
    assert "Current task:" in handoff and "Verify the authentication fix." in handoff
    assert "Completed work:" in handoff
    assert "Located the validation regex in auth.py" in handoff
    assert "Blockers:" in handoff
    assert "Relevant files:" in handoff and "tests/test_auth.py" in handoff
    assert "Next action:" in handoff and "Run the full test suite." in handoff
    assert "Notes:" in handoff


def test_handoff_readable():
    handoff = build_handoff(full_session())
    assert handoff.startswith("You are taking over an existing coding task in CHURRO.")
    assert "\n\nPROJECT STATE\n\n" in handoff
    assert handoff.endswith(
        "Continue from this state. Do not repeat completed work unnecessarily."
    )
    assert "  " not in handoff
    assert "\n\n\n" not in handoff


def test_token_estimate_returned():
    handoff = build_handoff(full_session())
    chars = character_count(handoff)
    tokens = estimate_tokens(handoff)

    assert chars > 0
    assert tokens > 0
    assert tokens == max(1, (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)
    assert tokens <= chars


def test_deterministic_output():
    session = full_session()
    first = build_handoff(session)
    second = build_handoff(session)
    assert first == second


def test_build_handoff_does_not_mutate_session():
    session = full_session()
    before = session.model_dump()

    build_handoff(session)

    assert session.model_dump() == before


def test_large_completed_work_measurable():
    session = create_session(goal="Big task", provider="openai", model="gpt-4o")
    session.state.completed_work = [f"Step {i} complete" for i in range(100)]

    small = build_handoff(full_session())
    large = build_handoff(session)

    assert character_count(large) > character_count(small)
    assert estimate_tokens(large) > estimate_tokens(small)
    assert estimate_tokens(large) > 100


TEST_FUNCTIONS = [
    test_complete_state_produces_correct_handoff,
    test_empty_fields_handled_cleanly,
    test_conversation_history_not_included,
    test_provider_information_not_included,
    test_all_state_fields_appear,
    test_handoff_readable,
    test_token_estimate_returned,
    test_deterministic_output,
    test_build_handoff_does_not_mutate_session,
    test_large_completed_work_measurable,
]


def main() -> None:
    failures = 0
    for fn in TEST_FUNCTIONS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            print(f"PASS  {fn.__name__}")
    if failures:
        raise SystemExit(f"\n{failures} test(s) failed")
    print(f"\nAll {len(TEST_FUNCTIONS)} tests passed.")


if __name__ == "__main__":
    main()