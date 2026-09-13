from churro.core.handoff import build_handoff, character_count, estimate_tokens
from churro.core.session import ConversationMessage, create_session


def main() -> None:
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
    session.state.relevant_files = ["auth.py", "tests/test_auth.py"]
    session.state.next_action = "Run the full test suite."
    session.state.notes = "The previous model reported that the targeted tests pass."

    session.conversation_history.append(
        ConversationMessage(
            role="user", content="This conversation is NOT part of the handoff."
        )
    )

    handoff = build_handoff(session)

    print(handoff)
    print()
    print(f"Handoff size: ~{estimate_tokens(handoff)} tokens ({character_count(handoff)} chars)")


if __name__ == "__main__":
    main()