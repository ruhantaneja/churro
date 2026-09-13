from pathlib import Path

from churro.core.session import (
    ConversationMessage,
    Session,
    create_session,
    load_session,
    save_session,
)

SAVE_PATH = Path(__file__).resolve().parent.parent / "sessions" / "demo.json"


def main() -> None:
    session = create_session(
        goal="Fix the unicode email login bug in auth.py",
        provider="openai",
        model="gpt-4o",
    )
    print(f"Created session: {session.session_id}")

    session.state.current_task = "Analyzing the email validation function"
    session.state.completed_work = ["Located the regex at auth.py:42"]
    session.state.status = "in_progress"
    session.state.relevant_files = ["auth.py", "tests/test_auth.py"]
    session.state.next_action = "Update the regex to support unicode emails"
    session.conversation_history.append(
        ConversationMessage(role="user", content="Fix the unicode email login bug in auth.py")
    )
    session.conversation_history.append(
        ConversationMessage(
            role="assistant",
            content="I found the bug at auth.py:42 where the regex rejects unicode emails.",
        )
    )

    save_session(session, str(SAVE_PATH))
    print(f"Saved to: {SAVE_PATH}")

    loaded = load_session(str(SAVE_PATH))
    print(f"Loaded session: {loaded.session_id}")

    assert loaded == session, "Loaded session does not match the original!"
    print("VERIFIED: loaded session matches the original.")

    print("\n--- Loaded session state ---")
    print(f"goal:            {loaded.state.goal}")
    print(f"current_task:    {loaded.state.current_task}")
    print(f"status:          {loaded.state.status}")
    print(f"completed_work:  {loaded.state.completed_work}")
    print(f"relevant_files:  {loaded.state.relevant_files}")
    print(f"next_action:     {loaded.state.next_action}")
    print("\n--- Conversation history ---")
    for message in loaded.conversation_history:
        print(f"[{message.role}] {message.content}")


if __name__ == "__main__":
    main()