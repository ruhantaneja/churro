import json
import tempfile
from pathlib import Path

from churro.core.session import (
    Session,
    create_session,
    load_session,
    save_session,
)
from churro.core.session_manager import process_ai_response


def state_block(data):
    return "<state_update>\n" + json.dumps(data, indent=2) + "\n</state_update>"


def make_session() -> Session:
    session = create_session(
        goal="Fix the unicode email bug in auth.py",
        provider="openai",
        model="gpt-4o",
    )
    session.state.status = "in_progress"
    session.state.next_action = "Investigate tests"
    session.state.blockers = []
    session.state.completed_work = ["Found auth.py"]
    session.state.relevant_files = ["auth.py"]
    return session


def test_valid_response_updates_state():
    session = make_session()
    proposal = {
        "status": "in_progress",
        "current_task": "Investigating",
        "completed_work": ["Located validation function"],
        "blockers": [],
        "relevant_files": ["auth.py", "tests/test_auth.py"],
        "next_action": "Fix the regex",
        "notes": "Regex rejects unicode",
    }
    result = process_ai_response(session, "Inspected the flow.\n" + state_block(proposal))

    assert session.state.current_task == "Investigating"
    assert session.state.next_action == "Fix the regex"
    assert session.state.relevant_files == ["auth.py", "tests/test_auth.py"]
    assert session.state.notes == "Regex rejects unicode"
    assert result.merged_fields["current_task"] == "Investigating"
    assert result.rejected_fields == {}
    assert result.state_update == proposal


def test_invalid_fields_do_not_overwrite_existing_state():
    session = make_session()
    assert session.state.status == "in_progress"
    assert session.state.next_action == "Investigate tests"
    assert session.state.blockers == []

    proposal = {
        "status": "banana",
        "next_action": "Run tests",
        "blockers": ["Dependency missing"],
    }
    result = process_ai_response(session, "Fixing.\n" + state_block(proposal))

    assert session.state.status == "in_progress"
    assert session.state.next_action == "Run tests"
    assert session.state.blockers == ["Dependency missing"]
    assert result.rejected_fields == {
        "status": "Must be one of: blocked, done, in_progress, not_started"
    }
    assert "status" not in result.merged_fields


def test_missing_fields_preserve_existing_values():
    session = make_session()
    result = process_ai_response(session, "Noted.\n" + state_block({"notes": "New note"}))

    assert session.state.notes == "New note"
    assert session.state.status == "in_progress"
    assert session.state.current_task == ""
    assert session.state.next_action == "Investigate tests"
    assert session.state.completed_work == ["Found auth.py"]
    assert session.state.relevant_files == ["auth.py"]
    assert result.merged_fields == {"notes": "New note"}


def test_partial_state_updates():
    session = make_session()
    proposal = {
        "status": "done",
        "secret_internal_field": "whatever",
        "completed_work": "not a list",
        "next_action": "Ship it",
    }
    result = process_ai_response(session, "All done.\n" + state_block(proposal))

    assert session.state.status == "done"
    assert session.state.next_action == "Ship it"
    assert "secret_internal_field" in result.rejected_fields
    assert "completed_work" in result.rejected_fields
    assert result.merged_fields == {"status": "done", "next_action": "Ship it"}


def test_conversation_history_receives_clean_response():
    session = make_session()
    process_ai_response(session, "I inspected the flow.\n" + state_block({"notes": "x"}))

    assert len(session.conversation_history) == 1
    assert session.conversation_history[0].role == "assistant"
    assert session.conversation_history[0].content == "I inspected the flow."
    assert session.conversation_history[0].timestamp is not None


def test_state_update_tags_not_stored_in_assistant_content():
    session = make_session()
    process_ai_response(session, "I inspected the flow.\n" + state_block({"notes": "x"}))
    process_ai_response(session, "Continuing.\n" + state_block({"status": "blocked"}))

    for message in session.conversation_history:
        assert "<state_update>" not in message.content
        assert "</state_update>" not in message.content


def test_completed_work_merges_without_losing_previous():
    session = make_session()
    assert session.state.completed_work == ["Found auth.py"]

    proposal = {"completed_work": ["Found auth.py", "Located validation function"]}
    process_ai_response(session, "Progress.\n" + state_block(proposal))

    assert session.state.completed_work == [
        "Found auth.py",
        "Located validation function",
    ]

    proposal = {"completed_work": ["Located validation function", "Added a test"]}
    result = process_ai_response(session, "More progress.\n" + state_block(proposal))

    assert session.state.completed_work == [
        "Found auth.py",
        "Located validation function",
        "Added a test",
    ]
    assert result.merged_fields["completed_work"] == session.state.completed_work


def test_blockers_are_replaced():
    session = make_session()
    session.state.blockers = ["Old blocker not relevant anymore"]

    process_ai_response(
        session, "Note.\n" + state_block({"blockers": ["New blocker only"]})
    )

    assert session.state.blockers == ["New blocker only"]


def test_relevant_files_are_replaced():
    session = make_session()
    assert session.state.relevant_files == ["auth.py"]

    process_ai_response(
        session, "Note.\n" + state_block({"relevant_files": ["auth.py", "config.py"]})
    )

    assert session.state.relevant_files == ["auth.py", "config.py"]


def test_malformed_response_does_not_crash():
    session = make_session()
    raw = "Unable to parse this.\n<state_update>\n{\"status\": oops\n</state_update>"

    result = process_ai_response(session, raw)

    assert session.state.status == "in_progress"
    assert session.state.next_action == "Investigate tests"
    assert result.state_update is None
    assert result.merged_fields == {}
    assert result.rejected_fields == {}
    assert len(session.conversation_history) == 1
    assert session.conversation_history[0].content == "Unable to parse this."


def test_session_save_load_after_processing(tmp_path: Path | None = None):
    session = make_session()
    process_ai_response(
        session,
        "Found it.\n" + state_block({"next_action": "Patch the regex"}),
    )

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    path = tmp_path / "sessions" / "processed.json"
    save_session(session, str(path))
    loaded = load_session(str(path))

    assert loaded == session
    assert loaded.state.next_action == "Patch the regex"
    assert loaded.state.completed_work == ["Found auth.py"]
    assert loaded.conversation_history[-1].content == "Found it."


def test_goal_cannot_be_overwritten_by_state_update():
    session = make_session()
    proposal = {"goal": "Replaced goal", "next_action": "Keep going"}
    result = process_ai_response(session, "Doing work.\n" + state_block(proposal))

    assert session.state.goal == "Fix the unicode email bug in auth.py"
    assert session.state.next_action == "Keep going"
    assert result.merged_fields == {"next_action": "Keep going"}
    assert "goal" not in result.merged_fields
    assert any("protected" in w for w in result.warnings)


def test_goal_only_state_update_leaves_state_unchanged():
    session = make_session()
    before = session.state.model_dump()
    result = process_ai_response(
        session, "No change.\n" + state_block({"goal": "some other goal"})
    )

    assert session.state.goal == "Fix the unicode email bug in auth.py"
    assert result.merged_fields == {}
    assert session.state.model_dump() == before
    assert any("protected" in w for w in result.warnings)


TEST_FUNCTIONS = [
    test_valid_response_updates_state,
    test_invalid_fields_do_not_overwrite_existing_state,
    test_missing_fields_preserve_existing_values,
    test_partial_state_updates,
    test_conversation_history_receives_clean_response,
    test_state_update_tags_not_stored_in_assistant_content,
    test_completed_work_merges_without_losing_previous,
    test_blockers_are_replaced,
    test_relevant_files_are_replaced,
    test_malformed_response_does_not_crash,
    test_session_save_load_after_processing,
    test_goal_cannot_be_overwritten_by_state_update,
    test_goal_only_state_update_leaves_state_unchanged,
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