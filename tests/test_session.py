"""Offline tests for the persistent Session model (Step: AgentFrame).

Covers the AgentFrame continuation metadata: construction, bounds, and
round-tripping through the existing save/load mechanism without changing
the shape of legacy sessions.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from churro.core.session import (
    AgentFrame,
    ConversationMessage,
    Session,
    create_session,
    load_session,
    save_session,
)


def make_frame(**overrides) -> AgentFrame:
    values = {
        "task": "Add authentication to the project",
        "iterations_used": 3,
        "max_iterations": 10,
        "stop_reason": "interrupted",
    }
    values.update(overrides)
    return AgentFrame(**values)


# -------------------------------------------------------------- tests
def test_default_pending_agent_is_none():
    session = create_session(goal="Goal", provider="openai", model="gpt-4o")
    assert session.pending_agent is None


def test_agent_frame_constructs_with_required_fields():
    frame = make_frame()
    assert frame.task == "Add authentication to the project"
    assert frame.iterations_used == 3
    assert frame.max_iterations == 10
    assert frame.stop_reason == "interrupted"
    assert frame.tool_digest == []
    assert frame.completed_at is not None


def test_agent_frame_accepts_tool_digest():
    digest = [
        "read_file(call_1) -> success",
        "write_file(call_2) -> success",
        "run_tests(call_3) -> failure",
    ]
    frame = make_frame(tool_digest=digest)
    assert frame.tool_digest == digest


def test_agent_frame_serializes_through_session():
    digest = ["read_file(call_1) -> success"]
    frame = make_frame(tool_digest=digest)
    session = create_session(goal="Goal", provider="openai", model="gpt-4o")
    session.pending_agent = frame

    dumped = session.model_dump()
    agent = dumped["pending_agent"]
    assert agent["task"] == "Add authentication to the project"
    assert agent["iterations_used"] == 3
    assert agent["max_iterations"] == 10
    assert agent["stop_reason"] == "interrupted"
    assert agent["tool_digest"] == digest

    round_trip = Session.model_validate(dumped)
    assert round_trip.pending_agent == frame


def test_pending_agent_frame_has_no_output_or_history():
    session = create_session(goal="Goal", provider="openai", model="gpt-4o")
    session.conversation_history.append(
        ConversationMessage(role="user", content="SECRET_HISTORY")
    )
    session.pending_agent = make_frame(
        tool_digest=["read_file(call_1) -> success"]
    )

    dumped = session.pending_agent.model_dump()
    assert dumped["tool_digest"] == ["read_file(call_1) -> success"]
    assert "output" not in dumped
    assert "result" not in dumped
    assert "SECRET_HISTORY" not in session.pending_agent.model_dump_json()


def test_session_with_pending_agent_saves_and_loads_losslessly():
    digest = ["read_file(call_1) -> success", "run_tests(call_2) -> failure"]
    frame = make_frame(tool_digest=digest)
    session = create_session(goal="Goal", provider="openai", model="gpt-4o")
    session.pending_agent = frame
    session.state.current_task = "Add routes"
    session.conversation_history.append(
        ConversationMessage(role="user", content="Continue")
    )

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "session.json"
        save_session(session, str(path))
        loaded = load_session(str(path))

    assert loaded == session
    assert loaded.pending_agent == frame
    assert loaded.pending_agent.tool_digest == digest
    assert loaded.state.current_task == "Add routes"


def test_legacy_session_without_pending_agent_loads():
    legacy = {
        "session_id": "legacy12345",
        "created_at": "2026-09-13T08:23:03.570266Z",
        "active_provider": "openai",
        "active_model": "gpt-4o",
        "state": {
            "goal": "Fix the login bug",
            "current_task": "",
            "completed_work": ["Found auth.py"],
            "status": "in_progress",
            "blockers": [],
            "relevant_files": ["auth.py"],
            "next_action": "Patch it",
            "notes": "",
        },
        "conversation_history": [
            {
                "role": "user",
                "content": "Fix the login bug",
                "timestamp": "2026-09-13T08:23:03.570322Z",
            }
        ],
        "archived_conversations": [],
    }
    raw = json.dumps(legacy)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "legacy.json"
        path.write_text(raw, encoding="utf-8")
        loaded = load_session(str(path))

    assert loaded.pending_agent is None
    assert loaded.session_id == "legacy12345"
    assert loaded.active_provider == "openai"
    assert loaded.active_model == "gpt-4o"
    assert loaded.state.goal == "Fix the login bug"
    assert loaded.state.status == "in_progress"
    assert loaded.state.completed_work == ["Found auth.py"]
    assert loaded.state.next_action == "Patch it"
    assert len(loaded.conversation_history) == 1
    assert loaded.archived_conversations == []


def test_completed_at_is_set_and_serialized():
    frame = make_frame()
    assert isinstance(frame.completed_at.isoformat(), str)
    raw = frame.model_dump_json()
    assert "completed_at" in raw


# ------------------------------------------------------- bounds/validation
def test_negative_iterations_used_rejected():
    with pytest.raises(ValidationError):
        make_frame(iterations_used=-1)


def test_non_positive_max_iterations_rejected():
    for bad in (0, -5):
        with pytest.raises(ValidationError):
            make_frame(max_iterations=bad)


def test_iterations_used_beyond_budget_rejected():
    with pytest.raises(ValidationError):
        make_frame(iterations_used=11)


def test_empty_task_rejected():
    with pytest.raises(ValidationError):
        make_frame(task="")


def test_empty_stop_reason_rejected():
    with pytest.raises(ValidationError):
        make_frame(stop_reason="")


def test_empty_tool_digest_line_rejected():
    with pytest.raises(ValidationError):
        make_frame(tool_digest=[""])


def test_oversized_tool_digest_line_rejected():
    with pytest.raises(ValidationError):
        make_frame(tool_digest=["x" * 1000])


def test_too_many_tool_digest_lines_rejected():
    with pytest.raises(ValidationError):
        make_frame(tool_digest=["x"] * 201)


TEST_FUNCTIONS = [
    test_default_pending_agent_is_none,
    test_agent_frame_constructs_with_required_fields,
    test_agent_frame_accepts_tool_digest,
    test_agent_frame_serializes_through_session,
    test_pending_agent_frame_has_no_output_or_history,
    test_session_with_pending_agent_saves_and_loads_losslessly,
    test_legacy_session_without_pending_agent_loads,
    test_completed_at_is_set_and_serialized,
    test_negative_iterations_used_rejected,
    test_non_positive_max_iterations_rejected,
    test_iterations_used_beyond_budget_rejected,
    test_empty_task_rejected,
    test_empty_stop_reason_rejected,
    test_empty_tool_digest_line_rejected,
    test_oversized_tool_digest_line_rejected,
    test_too_many_tool_digest_lines_rejected,
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