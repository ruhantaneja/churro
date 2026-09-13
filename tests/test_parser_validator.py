import json

from churro.core.parser import parse_response
from churro.core.validator import validate_state_update


def state_block(data):
    return "<state_update>\n" + json.dumps(data, indent=2) + "\n</state_update>"


def test_valid_response_and_state():
    data = {
        "status": "in_progress",
        "current_task": "Investigating the authentication bug",
        "completed_work": ["Located the authentication function"],
        "blockers": [],
        "relevant_files": ["auth.py"],
        "next_action": "Inspect the validation logic",
        "notes": "The validation regex may be rejecting unicode",
    }
    raw = "I found a bug. Let me investigate.\n" + state_block(data)

    parsed = parse_response(raw)
    assert parsed.clean_response == "I found a bug. Let me investigate."
    assert parsed.state_update == data
    assert parsed.warnings == []

    validated = validate_state_update(parsed.state_update)
    assert validated.valid == data
    assert validated.invalid == {}
    assert validated.warnings == []


def test_no_state_block():
    parsed = parse_response("Just a plain reply. No state here.")
    assert parsed.state_update is None
    assert parsed.clean_response == "Just a plain reply. No state here."


def test_malformed_json():
    raw = "Here is my plan.\n<state_update>\n{\"status\": \"in_progress\", oops}\n</state_update>"
    parsed = parse_response(raw)
    assert parsed.state_update is None
    assert parsed.clean_response == "Here is my plan."
    assert any("not valid JSON" in w for w in parsed.warnings)


def test_missing_closing_tag():
    raw = "Working on it.\n<state_update>\n{\"status\": \"in_progress\"}\n"
    parsed = parse_response(raw)
    assert parsed.state_update is None
    assert parsed.clean_response == "Working on it."
    assert any("without a closing tag" in w for w in parsed.warnings)


def test_invalid_status():
    proposal = {"status": "banana", "notes": "some note"}
    validated = validate_state_update(proposal)
    assert "status" in validated.invalid
    assert validated.valid == {"notes": "some note"}


def test_wrong_field_types():
    proposal = {
        "status": "in_progress",
        "completed_work": "auth.py",
        "blockers": [1, 2],
    }
    validated = validate_state_update(proposal)
    assert "completed_work" in validated.invalid
    assert "blockers" in validated.invalid
    assert validated.valid == {"status": "in_progress"}


def test_unknown_fields_rejected():
    proposal = {"status": "in_progress", "secret_internal_field": "whatever"}
    validated = validate_state_update(proposal)
    assert validated.valid == {"status": "in_progress"}
    assert "secret_internal_field" in validated.invalid
    assert "Unknown field" in validated.invalid["secret_internal_field"]


def test_partially_valid_state():
    proposal = {
        "status": "banana",
        "next_action": "Run tests",
        "blockers": ["Missing dependency"],
    }
    validated = validate_state_update(proposal)
    assert "status" in validated.invalid
    assert validated.valid == {
        "next_action": "Run tests",
        "blockers": ["Missing dependency"],
    }


def test_empty_strings_where_prohibited():
    proposal = {"goal": "", "next_action": "   ", "notes": ""}
    validated = validate_state_update(proposal)
    assert "goal" in validated.invalid
    assert "next_action" in validated.invalid
    assert validated.valid == {"notes": ""}


def test_multiple_state_blocks():
    raw = (
        "First update here.\n"
        + state_block({"status": "in_progress"})
        + "Still working.\n"
        + state_block({"status": "blocked"})
    )
    parsed = parse_response(raw)
    assert parsed.clean_response == "First update here.\nStill working."
    assert parsed.state_update == {"status": "in_progress"}
    assert any("only the first" in w for w in parsed.warnings)


def test_state_block_at_start():
    raw = state_block({"status": "done"}) + "\nAll finished!"
    parsed = parse_response(raw)
    assert parsed.clean_response == "All finished!"
    assert parsed.state_update == {"status": "done"}


def test_state_block_not_a_json_object():
    raw = "Reply here.\n" + state_block([1, 2, 3])
    parsed = parse_response(raw)
    assert parsed.state_update is None
    assert parsed.clean_response == "Reply here."
    assert any("not a JSON object" in w for w in parsed.warnings)


def test_no_valid_fields_means_state_not_updated():
    proposal = {"status": "banana", "completed_work": 42}
    validated = validate_state_update(proposal)
    assert validated.valid == {}
    assert validated.accepted is False
    assert any("No fields passed" in w for w in validated.warnings)


TEST_FUNCTIONS = [
    test_valid_response_and_state,
    test_no_state_block,
    test_malformed_json,
    test_missing_closing_tag,
    test_invalid_status,
    test_wrong_field_types,
    test_unknown_fields_rejected,
    test_partially_valid_state,
    test_empty_strings_where_prohibited,
    test_multiple_state_blocks,
    test_state_block_at_start,
    test_state_block_not_a_json_object,
    test_no_valid_fields_means_state_not_updated,
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