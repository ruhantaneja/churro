from churro.core.parser import parse_response
from churro.core.validator import validate_state_update

CASES = {
    "1. Valid response + state": (
        "I found the bug in auth.py.\n"
        "<state_update>\n"
        '{"status": "in_progress", "current_task": "Investigating",'
        ' "completed_work": ["Located the auth function"], "blockers": [],'
        ' "relevant_files": ["auth.py"], "next_action": "Inspect logic",'
        ' "notes": "Regex may reject unicode"}\n'
        "</state_update>"
    ),
    "2. Partially valid state": (
        "Fixing now.\n"
        "<state_update>\n"
        '{"status": "banana", "next_action": "Run tests",'
        ' "blockers": ["Missing dependency"]}\n'
        "</state_update>"
    ),
    "3. Unknown field": (
        "On it.\n"
        "<state_update>\n"
        '{"status": "in_progress", "secret_internal_field": "whatever"}\n'
        "</state_update>"
    ),
    "4. Malformed JSON": (
        "Looking into it.\n"
        "<state_update>\n"
        '{"status": "in_progress", oops}\n'
        "</state_update>"
    ),
}


def main() -> None:
    for label, raw in CASES.items():
        print(f"=== {label} ===")
        print(f"INPUT    : {raw!r}")
        parsed = parse_response(raw)
        print(f"  PARSER -> clean_response: {parsed.clean_response!r}")
        print(f"           state_update  : {parsed.state_update}")
        print(f"           warnings      : {parsed.warnings}")
        validated = validate_state_update(parsed.state_update)
        print(f"  VALIDATOR -> valid      : {validated.valid}")
        print(f"               invalid    : {validated.invalid}")
        print(f"               warnings   : {validated.warnings}")
        print()
        assert not parsed.state_update or "No fields passed" not in "".join(validated.warnings)


if __name__ == "__main__":
    main()