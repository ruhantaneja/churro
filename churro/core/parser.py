import json
from dataclasses import dataclass, field
from typing import Any

START_TAG = "<state_update>"
END_TAG = "</state_update>"


@dataclass
class ParseResult:
    clean_response: str = ""
    state_update: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


def parse_response(raw: str) -> ParseResult:
    """Split a raw AI response into clean readable text and an optional state update.

    The parser is side-effect free: it never touches session state.
    """
    if not isinstance(raw, str):
        return ParseResult(warnings=["Input is not a string"])

    if START_TAG not in raw:
        return ParseResult(clean_response=raw.strip())

    warnings: list[str] = []

    if END_TAG not in raw:
        head = raw.split(START_TAG, 1)[0]
        warnings.append("Found <state_update> without a closing tag")
        return ParseResult(clean_response=head.strip(), warnings=warnings)

    blocks: list[str] = []
    rest = raw
    while True:
        start = rest.find(START_TAG)
        if start == -1:
            break
        end = rest.find(END_TAG, start)
        if end == -1:
            warnings.append("Found an unclosed <state_update> block")
            break
        blocks.append(rest[start + len(START_TAG):end].strip())
        rest = rest[:start] + rest[end + len(END_TAG):]

    if len(blocks) > 1:
        warnings.append(f"Found {len(blocks)} state blocks; only the first was parsed")

    clean = rest.strip()

    if not blocks:
        return ParseResult(clean_response=clean, warnings=warnings)

    first = blocks[0]
    if not first:
        warnings.append("State block is empty")
        return ParseResult(clean_response=clean, state_update=None, warnings=warnings)

    try:
        parsed = json.loads(first)
    except json.JSONDecodeError as exc:
        warnings.append(f"State block is not valid JSON: {exc}")
        return ParseResult(clean_response=clean, state_update=None, warnings=warnings)

    if not isinstance(parsed, dict):
        warnings.append("State block is not a JSON object")
        return ParseResult(clean_response=clean, state_update=None, warnings=warnings)

    return ParseResult(clean_response=clean, state_update=parsed, warnings=warnings)