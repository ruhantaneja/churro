from dataclasses import dataclass, field
from typing import Any

from churro.core.parser import parse_response
from churro.core.session import ConversationMessage, Session, SessionState
from churro.core.validator import validate_state_update

_ACCUMULATING_LISTS = frozenset({"completed_work"})
_REPLACING_LISTS = frozenset({"blockers", "relevant_files"})


@dataclass
class ProcessResult:
    clean_response: str = ""
    state_update: dict[str, Any] | None = None
    merged_fields: dict[str, Any] = field(default_factory=dict)
    rejected_fields: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _merge_state(state: SessionState, valid: dict[str, Any]) -> dict[str, Any]:
    """Merge only pre-validated fields into the session state.

    This is the single place SessionState may be mutated by an AI response.
    """
    merged: dict[str, Any] = {}

    for key, value in valid.items():
        if key in _ACCUMULATING_LISTS:
            combined = list(state.completed_work)
            for item in value:
                if item not in combined:
                    combined.append(item)
            setattr(state, key, combined)
        elif key in _REPLACING_LISTS:
            setattr(state, key, list(value))
        else:
            setattr(state, key, value)

        merged[key] = getattr(state, key)

    return merged


def process_ai_response(session: Session, raw_response: str) -> ProcessResult:
    """Turn a raw AI response into a persistent, validated state change.

    Flow: parse -> validate -> merge only valid fields -> append clean text.
    """
    parsed = parse_response(raw_response)

    result = ProcessResult(
        clean_response=parsed.clean_response,
        state_update=parsed.state_update,
        warnings=list(parsed.warnings),
    )

    if parsed.state_update is not None:
        validated = validate_state_update(parsed.state_update)
        result.rejected_fields = dict(validated.invalid)
        result.warnings.extend(validated.warnings)
        if validated.valid:
            result.merged_fields = _merge_state(session.state, validated.valid)

    session.conversation_history.append(
        ConversationMessage(role="assistant", content=parsed.clean_response)
    )

    return result