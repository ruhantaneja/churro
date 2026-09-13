from dataclasses import dataclass, field
from typing import Any

VALID_STATUSES = {"not_started", "in_progress", "blocked", "done"}

ALLOWED_FIELDS = frozenset(
    {"goal", "current_task", "completed_work", "status",
     "blockers", "relevant_files", "next_action", "notes"}
)
LIST_FIELDS = frozenset({"completed_work", "blockers", "relevant_files"})
STRING_FIELDS = frozenset({"goal", "current_task", "next_action", "notes"})
NON_EMPTY_STRING_FIELDS = frozenset({"goal", "current_task", "next_action"})


@dataclass
class ValidationResult:
    valid: dict[str, Any] = field(default_factory=dict)
    invalid: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return bool(self.valid)


def validate_state_update(update: Any) -> ValidationResult:
    """Validate a proposed state update field by field.

    The validator is side-effect free: it reports what MAY be merged,
    it never merges anything itself.
    """
    result = ValidationResult()

    if not isinstance(update, dict):
        result.warnings.append("State update is not an object; nothing accepted")
        return result

    if not update:
        result.warnings.append("State update is empty; nothing to validate")
        return result

    for key, value in update.items():
        if key not in ALLOWED_FIELDS:
            result.invalid[key] = "Unknown field; not allowed in persistent state"
            continue

        if key in LIST_FIELDS:
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                result.valid[key] = value
            else:
                result.invalid[key] = "Must be a list of strings"

        elif key in STRING_FIELDS:
            if not isinstance(value, str):
                result.invalid[key] = "Must be a string"
            elif key in NON_EMPTY_STRING_FIELDS and not value.strip():
                result.invalid[key] = "Must be a non-empty string"
            else:
                result.valid[key] = value

        else:
            if value in VALID_STATUSES:
                result.valid[key] = value
            else:
                result.invalid[key] = (
                    f"Must be one of: {', '.join(sorted(VALID_STATUSES))}"
                )

    if not result.valid:
        result.warnings.append("No fields passed validation; state was not updated")

    return result