from math import ceil

from churro.core.session import AgentFrame, Session, SessionState

CHARS_PER_TOKEN = 4


def _list_or_none(items: list[str]) -> str:
    if not items:
        return "None"
    return "\n".join(f"* {item}" for item in items)


def _value_or_none(value: str) -> str:
    return value if value else "None"


def _state_pairs(state: SessionState) -> tuple[tuple[str, str], ...]:
    return (
        ("Goal", _value_or_none(state.goal)),
        ("Status", state.status),
        ("Current task", _value_or_none(state.current_task)),
        ("Completed work", _list_or_none(state.completed_work)),
        ("Blockers", _list_or_none(state.blockers)),
        ("Relevant files", _list_or_none(state.relevant_files)),
        ("Next action", _value_or_none(state.next_action)),
        ("Notes", _value_or_none(state.notes)),
    )


def format_state(state: SessionState) -> str:
    """Render the state fields as a compact block (labels + values)."""
    lines: list[str] = []
    for label, value in _state_pairs(state):
        lines.append(f"{label}:")
        lines.append(value)
        lines.append("")
    return "\n".join(lines)


def build_handoff(source: Session | SessionState) -> str:
    """Convert CHURRO's persistent state into a compact, model-independent handoff.

    Pure transformation: State -> String. Never mutates the source, never reads
    conversation history, never references providers.
    """
    state = source.state if isinstance(source, Session) else source

    return (
        "You are taking over an existing coding task in CHURRO.\n\n"
        "PROJECT STATE\n\n"
        f"{format_state(state)}\n"
        "Continue from this state. Do not repeat completed work unnecessarily."
    )


def format_agent_frame(frame: AgentFrame) -> str:
    """Render an interrupted-agent frame as a compact continuation block.

    Pure transformation: AgentFrame -> String. Tells a fresh model what was
    attempted and to continue from the workspace. Never includes tool
    outputs, file contents, provider objects, or any part of the old model's
    conversation.
    """
    if frame.tool_digest:
        tools = "\n".join(f"  {line}" for line in frame.tool_digest)
    else:
        tools = "  (none)"
    return (
        "INTERRUPTED AGENT\n"
        f"Task: {frame.task}\n"
        f"Previous stop reason: {frame.stop_reason}\n"
        f"Previous iterations: {frame.iterations_used}/{frame.max_iterations}\n"
        "Tools already executed:\n"
        f"{tools}\n\n"
        "Important:\n"
        "The workspace contains the actual current state.\n"
        "Inspect the workspace before making assumptions.\n"
        "Continue the original task rather than restarting it."
    )


def character_count(text: str) -> int:
    return len(text)


def estimate_tokens(text: str) -> int:
    """Approximate token count using ~4 characters per token.

    This heuristic matches English text tokenizers (GPT, Claude) within
    roughly 15-25%. It overcounts non-English text and undercounts dense
    code. Adequate for display ("~420 tokens"), not for hard context
    enforcement or billing.
    """
    return max(1, ceil(len(text) / CHARS_PER_TOKEN))