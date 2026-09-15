from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionState(BaseModel):
    goal: str = ""
    current_task: str = ""
    completed_work: list[str] = Field(default_factory=list)
    status: Literal["not_started", "in_progress", "blocked", "done"] = "not_started"
    blockers: list[str] = Field(default_factory=list)
    relevant_files: list[str] = Field(default_factory=list)
    next_action: str = ""
    notes: str = ""


class ArchivedConversation(BaseModel):
    provider: str
    model: str
    archived_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[ConversationMessage] = Field(default_factory=list)


# Upper bounds that keep the compact continuation digest from becoming
# a de-facto transcript. The workspace is the source of truth for actual
# work; the digest only names what was dispatched.
_DIGEST_MAX_LINES = 200
_DIGEST_MAX_LINE_CHARS = 200

_DigestLine = Annotated[str, Field(min_length=1, max_length=_DIGEST_MAX_LINE_CHARS)]


class AgentFrame(BaseModel):
    """Compact metadata about an interrupted autonomous run.

    This is not a checkpoint of the model's conversation. It records only
    enough provider-neutral context for a later continuation to describe
    what already happened and keep going: the task, how much of the budget
    was consumed, why the run stopped, and one-line summaries of the tools
    that were executed. Full tool outputs, conversation history, and any
    provider-specific state deliberately never enter this frame.
    """

    task: str = Field(min_length=1)
    iterations_used: int = Field(ge=0)
    max_iterations: int = Field(ge=1)
    stop_reason: str = Field(min_length=1)
    tool_digest: list[_DigestLine] = Field(
        default_factory=list, max_length=_DIGEST_MAX_LINES
    )
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _check_iterations_within_budget(self) -> "AgentFrame":
        if self.iterations_used > self.max_iterations:
            raise ValueError("iterations_used must not exceed max_iterations")
        return self


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active_provider: str = ""
    active_model: str = ""
    state: SessionState = Field(default_factory=SessionState)
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    archived_conversations: list[ArchivedConversation] = Field(default_factory=list)
    pending_agent: AgentFrame | None = None


def create_session(goal: str, provider: str, model: str) -> Session:
    return Session(
        active_provider=provider,
        active_model=model,
        state=SessionState(goal=goal),
    )


def save_session(session: Session, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(session.model_dump_json(indent=2), encoding="utf-8")


def load_session(path: str) -> Session:
    raw = Path(path).read_text(encoding="utf-8")
    return Session.model_validate_json(raw)