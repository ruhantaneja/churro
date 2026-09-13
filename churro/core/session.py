from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


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


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active_provider: str = ""
    active_model: str = ""
    state: SessionState = Field(default_factory=SessionState)
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    archived_conversations: list[ArchivedConversation] = Field(default_factory=list)


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