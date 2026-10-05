from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Memory:
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = "default_user"
    text: str = ""
    subject: str = ""
    attribute: str = ""
    value: str = ""
    scope: str = ""
    context: str = ""
    time: str | None = None
    source: str = "user"
    confidence: float | None = None
    importance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    version: int = 1
    supersedes: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    relationship_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_decision: str | None = None
    last_decision_reason: str | None = None
    last_decision_confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("memory_id", "user_id", "text", "subject", "attribute", "value", "scope", "context", "source"):
            setattr(self, name, " ".join(str(getattr(self, name) or "").strip().split()))
        if not self.user_id:
            self.user_id = "default_user"
        if not isinstance(self.metadata, dict):
            raise TypeError("Memory.metadata must be a dictionary")
        for name in ("confidence", "importance"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must be between 0 and 1")
                setattr(self, name, value)
        if self.version < 1:
            raise ValueError("Memory.version must be >= 1")
        if self.status not in {"active", "pending", "superseded", "archived"}:
            raise ValueError(f"Unsupported memory status: {self.status}")

    @property
    def active(self) -> bool:
        return self.status == "active"

    def update_timestamp(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        source = dict(data)
        if "status" not in source:
            source["status"] = "active" if source.pop("active", True) else "superseded"
        source.setdefault("metadata", {})
        source.setdefault("text", str(source.get("value", "")))
        source.setdefault("user_id", "default_user")
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in source.items() if k in allowed})

    def __repr__(self) -> str:
        return f"Memory(id={self.memory_id!r}, attribute={self.attribute!r}, scope={self.scope!r}, status={self.status!r})"


@dataclass
class MemoryQuery:
    text: str = ""
    subject: str = ""
    attribute: str = ""
    value: str = ""
    scope: str = ""
    context: str = ""
    time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("text", "subject", "attribute", "value", "scope", "context"):
            setattr(self, name, " ".join(str(getattr(self, name) or "").strip().split()))
        if not isinstance(self.metadata, dict):
            raise TypeError("MemoryQuery.metadata must be a dictionary")

    @classmethod
    def from_memory(cls, memory: Memory) -> "MemoryQuery":
        return cls(
            text=memory.text,
            subject=memory.subject,
            attribute=memory.attribute,
            value=memory.value,
            scope=memory.scope,
            context=memory.context,
            time=memory.time,
            metadata=dict(memory.metadata),
        )
