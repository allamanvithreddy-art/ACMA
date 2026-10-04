from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Memory:
    memory_id: str = field(default_factory=lambda: str(uuid4()))
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

    # Extraction output. These fields describe evidence, not domain vocabulary.
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
        self.memory_id = str(self.memory_id)
        self.text = str(self.text or "").strip()
        self.subject = str(self.subject or "").strip()
        self.attribute = str(self.attribute or "").strip()
        self.value = str(self.value or self.text).strip()
        self.scope = str(self.scope or "").strip()
        self.context = str(self.context or "").strip()
        self.source = str(self.source or "user").strip()

        if not isinstance(self.metadata, dict):
            raise TypeError("Memory.metadata must be a dictionary")

        for name in ("confidence", "importance"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            if value is not None:
                setattr(self, name, float(value))

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
        data = dict(data)

        # Migrate the older representation without silently discarding fields.
        if "status" not in data:
            data["status"] = (
                "active" if data.pop("active", True) else "superseded"
            )

        if "metadata" not in data:
            data["metadata"] = {}

        if "text" not in data:
            data["text"] = str(data.get("value", ""))

        return cls(**{
            key: value
            for key, value in data.items()
            if key in cls.__dataclass_fields__
        })

    def __repr__(self) -> str:
        return (
            f"Memory(id={self.memory_id!r}, "
            f"attribute={self.attribute!r}, "
            f"scope={self.scope!r}, "
            f"status={self.status!r})"
        )


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
        for name in (
            "text", "subject", "attribute", "value",
            "scope", "context",
        ):
            setattr(self, name, str(getattr(self, name) or "").strip())

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

    def to_memory(self, memory_id: str | None = None) -> Memory:
        return Memory(
            memory_id=memory_id or str(uuid4()),
            text=self.text or self.value,
            subject=self.subject,
            attribute=self.attribute,
            value=self.value or self.text,
            scope=self.scope,
            context=self.context,
            time=self.time,
            metadata=dict(self.metadata),
        )