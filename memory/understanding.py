from __future__ import annotations

from typing import Any

from memory.schema import Memory


class MemoryUnderstanding:
    REQUIRED_FIELDS = ("subject", "attribute", "value")

    def from_structured_event(
        self,
        event: dict[str, Any],
        *,
        source: str = "user",
    ) -> Memory:
        if not isinstance(event, dict):
            raise TypeError(
                "Expected a structured memory event, not raw text."
            )

        missing = [
            field for field in self.REQUIRED_FIELDS
            if field not in event
        ]
        if missing:
            raise ValueError(
                f"Structured memory event is missing required fields: {missing}"
            )

        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("event.metadata must be an object")

        return Memory(
            memory_id=str(event["memory_id"]) if event.get("memory_id") else
                __import__("uuid").uuid4().hex,
            text=str(event.get("text") or event["value"]),
            subject=str(event["subject"]),
            attribute=str(event["attribute"]),
            value=str(event["value"]),
            scope=str(event.get("scope") or ""),
            context=str(event.get("context") or ""),
            time=event.get("time"),
            source=str(event.get("source") or source),
            confidence=event.get("confidence"),
            importance=event.get("importance"),
            metadata=dict(metadata),
        )