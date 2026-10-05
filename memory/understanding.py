from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from memory.schema import Memory


class MemoryUnderstanding:
    """Validate/canonicalize structured memory emitted by the frontend LLM.

    This class deliberately does not attempt semantic extraction from raw text.
    The upstream LLM is responsible for producing the structured memory fields.
    """

    REQUIRED_FIELDS = ("subject", "attribute", "value")

    @staticmethod
    def _clean(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    def from_structured_event(self, event: dict[str, Any], *, source: str = "user", user_id: str = "default_user") -> Memory:
        if not isinstance(event, dict):
            raise TypeError("Expected a structured memory event object.")

        missing = [field for field in self.REQUIRED_FIELDS if not self._clean(event.get(field))]
        if missing:
            raise ValueError(f"Structured memory event is missing required fields: {missing}")

        metadata = event.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise TypeError("event.metadata must be an object")

        confidence = event.get("confidence")
        importance = event.get("importance")
        for name, value in (("confidence", confidence), ("importance", importance)):
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        return Memory(
            memory_id=self._clean(event.get("memory_id")) or uuid4().hex,
            text=self._clean(event.get("text") or event["value"]),
            subject=self._clean(event["subject"]),
            attribute=self._clean(event["attribute"]),
            value=self._clean(event["value"]),
            scope=self._clean(event.get("scope")),
            context=self._clean(event.get("context")),
            time=event.get("time"),
            source=self._clean(event.get("source") or source),
            confidence=float(confidence) if confidence is not None else None,
            importance=float(importance) if importance is not None else None,
            metadata=dict(metadata),
            user_id=self._clean(event.get("user_id") or user_id),
        )
