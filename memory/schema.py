from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Memory:
    memory_id: str
    subject: str
    attribute: str
    value: str

    scope: str = "general"
    context: Optional[str] = None
    source: str = "user"
    confidence: float = 1.0

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    updated_at: Optional[str] = None
    active: bool = True

    def to_dict(self):
        return {
            "memory_id": self.memory_id,
            "subject": self.subject,
            "attribute": self.attribute,
            "value": self.value,
            "scope": self.scope,
            "context": self.context,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active": self.active,
        }
