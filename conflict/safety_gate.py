from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from memory.schema import Memory
from conflict.relationship import classify_relationship


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    next_stage: str
    relationship: str
    action_hint: str | None
    reason: str
    signals: dict[str, Any]

    @property
    def action(self) -> str | None:
        return self.action_hint

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action_hint
        return data

    def __getitem__(self, key: str) -> Any:
        if key == "action":
            return self.action_hint
        return getattr(self, key)


def evaluate_rule_safety(
    old_memory: Memory,
    new_memory: Memory,
    similarity: float | None = None,
    relationship_info: dict[str, Any] | None = None,
) -> SafetyResult:
    """Conservative gate: SAFE means deterministic rules are strong enough.

    A specific event does NOT automatically become safe merely because it has
    a different context. A same-subject/same-attribute value change remains
    potentially conflicting and is sent to NLI unless an explicit targeted
    replacement is already present.
    """
    relationship_info = relationship_info or classify_relationship(old_memory, new_memory)
    relationship = relationship_info["relationship"]
    update = relationship_info["update"]
    context = relationship_info["context"]
    signals = {
        "retrieval_score": similarity,
        "metadata": relationship_info.get("metadata", {}),
        "context": context,
        "update": update,
        "structural": relationship_info.get("structural", {}),
    }

    if relationship == "duplicate":
        return SafetyResult(
            safe=True,
            next_stage="decision",
            relationship=relationship,
            action_hint="Ignore",
            reason="The new claim is structurally identical to an existing memory.",
            signals=signals,
        )

    if relationship == "independent":
        return SafetyResult(
            safe=True,
            next_stage="decision",
            relationship=relationship,
            action_hint="Preserve",
            reason="The claims concern different subjects or attributes and no explicit link indicates replacement.",
            signals=signals,
        )

    if relationship in {"compatible", "paraphrase"}:
        return SafetyResult(
            safe=True,
            next_stage="decision",
            relationship=relationship,
            action_hint="Preserve",
            reason="The structured evidence indicates a compatible claim.",
            signals=signals,
        )

    # An explicit, targeted replacement is strong deterministic evidence.
    if relationship == "update" and update.get("replacement_supported"):
        return SafetyResult(
            safe=True,
            next_stage="decision",
            relationship=relationship,
            action_hint="Resolve",
            reason="The incoming memory explicitly targets the old memory as a replacement.",
            signals=signals,
        )

    # Same claim with different value, event/general differences, constraint
    # violations, or unknown structure are not safe to resolve by rules alone.
    return SafetyResult(
        safe=False,
        next_stage="nli",
        relationship=relationship,
        action_hint=None,
        reason="The available deterministic evidence is insufficient to safely accept, replace, or ignore the old memory.",
        signals=signals,
    )
