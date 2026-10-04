from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from memory.schema import Memory
from conflict.scope_rules import classify_relationship
from conflict.metadata import normalize_field


@dataclass
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
        d = asdict(self)
        d["action"] = self.action_hint
        return d

    def __getitem__(self, key: str) -> Any:
        if key == "action":
            return self.action_hint
        return getattr(self, key)


def evaluate_rule_safety(
    old_memory: Memory,
    new_memory: Memory,
    similarity: float | None = None,
) -> SafetyResult:
    relationship_info = classify_relationship(old_memory, new_memory)
    signals = dict(relationship_info)
    signals["retrieval_score"] = similarity

    rel = relationship_info.get("relationship", "unresolved")

    # 1. Duplicate
    if rel == "duplicate":
        return SafetyResult(
            safe=True,
            next_stage="decision",
            relationship="duplicate",
            action_hint="Ignore",
            reason="The claims identify the same stored information.",
            signals=signals,
        )

    # 2. Independent facts (different subjects, attributes, or independent contexts)
    if rel == "independent":
        return SafetyResult(
            safe=True,
            next_stage="decision",
            relationship="independent",
            action_hint="Preserve",
            reason="The claims are independent (different subject, attribute, or scope).",
            signals=signals,
        )

    # 3. Specific event exceptions to general preferences
    if rel == "specific_event":
        return SafetyResult(
            safe=True,
            next_stage="decision",
            relationship="specific_event",
            action_hint="Preserve",
            reason="Specific event instance does not supersede general preference or constraint.",
            signals=signals,
        )

    # 4. Compatible / Paraphrase
    if rel in ("compatible", "paraphrase"):
        return SafetyResult(
            safe=True,
            next_stage="decision",
            relationship=rel,
            action_hint="Preserve",
            reason="The claims are mutually compatible and do not conflict.",
            signals=signals,
        )

    # 5. Potential conflict, update, constraint violation, or unresolved -> route to NLI
    return SafetyResult(
        safe=False,
        next_stage="nli",
        relationship=rel,
        action_hint=None,
        reason=(
            "The candidate involves potential conflict, update, or uncertainty; "
            "semantic evidence verification is required."
        ),
        signals=signals,
    )