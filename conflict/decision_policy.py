from __future__ import annotations

from typing import Any


class DecisionResult(dict):
    """Structured decision that remains backwards-compatible with string comparisons."""

    def __init__(self, action: str, reason: str, requires_user_confirmation: bool = False, details: dict[str, Any] | None = None):
        super().__init__(
            action=action,
            reason=reason,
            requires_user_confirmation=requires_user_confirmation,
            **(details or {}),
        )
        self.action = action
        self.reason = reason
        self.requires_user_confirmation = requires_user_confirmation

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.action.casefold() == other.casefold()
        return super().__eq__(other)

    def __str__(self) -> str:
        return self.action


def _nli(evidence: dict[str, Any]) -> tuple[str | None, float, float]:
    result = evidence.get("nli")
    if not result:
        return None, 0.0, 0.0
    return (
        str(result.get("label", "")).casefold() or None,
        float(result.get("confidence") or 0.0),
        float(result.get("margin") or 0.0),
    )


def decide_action(
    *,
    safety: Any = None,
    evidence: Any = None,
    relationship: str | None = None,
    **_: Any,
) -> DecisionResult:
    safety_dict = safety.to_dict() if hasattr(safety, "to_dict") else dict(safety or {})
    evidence_dict = dict(evidence or {})

    rel = relationship or evidence_dict.get("relationship") or safety_dict.get("relationship") or "unresolved"
    update = evidence_dict.get("update") or {}
    context = evidence_dict.get("context") or {}
    nli_label, nli_confidence, nli_margin = _nli(evidence_dict)

    explicit_target = bool(
        update.get("targeted_to_old_memory")
        or update.get("explicit_supersession")
    )

    replacement_supported = bool(
        update.get("replacement_supported")
        or update.get("explicit_replacement")
    )
    new_is_event = bool(context.get("new_is_event"))
    same_subject_attribute = bool(update.get("same_subject_and_attribute"))
    different_value = bool(update.get("different_value"))

    # A general memory and a later event that changes the value of the same
    # attribute is an exception candidate, not an automatic Preserve. A human
    # would normally want clarification before rewriting the standing memory.
    if new_is_event and same_subject_attribute and different_value and not explicit_target:
        return DecisionResult(
            "Ask",
            "The new event differs from an existing standing memory, but it does not explicitly say that the standing memory has changed.",
            True,
        )

    # 1. Deterministic safe outcomes.
    if rel == "duplicate":
        return DecisionResult("Ignore", "The incoming memory is already represented by the existing claim.")

    if rel == "independent":
        return DecisionResult("Preserve", "The memory concerns a different subject or attribute and does not replace the existing memory.")

    if rel in {"compatible", "paraphrase"} and nli_label != "contradiction":
        return DecisionResult("Preserve", "The memories are compatible and can coexist.")

    # 2. Explicit targeted replacement can resolve a contradiction.
    if explicit_target and replacement_supported:
        if nli_label in {None, "contradiction", "neutral", "entailment"}:
            return DecisionResult(
                "Resolve",
                "The incoming memory explicitly identifies the old memory as the claim being replaced.",
            )

    # 3. Semantic evidence.
    if nli_label == "contradiction":
        # Human-like conservative rule: a contradiction without an explicit
        # replacement is ambiguous, including event exceptions.
        if new_is_event and same_subject_attribute:
            return DecisionResult(
                "Ask",
                "The new event conflicts with an existing memory, but it does not explicitly say that the existing preference or fact has changed.",
                True,
            )
        if nli_confidence >= 0.75 and nli_margin >= 0.20 and not explicit_target:
            return DecisionResult(
                "Ask",
                "The memories contradict each other, but there is no sufficiently strong replacement signal.",
                True,
            )
        return DecisionResult("Ask", "The evidence is contradictory and the intended memory change is unclear.", True)

    if nli_label == "entailment":
        if nli_confidence >= 0.90 and nli_margin >= 0.20:
            return DecisionResult("Ignore", "The new statement is strongly entailed by an existing memory, so storing another copy is unnecessary.")
        return DecisionResult("Preserve", "The statements are semantically compatible.")

    if nli_label == "neutral":
        if rel == "independent":
            return DecisionResult("Preserve", "The statements are neutral and structurally independent.")
        if explicit_target and replacement_supported:
            return DecisionResult("Resolve", "Structured update metadata explicitly supports replacing the old memory.")
        return DecisionResult("Ask", "The statements are neither clearly equivalent nor clearly a supported replacement.", True)

    # 4. SAFE rule route from the Safety Gate.
    if safety_dict.get("safe") is True:
        hint = safety_dict.get("action_hint")
        if hint in {"Ignore", "Preserve", "Resolve"}:
            return DecisionResult(hint, safety_dict.get("reason", "Deterministic safety rule approved the action."))

    # 5. Conservative default.
    return DecisionResult("Ask", "The available evidence is insufficient for a safe automatic memory change.", True)
