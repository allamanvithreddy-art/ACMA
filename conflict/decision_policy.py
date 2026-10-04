from __future__ import annotations

from typing import Any
from conflict.evidence_score import calculate_update_signal


class DecisionResult(dict):
    """
    Dual string/dict representation for pipeline decisions.
    Behaves as a dict for structured access (`res['action']`) and
    compares equal to action strings (`res == 'Preserve'`).
    """
    def __init__(
        self,
        action: str,
        reason: str,
        requires_user_confirmation: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            action=action,
            reason=reason,
            requires_user_confirmation=requires_user_confirmation,
            **(details or {})
        )
        self.action = action
        self.reason = reason
        self.requires_user_confirmation = requires_user_confirmation

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.action.casefold() == other.casefold()
        if isinstance(other, dict):
            return self.get("action", "").casefold() == other.get("action", "").casefold()
        return super().__eq__(other)

    def __str__(self) -> str:
        return self.action

    def __repr__(self) -> str:
        return f"DecisionResult(action={self.action!r}, reason={self.reason!r})"


def decide_action(
    *,
    safety: Any = None,
    evidence: Any = None,
    relationship: str | None = None,
    evidence_score: Any = None,
    nli_label: str | None = None,
    nli_confidence: float | None = None,
    nli_margin: float | None = None,
    update_signal: Any = None,
    **kwargs: Any,
) -> DecisionResult:
    """
    Decide the final memory action: Ignore, Preserve, Resolve, or Ask.
    Supports both keyword dictionary input and direct parameter arguments.
    """
    # 1. Normalize parameters from either invocation format
    safety_dict = safety.to_dict() if hasattr(safety, "to_dict") else (safety if isinstance(safety, dict) else {})
    evidence_dict = evidence if isinstance(evidence, dict) else {}

    rel = (
        relationship
        or evidence_dict.get("relationship")
        or safety_dict.get("relationship")
        or "unresolved"
    )

    nli = evidence_dict.get("nli")
    if nli is not None and isinstance(nli, dict):
        label = str(nli.get("label", "")).casefold()
        conf = float(nli.get("confidence", 0.0))
        margin = float(nli.get("margin", 0.0))
    else:
        label = str(nli_label or "").casefold() if nli_label else None
        conf = float(nli_confidence or 0.0)
        margin = float(nli_margin or 0.0)

    update = evidence_dict.get("update") or {}
    upd_score = calculate_update_signal(
        update_signal if update_signal is not None else update
    )

    explicitly_targeted = bool(
        update.get("targeted_to_old_memory")
        or update.get("explicit_supersession")
        or kwargs.get("explicitly_targeted")
    )
    explicit_replacement = bool(
        update.get("explicit_replacement")
        or kwargs.get("explicit_replacement")
    )

    # 2. Rule-based / Deterministic SAFE routes
    if rel == "duplicate":
        return DecisionResult(
            action="Ignore",
            reason="The candidate is a duplicate of an existing memory.",
            requires_user_confirmation=False,
        )

    if rel == "independent":
        return DecisionResult(
            action="Preserve",
            reason="The memories are independent (different subject, attribute, or scope).",
            requires_user_confirmation=False,
        )

    if rel == "specific_event":
        return DecisionResult(
            action="Preserve",
            reason="The new memory is a specific event instance/exception, not a replacement.",
            requires_user_confirmation=False,
        )

    if rel in ("compatible", "paraphrase"):
        return DecisionResult(
            action="Preserve",
            reason="The new memory is compatible with existing memory.",
            requires_user_confirmation=False,
        )

    # 3. If NLI was not run and Safety gate was SAFE
    if label is None:
        if safety_dict.get("safe") is True:
            hint = safety_dict.get("action_hint") or "Preserve"
            return DecisionResult(
                action=hint,
                reason=safety_dict.get("reason", "Rule safety gate approved."),
                requires_user_confirmation=False,
            )
        # Not safe and no NLI -> Ask
        return DecisionResult(
            action="Ask",
            reason="Insufficient evidence for a safe deterministic decision.",
            requires_user_confirmation=True,
        )

    # 4. Semantic Decision with NLI
    # Case A: Entailment
    if label == "entailment":
        if conf >= 0.90 and margin >= 0.70 and upd_score < 0.50:
            # High entailment without update signals indicates semantic duplicate
            return DecisionResult(
                action="Ignore",
                reason="NLI reports high-confidence entailment; candidate is already represented.",
                requires_user_confirmation=False,
            )
        return DecisionResult(
            action="Preserve",
            reason="NLI reports entailment; the existing memory is confirmed/preserved.",
            requires_user_confirmation=False,
        )

    # Case B: Contradiction
    if label == "contradiction":
        # Check for update evidence (linguistic markers, explicit replacement, or temporal progression)
        has_update_evidence = (
            explicitly_targeted
            or explicit_replacement
            or upd_score >= 0.65
            or rel == "update"
        )

        if has_update_evidence:
            return DecisionResult(
                action="Resolve",
                reason=(
                    "NLI reports contradiction and update evidence supports "
                    "superseding the existing memory."
                ),
                requires_user_confirmation=False,
            )

        if rel == "constraint_violation":
            return DecisionResult(
                action="Ask",
                reason=(
                    "Evidence indicates a potential constraint violation without "
                    "established update authorization; user confirmation required."
                ),
                requires_user_confirmation=True,
            )

        # Unexplained contradiction -> genuine conflict requiring user clarification
        return DecisionResult(
            action="Ask",
            reason=(
                "NLI reports contradiction between memories without explicit "
                "update or supersession evidence."
            ),
            requires_user_confirmation=True,
        )

    # Case C: Neutral
    if label == "neutral":
        # If the claims are about different topics/attributes or non-conflicting details
        if rel in ("independent", "specific_event", "unresolved") and upd_score < 0.70:
            # Compatible / non-contradictory claims should be retained
            return DecisionResult(
                action="Preserve",
                reason="NLI reports neutral; statements are mutually compatible.",
                requires_user_confirmation=False,
            )

        if upd_score >= 0.80 and explicitly_targeted:
            return DecisionResult(
                action="Resolve",
                reason="Strong targeted update signal supports resolving the memory.",
                requires_user_confirmation=False,
            )

        return DecisionResult(
            action="Ask",
            reason="NLI evidence is neutral and ambiguity cannot be resolved automatically.",
            requires_user_confirmation=True,
        )

    # Fallback
    return DecisionResult(
        action="Ask",
        reason="Evidence is inconclusive; user confirmation is required.",
        requires_user_confirmation=True,
    )