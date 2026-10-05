from __future__ import annotations

from typing import Any

from memory.schema import Memory
from conflict.metadata import compare_metadata


class EvidenceScore(float):
    """Numeric score with structured provenance details."""

    def __new__(cls, value: float = 0.0, details: dict[str, Any] | None = None):
        obj = super().__new__(cls, float(value))
        obj.details = dict(details or {})
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        return self.details.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.details[key]

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.details)
        result["score"] = float(self)
        return result


def calculate_nli_confidence(nli: dict[str, Any] | None) -> float:
    if not nli:
        return 0.0
    return float(nli.get("confidence") or max((nli.get("scores") or {"x": 0.0}).values()))


def calculate_nli_margin(nli: dict[str, Any] | None) -> float:
    if not nli:
        return 0.0
    if nli.get("margin") is not None:
        return float(nli["margin"])
    values = sorted((nli.get("scores") or {}).values(), reverse=True)
    return float(values[0] - values[1]) if len(values) >= 2 else 0.0


def calculate_update_signal(update: Any) -> float:
    if isinstance(update, (int, float)):
        return min(max(float(update), 0.0), 1.0)
    if isinstance(update, dict):
        for key in ("signal_strength", "linguistic_signal", "score"):
            if update.get(key) is not None:
                return min(max(float(update[key]), 0.0), 1.0)
        if update.get("targeted_to_old_memory"):
            return 1.0
        if update.get("explicit_update"):
            return 0.9
    return 0.0


def calculate_metadata_agreement(
    old_memory_or_relationship: Memory | dict[str, Any],
    new_memory: Memory | None = None,
) -> float:
    if isinstance(old_memory_or_relationship, Memory) and isinstance(new_memory, Memory):
        comparison = compare_metadata(old_memory_or_relationship, new_memory)
    elif isinstance(old_memory_or_relationship, dict):
        comparison = old_memory_or_relationship.get("metadata", old_memory_or_relationship)
    else:
        return 0.0

    keys = ("same_subject", "same_attribute", "same_scope", "same_context")
    present = [bool(comparison.get(k)) for k in keys]
    return sum(present) / len(present)


def calculate_evidence_score(
    *,
    nli_confidence: float | None = None,
    nli_margin: float | None = None,
    similarity: float | None = None,
    metadata_agreement: float | None = None,
    update_signal: float | None = None,
    context_evidence: float | None = None,
    source_reliability: float | None = None,
    **details: Any,
) -> EvidenceScore:
    """Transparent evidence composite for ranking/reporting.

    This score is deliberately not the sole decision criterion. Decision Policy
    still enforces safety rules such as contradiction-without-update => Ask.
    """
    components = {
        "nli_confidence": min(max(float(nli_confidence or 0.0), 0.0), 1.0),
        "nli_margin": min(max(float(nli_margin or 0.0), 0.0), 1.0),
        "retrieval": min(max(float(similarity or 0.0), 0.0), 1.0),
        "metadata": min(max(float(metadata_agreement or 0.0), 0.0), 1.0),
        "update": min(max(float(update_signal or 0.0), 0.0), 1.0),
        "context": min(max(float(context_evidence or 0.0), 0.0), 1.0),
        "source": min(max(float(source_reliability if source_reliability is not None else 1.0), 0.0), 1.0),
    }

    score = (
        0.30 * components["nli_confidence"]
        + 0.15 * components["nli_margin"]
        + 0.15 * components["retrieval"]
        + 0.15 * components["metadata"]
        + 0.15 * components["update"]
        + 0.05 * components["context"]
        + 0.05 * components["source"]
    )

    payload = {
        **components,
        "details": details,
    }
    return EvidenceScore(min(max(score, 0.0), 1.0), payload)


def analyze_evidence(
    *,
    safety: dict[str, Any],
    relationship: dict[str, Any],
    nli: dict[str, Any] | None,
) -> dict[str, Any]:
    update = relationship.get("update", {})
    metadata = relationship.get("metadata", {})
    context = relationship.get("context", {})
    signals = safety.get("signals", {})

    nli_confidence = calculate_nli_confidence(nli)
    nli_margin = calculate_nli_margin(nli)
    retrieval = signals.get("retrieval_score")
    metadata_score = calculate_metadata_agreement(metadata)
    update_score = calculate_update_signal(update)

    if context.get("context_relation") == "same":
        context_evidence = 1.0
    elif context.get("context_relation") == "different":
        context_evidence = 0.5
    else:
        context_evidence = 0.0

    source_reliability = 1.0
    details = calculate_evidence_score(
        nli_confidence=nli_confidence,
        nli_margin=nli_margin,
        similarity=retrieval,
        metadata_agreement=metadata_score,
        update_signal=update_score,
        context_evidence=context_evidence,
        source_reliability=source_reliability,
        relationship=relationship.get("relationship"),
        safety_route=safety.get("next_stage"),
    )

    return {
        "score": float(details),
        "evidence_score": float(details),
        "nli_confidence": nli_confidence,
        "nli_margin": nli_margin,
        "retrieval_score": retrieval,
        "metadata_agreement": metadata_score,
        "update_signal": update_score,
        "context_evidence": context_evidence,
        "source_reliability": source_reliability,
        "relationship": relationship.get("relationship"),
        "context": context,
        "update": update,
        "nli": nli,
        "metadata": metadata,
        "structural": relationship.get("structural", {}),
        "safety_route": safety.get("next_stage"),
        "evidence_provenance": {
            "relationship_source": relationship.get("structural", {}).get("extraction_source"),
            "update_source": update.get("evidence_source"),
            "nli_model": nli.get("model") if nli else None,
        },
    }
