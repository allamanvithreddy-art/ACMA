from __future__ import annotations

from typing import Any
from memory.schema import Memory
from conflict.metadata import compare_metadata


class EvidenceScore(float):
    """
    Numeric evidence composite that also carries structured evidence details,
    allowing both float formatting (e.g. f"{score:.4f}") and dict lookups.
    """
    def __new__(cls, value: float = 0.0, details: dict[str, Any] | None = None):
        instance = super().__new__(cls, float(value))
        instance.details = dict(details or {})
        return instance

    def get(self, key: str, default: Any = None) -> Any:
        return self.details.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.details[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.details

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.details)
        d["score"] = float(self)
        return d


def analyze_evidence(
    *,
    safety: dict[str, Any],
    relationship: dict[str, Any],
    nli: dict[str, Any] | None,
) -> dict[str, Any]:
    signals = safety.get("signals", {})
    update = relationship.get("update", {})

    return {
        "safety_route": safety.get("next_stage"),
        "relationship": (
            safety.get("relationship")
            or relationship.get("relationship")
        ),
        "metadata": relationship.get("metadata", {}),
        "structural": relationship.get("structural", {}),
        "update": update,
        "nli": nli,
        "nli_was_run": nli is not None,
        "retrieval_score": signals.get("retrieval_score"),
        "evidence_provenance": {
            "relationship_source": relationship.get(
                "structural", {}
            ).get("extraction_source"),
            "update_source": update.get("evidence_source"),
            "nli_model": nli.get("model") if nli else None,
        },
    }


def calculate_nli_confidence(nli_result: dict[str, Any] | None) -> float | None:
    if not nli_result:
        return None
    value = nli_result.get("confidence")
    if value is not None:
        return float(value)
    scores = nli_result.get("scores", {})
    if scores:
        return float(max(scores.values()))
    return None


def calculate_nli_margin(nli_result: dict[str, Any] | None) -> float | None:
    if not nli_result:
        return None
    value = nli_result.get("margin")
    if value is not None:
        return float(value)
    scores = nli_result.get("scores", {})
    if len(scores) >= 2:
        ranked = sorted(scores.values(), reverse=True)
        return float(ranked[0] - ranked[1])
    return 0.0


def calculate_update_signal(
    signal_or_result: Any,
) -> float:
    """
    Extract a numeric update signal (0.0 to 1.0) from detection results.
    """
    if isinstance(signal_or_result, (int, float)):
        return float(signal_or_result)
    if isinstance(signal_or_result, dict):
        if "signal_strength" in signal_or_result:
            return float(signal_or_result["signal_strength"])
        if "score" in signal_or_result:
            return float(signal_or_result["score"])
        if "linguistic_signal" in signal_or_result:
            return float(signal_or_result["linguistic_signal"])
        if "update" in signal_or_result:
            sub = signal_or_result["update"]
            if isinstance(sub, dict):
                return calculate_update_signal(sub)
        if signal_or_result.get("targeted_to_old_memory"):
            return 1.0
        if signal_or_result.get("explicit_update"):
            return 0.9
    return 0.0


def calculate_metadata_agreement(
    first_arg: Any,
    second_arg: Any = None,
) -> float:
    """
    Returns numeric agreement score (0.0 to 1.0) between memories.
    Supports either (old_memory, new_memory) or (relationship_dict).
    """
    if second_arg is not None and isinstance(first_arg, Memory) and isinstance(second_arg, Memory):
        comp = compare_metadata(first_arg, second_arg)
        matches = sum([
            1.0 if comp.get("same_subject") else 0.0,
            1.0 if comp.get("same_attribute") else 0.0,
            1.0 if comp.get("same_scope") else 0.0,
            1.0 if comp.get("same_context") else 0.0,
        ])
        return matches / 4.0

    if isinstance(first_arg, dict):
        meta = first_arg.get("metadata", first_arg)
        matches = sum([
            1.0 if meta.get("same_subject") else 0.0,
            1.0 if meta.get("same_attribute") else 0.0,
            1.0 if meta.get("same_scope") else 0.0,
            1.0 if meta.get("same_context") else 0.0,
        ])
        return matches / 4.0

    return 0.5


def calculate_evidence_score(
    nli_confidence: float | None = None,
    nli_margin: float | None = None,
    similarity: float | None = None,
    metadata_agreement: Any = None,
    update_signal: Any = None,
    source_reliability: float | None = None,
    **kwargs: Any,
) -> EvidenceScore:
    """
    Compute structured evidence and a composite evidence score.
    Returns an EvidenceScore (float subclass) for dual float/dict compatibility.
    """
    conf = float(nli_confidence or 0.0)
    sim = float(similarity or 0.0)
    upd = calculate_update_signal(update_signal)
    meta = calculate_metadata_agreement(metadata_agreement)
    rel = float(source_reliability or 1.0)

    # Composite scalar: weighted combination of signals
    composite = (0.35 * conf) + (0.25 * upd) + (0.20 * sim) + (0.10 * meta) + (0.10 * rel)
    composite = min(max(composite, 0.0), 1.0)

    details = {
        "nli_confidence": nli_confidence,
        "nli_margin": nli_margin,
        "retrieval_score": similarity,
        "metadata_evidence": metadata_agreement,
        "update_evidence": update_signal,
        "source_reliability": source_reliability,
        "numeric_update_signal": upd,
        "numeric_metadata_agreement": meta,
        "additional_evidence": kwargs,
        "score": composite,
    }

    return EvidenceScore(composite, details)