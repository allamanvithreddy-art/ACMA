from __future__ import annotations

import re
from typing import Any, Union

from memory.schema import Memory


# Explicit language that normally indicates a state replacement.
STRONG_UPDATE_PATTERNS: dict[str, float] = {
    r"\bswitched\s+from\b": 1.00,
    r"\bswitched\s+to\b": 0.98,
    r"\bchanged\s+from\b": 1.00,
    r"\bchanged\s+to\b": 0.98,
    r"\breplaced\b": 1.00,
    r"\breplace\b": 0.95,
    r"\bmigrated\s+from\b": 0.98,
    r"\bmigrated\s+to\b": 0.98,
    r"\bmoved\s+from\b": 0.95,
    r"\bmoved\s+to\b": 0.95,
    r"\bshifted\s+from\b": 0.95,
    r"\bshifted\s+to\b": 0.95,
    r"\btransitioned\s+from\b": 0.95,
    r"\btransitioned\s+to\b": 0.95,
    r"\bconverted\s+from\b": 0.95,
    r"\bconverted\s+to\b": 0.95,

    r"\bno longer\b": 1.00,
    r"\bnot anymore\b": 0.98,
    r"\bnot any more\b": 0.98,
    r"\bstopped\s+using\b": 0.98,
    r"\bstopped\s+doing\b": 0.90,
    r"\bquit\s+using\b": 0.96,
    r"\bdiscontinued\b": 0.94,

    r"\bnow\s+use\b": 0.96,
    r"\bnow\s+prefer\b": 0.96,
    r"\bnow\s+want\b": 0.92,
    r"\bcurrently\s+use\b": 0.88,
    r"\bcurrently\s+prefer\b": 0.88,

    r"\bchanged\s+my\s+mind\b": 0.95,
    r"\bi\s+now\s+prefer\b": 0.96,
    r"\bi\s+no\s+longer\s+prefer\b": 1.00,

    r"\bfrom\s+now\s+on\b": 0.96,
    r"\bgoing\s+forward\b": 0.96,
    r"\bhenceforth\b": 0.94,
    r"\beffective\s+(?:today|now|immediately)\b": 0.94,

    r"\binstead\s+of\b": 0.94,
    r"\brather\s+than\b": 0.90,
    r"\bin\s+place\s+of\b": 0.92,
}


WEAK_UPDATE_PATTERNS: dict[str, float] = {
    r"\bnow\b": 0.60,
    r"\bcurrently\b": 0.58,
    r"\bpresently\b": 0.55,
    r"\brecently\b": 0.52,
    r"\blately\b": 0.52,
    r"\bthese\s+days\b": 0.55,
    r"\bchanged\b": 0.60,
    r"\bswitched\b": 0.65,
    r"\bmoved\b": 0.60,
    r"\bstarted\b": 0.55,
    r"\bbegan\b": 0.55,
    r"\bused\s+to\b": 0.65,
    r"\bpreviously\b": 0.60,
    r"\bformerly\b": 0.65,
    r"\bearlier\b": 0.55,
}


TEMPORARY_PATTERNS: set[str] = {
    r"\btemporarily\b",
    r"\bfor\s+now\b",
    r"\bfor\s+today\b",
    r"\bfor\s+tonight\b",
    r"\bfor\s+tomorrow\b",
    r"\bthis\s+time\b",
    r"\bthis\s+week\b",
    r"\bthis\s+month\b",
    r"\bduring\s+this\s+period\b",
    r"\bin\s+this\s+context\b",
    r"\bonly\s+for\b",
    r"\bjust\s+for\b",
}


NEGATION_PATTERNS: set[str] = {
    r"\bnot\b",
    r"\bnever\b",
    r"\bno\s+longer\b",
    r"\bwithout\b",
    r"\bdo\s+not\b",
    r"\bdon't\b",
    r"\bdoes\s+not\b",
    r"\bdoesn't\b",
    r"\bdid\s+not\b",
    r"\bdidn't\b",
    r"\bwon't\b",
    r"\bwouldn't\b",
}


TRANSITION_PATTERNS: dict[str, float] = {
    r"\bfrom\s+.+?\s+to\s+.+": 1.00,
    r"\bfrom\s+.+?\s+into\s+.+": 0.96,
    r"\bfrom\s+.+?\s+toward\s+.+": 0.90,
    r"\b.+?\s+instead\s+of\s+.+": 0.94,
    r"\b.+?\s+rather\s+than\s+.+": 0.90,
    r"\b.+?\s+in\s+place\s+of\s+.+": 0.92,
}


def _extract_text(
    statement_or_memory: Union[str, Memory, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if isinstance(statement_or_memory, Memory):
        return (
            str(statement_or_memory.text or statement_or_memory.value or ""),
            dict(statement_or_memory.metadata or {}),
        )

    if isinstance(statement_or_memory, dict):
        return (
            str(
                statement_or_memory.get("text")
                or statement_or_memory.get("value")
                or ""
            ),
            dict(statement_or_memory.get("metadata") or {}),
        )

    return str(statement_or_memory or ""), {}


def _find_patterns(
    text: str,
    patterns: dict[str, float],
) -> tuple[list[str], float]:
    matches: list[str] = []
    maximum = 0.0

    for pattern, weight in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)
            maximum = max(maximum, weight)

    return matches, maximum


def detect_update_signals(
    statement_or_memory: Union[str, Memory, dict[str, Any]],
) -> dict[str, Any]:
    """
    Detect linguistic evidence that a statement represents a change.

    Important:
    This function does not trust dataset operation labels or benchmark
    annotations. It only analyzes the incoming statement itself.
    """

    text, _ = _extract_text(statement_or_memory)
    normalized = " ".join(text.casefold().split())

    strong_markers, strong_score = _find_patterns(
        normalized,
        STRONG_UPDATE_PATTERNS,
    )

    weak_markers, weak_score = _find_patterns(
        normalized,
        WEAK_UPDATE_PATTERNS,
    )

    temporary_markers, _ = _find_patterns(
        normalized,
        {pattern: 1.0 for pattern in TEMPORARY_PATTERNS},
    )

    negation_markers, _ = _find_patterns(
        normalized,
        {pattern: 1.0 for pattern in NEGATION_PATTERNS},
    )

    transition_markers, transition_score = _find_patterns(
        normalized,
        TRANSITION_PATTERNS,
    )

    explicit_replacement_signal = (
        transition_score >= 0.90
        or strong_score >= 0.94
    )

    temporary_context = bool(temporary_markers)

    likely_temporary_change = (
        temporary_context
        and not explicit_replacement_signal
    )

    score = max(
        strong_score,
        transition_score,
        weak_score,
    )

    has_signal = (
        strong_score >= 0.70
        or transition_score >= 0.80
        or (
            weak_score >= 0.55
            and not likely_temporary_change
        )
    )

    markers = [
        *(f"strong:{marker}" for marker in strong_markers),
        *(f"weak:{marker}" for marker in weak_markers),
        *(f"transition:{marker}" for marker in transition_markers),
    ]

    return {
        "has_update_signal": has_signal,
        "signal_strength": score,
        "score": score,
        "markers": markers,
        "update_detected": has_signal,
        "explicit_replacement_signal": explicit_replacement_signal,
        "transition_detected": transition_score >= 0.80,
        "transition_strength": transition_score,
        "temporary_context": temporary_context,
        "temporary_markers": temporary_markers,
        "likely_temporary_change": likely_temporary_change,
        "negation_detected": bool(negation_markers),
        "negation_markers": negation_markers,
        "text": text,
    }


def analyze_update(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:
    """
    Compare two memories and determine whether the new memory contains
    evidence of replacing the old one.

    Only genuine memory lineage identifiers are accepted as structured
    targeting signals. Dataset operation labels are intentionally ignored.
    """

    new_meta = new_memory.metadata or {}

    update_of = new_meta.get("update_of")

    supersedes = new_meta.get("supersedes") or []
    if isinstance(supersedes, str):
        supersedes = [supersedes]

    explicit_supersession = (
        old_memory.memory_id in supersedes
        or update_of == old_memory.memory_id
    )

    signals = detect_update_signals(new_memory)

    temporal_progression = False

    if old_memory.time and new_memory.time:
        temporal_progression = str(new_memory.time) > str(old_memory.time)

    same_subject = (
        bool(old_memory.subject)
        and bool(new_memory.subject)
        and old_memory.subject.casefold().strip()
        == new_memory.subject.casefold().strip()
    )

    same_attribute = (
        bool(old_memory.attribute)
        and bool(new_memory.attribute)
        and old_memory.attribute.casefold().strip()
        == new_memory.attribute.casefold().strip()
    )

    same_subject_and_attribute = (
        same_subject and same_attribute
    )

    different_value = (
        old_memory.value.casefold().strip()
        != new_memory.value.casefold().strip()
    )

    strong_replacement_signal = (
        signals["explicit_replacement_signal"]
        or signals["transition_detected"]
    )

    targeted = (
        explicit_supersession
        or (
            same_subject_and_attribute
            and different_value
            and strong_replacement_signal
            and not signals["likely_temporary_change"]
        )
    )

    replacement_supported = (
        targeted
        and different_value
        and not signals["likely_temporary_change"]
    )

    return {
        "explicit_supersession": explicit_supersession,
        "targeted_to_old_memory": targeted,
        "update_of": update_of,
        "supersedes": list(supersedes),
        "linguistic_signal": signals["signal_strength"],
        "linguistic_markers": signals["markers"],
        "transition_detected": signals["transition_detected"],
        "temporary_context": signals["temporary_context"],
        "likely_temporary_change": signals["likely_temporary_change"],
        "negation_detected": signals["negation_detected"],
        "temporal_progression": temporal_progression,
        "same_subject_and_attribute": same_subject_and_attribute,
        "same_subject": same_subject,
        "same_attribute": same_attribute,
        "different_value": different_value,
        "replacement_supported": replacement_supported,
        "evidence_source": (
            "memory_lineage"
            if explicit_supersession
            else "incoming_text"
        ),
        "evidence_confidence": (
            1.0
            if explicit_supersession
            else signals["signal_strength"]
        ),
    }


def detect_update(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:
    return analyze_update(old_memory, new_memory)
