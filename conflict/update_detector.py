from __future__ import annotations

import re
from typing import Any, Union
from memory.schema import Memory


UPDATE_KEYWORDS = {
    # Direct replacement markers
    "now": 0.85,
    "currently": 0.70,
    "no longer": 0.95,
    "not anymore": 0.90,
    "switched": 0.95,
    "switched to": 0.95,
    "switched from": 0.95,
    "migrated": 0.90,
    "migrated to": 0.90,
    "moved": 0.85,
    "moved to": 0.90,
    "changed": 0.80,
    "changed to": 0.85,
    "instead of": 0.85,
    "rather than": 0.80,
    "replaced": 0.90,
    "stopped": 0.85,
    "started": 0.75,
    "began": 0.75,
    "prefer": 0.60,
    "previously": 0.75,
    "earlier": 0.65,
    "former": 0.70,
    "formerly": 0.75,
    "as of": 0.70,
    "new": 0.50,
    "latest": 0.60,
}


def detect_update_signals(
    statement_or_memory: Union[str, Memory, dict[str, Any]],
) -> dict[str, Any]:
    """
    Detect explicit or linguistic update signals in text or a memory object.
    Supports backward-compatibility with tests expecting `detect_update_signals`.
    """
    if isinstance(statement_or_memory, Memory):
        text = str(statement_or_memory.text or statement_or_memory.value or "")
        meta = statement_or_memory.metadata or {}
    elif isinstance(statement_or_memory, dict):
        text = str(statement_or_memory.get("text") or statement_or_memory.get("value") or "")
        meta = statement_or_memory.get("metadata") or {}
    else:
        text = str(statement_or_memory or "")
        meta = {}

    text_lower = text.lower()
    detected_markers = []
    max_strength = 0.0

    for marker, weight in UPDATE_KEYWORDS.items():
        # Word boundary match for safety
        pattern = r"\b" + re.escape(marker) + r"\b"
        if re.search(pattern, text_lower):
            detected_markers.append(marker)
            if weight > max_strength:
                max_strength = weight

    # Check explicit metadata signals as well
    if meta.get("is_update") is True:
        max_strength = max(max_strength, 0.95)
        detected_markers.append("metadata_is_update")
    if meta.get("is_replacement") is True:
        max_strength = max(max_strength, 0.95)
        detected_markers.append("metadata_is_replacement")
    if meta.get("update_of") or meta.get("supersedes"):
        max_strength = max(max_strength, 1.0)
        detected_markers.append("metadata_targeted")

    has_signal = len(detected_markers) > 0 and max_strength >= 0.60

    return {
        "has_update_signal": has_signal,
        "signal_strength": max_strength,
        "score": max_strength,
        "markers": detected_markers,
        "update_detected": has_signal,
        "text": text,
    }


def analyze_update(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:
    """
    Analyze whether `new_memory` updates or supersedes `old_memory`.
    Combines explicit metadata and linguistic change signals.
    """
    new_meta = new_memory.metadata or {}
    old_meta = old_memory.metadata or {}

    update_of = new_meta.get("update_of")
    supersedes = new_meta.get("supersedes") or []
    if isinstance(supersedes, str):
        supersedes = [supersedes]

    explicit_update = new_meta.get("is_update") is True
    explicit_supersession = (
        old_memory.memory_id in supersedes
        or update_of == old_memory.memory_id
    )

    explicit_replacement = new_meta.get("is_replacement")
    if explicit_replacement is not None and not isinstance(explicit_replacement, bool):
        raise TypeError("metadata.is_replacement must be boolean or null")

    # Detect linguistic update signals from new_memory text
    new_signals = detect_update_signals(new_memory)
    linguistic_score = float(new_signals["signal_strength"])

    # Check temporal ordering if timestamps exist
    temporal_progression = False
    if old_memory.time and new_memory.time:
        try:
            temporal_progression = str(new_memory.time) > str(old_memory.time)
        except Exception:
            pass

    # Targeted when metadata explicitly references old memory OR
    # when same subject and attribute with strong linguistic update signals
    same_subj_attr = (
        bool(old_memory.subject and new_memory.subject)
        and old_memory.subject.strip().casefold() == new_memory.subject.strip().casefold()
        and bool(old_memory.attribute and new_memory.attribute)
        and old_memory.attribute.strip().casefold() == new_memory.attribute.strip().casefold()
    )

    different_value = (
        old_memory.value.strip().casefold() != new_memory.value.strip().casefold()
    )

    targeted = explicit_supersession or (
        explicit_update and update_of == old_memory.memory_id
    )

    if not targeted and same_subj_attr and different_value and linguistic_score >= 0.70:
        targeted = True

    effective_replacement = (
        explicit_replacement
        if explicit_replacement is not None
        else (targeted and different_value)
    )

    return {
        "explicit_update": explicit_update,
        "explicit_supersession": explicit_supersession,
        "explicit_replacement": effective_replacement,
        "targeted_to_old_memory": targeted,
        "update_of": update_of,
        "supersedes": list(supersedes),
        "linguistic_signal": linguistic_score,
        "linguistic_markers": new_signals["markers"],
        "temporal_progression": temporal_progression,
        "same_subject_and_attribute": same_subj_attr,
        "evidence_source": new_meta.get("extraction_source") or "linguistic_and_metadata",
        "evidence_confidence": new_meta.get("extraction_confidence") or max(linguistic_score, 0.8 if targeted else 0.0),
    }


def detect_update(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:
    """Backwards-compatible wrapper."""
    return analyze_update(old_memory, new_memory)