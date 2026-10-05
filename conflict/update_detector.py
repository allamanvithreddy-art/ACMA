from __future__ import annotations

import re
from typing import Any, Union

from memory.schema import Memory


# ============================================================
# REAL-TIME UPDATE / CHANGE VOCABULARY
# ============================================================

STRONG_UPDATE_PATTERNS = {
    # Explicit replacement / switching
    r"\bswitched\s+from\b": 1.00,
    r"\bswitched\s+to\b": 0.98,
    r"\bswitched\b.*\bto\b": 0.98,

    r"\bchanged\s+from\b": 0.98,
    r"\bchanged\s+to\b": 0.95,
    r"\bchanged\b.*\bto\b": 0.95,

    r"\breplaced\b": 0.98,
    r"\breplace\b": 0.92,
    r"\breplacing\b": 0.92,

    r"\bmigrated\s+from\b": 0.98,
    r"\bmigrated\s+to\b": 0.98,
    r"\bmigrated\b.*\bto\b": 0.98,

    r"\bmoved\s+from\b": 0.95,
    r"\bmoved\s+to\b": 0.95,
    r"\bmoved\b.*\bto\b": 0.92,

    r"\bshifted\s+from\b": 0.96,
    r"\bshifted\s+to\b": 0.96,
    r"\bshifted\b.*\bto\b": 0.95,

    r"\btransitioned\s+from\b": 0.96,
    r"\btransitioned\s+to\b": 0.96,
    r"\btransitioned\b.*\bto\b": 0.95,

    r"\bconverted\s+from\b": 0.95,
    r"\bconverted\s+to\b": 0.95,

    r"\bupgraded\s+from\b": 0.90,
    r"\bupgraded\s+to\b": 0.90,

    r"\bdowngraded\s+from\b": 0.90,
    r"\bdowngraded\s+to\b": 0.90,

    # Explicit invalidation / stopping
    r"\bno longer\b": 1.00,
    r"\bnot anymore\b": 0.98,
    r"\bnot any more\b": 0.98,

    r"\bstopped\s+using\b": 0.98,
    r"\bstopped\s+doing\b": 0.90,
    r"\bstopped\s+preferring\b": 0.92,

    r"\bquit\s+using\b": 0.96,
    r"\bquit\s+doing\b": 0.90,

    r"\bdropped\b": 0.90,
    r"\babandoned\b": 0.92,
    r"\bdiscontinued\b": 0.94,

    # Adoption / new state
    r"\bstarted\s+using\b": 0.88,
    r"\bstarted\s+preferring\b": 0.88,
    r"\bstarted\s+doing\b": 0.80,

    r"\bbegan\s+using\b": 0.88,
    r"\bbegan\s+to\s+use\b": 0.88,
    r"\bbegan\s+preferring\b": 0.88,

    r"\badopted\b": 0.90,
    r"\badopting\b": 0.88,

    r"\bnow\s+use\b": 0.98,
    r"\bnow\s+prefer\b": 0.96,
    r"\bnow\s+want\b": 0.90,
    r"\bnow\s+have\b": 0.85,
    r"\bnow\s+live\b": 0.95,
    r"\bnow\s+work\b": 0.90,
    r"\bnow\s+study\b": 0.88,

    r"\bcurrently\s+use\b": 0.88,
    r"\bcurrently\s+prefer\b": 0.86,
    r"\bcurrently\s+live\b": 0.88,
    r"\bcurrently\s+work\b": 0.86,
    r"\bcurrently\s+study\b": 0.84,

    r"\bpresently\s+use\b": 0.84,
    r"\bpresently\s+prefer\b": 0.82,

    # Preference/state changes
    r"\bchanged\s+my\s+mind\b": 0.92,
    r"\bi\s+now\s+prefer\b": 0.96,
    r"\bi\s+no\s+longer\s+prefer\b": 1.00,
    r"\bi\s+used\s+to\s+prefer\b": 0.82,

    # Future/current policy changes
    r"\bfrom\s+now\s+on\b": 0.94,
    r"\bgoing\s+forward\b": 0.94,
    r"\bhenceforth\b": 0.94,
    r"\beffective\s+(?:today|now|immediately)\b": 0.94,

    # Temporal progression
    r"\bpreviously\b": 0.72,
    r"\bformerly\b": 0.80,
    r"\bformer\b": 0.72,
    r"\bearlier\b": 0.68,
    r"\bused\s+to\b": 0.76,
    r"\buntil\s+now\b": 0.84,
    r"\bup\s+until\s+now\b": 0.86,
    r"\bas\s+of\b": 0.82,
    r"\bfrom\s+today\b": 0.88,
    r"\bfrom\s+this\s+point\s+on\b": 0.94,

    # Transition language
    r"\binstead\s+of\b": 0.92,
    r"\brather\s+than\b": 0.86,
    r"\bin\s+place\s+of\b": 0.90,
}


WEAK_UPDATE_PATTERNS = {
    r"\bnow\b": 0.70,
    r"\bcurrently\b": 0.68,
    r"\bpresently\b": 0.65,
    r"\brecently\b": 0.62,
    r"\blately\b": 0.62,
    r"\bthese\s+days\b": 0.64,
    r"\btoday\b": 0.55,
    r"\bnew\b": 0.50,
    r"\blatest\b": 0.55,
    r"\bchanged\b": 0.65,
    r"\bswitched\b": 0.75,
    r"\bmoved\b": 0.65,
    r"\bstarted\b": 0.62,
    r"\bbegan\b": 0.62,
    r"\bprefer\b": 0.50,
}


TEMPORARY_PATTERNS = {
    r"\btemporarily\b",
    r"\bfor\s+now\b",
    r"\bfor\s+today\b",
    r"\bfor\s+tonight\b",
    r"\bfor\s+tomorrow\b",
    r"\bthis\s+time\b",
    r"\bthis\s+week\b",
    r"\bthis\s+month\b",
    r"\bfor\s+the\s+day\b",
    r"\bfor\s+the\s+meeting\b",
    r"\bfor\s+the\s+presentation\b",
    r"\bfor\s+the\s+event\b",
    r"\bfor\s+the\s+wedding\b",
    r"\bduring\s+the\b",
    r"\bwhile\s+at\b",
    r"\bwhile\s+in\b",
    r"\bonly\s+for\b",
    r"\bjust\s+for\b",
    r"\bthis\s+session\b",
    r"\bthis\s+conversation\b",
}


NEGATION_PATTERNS = {
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


TRANSITION_PATTERNS = {
    r"\bfrom\s+.+?\s+to\s+.+": 1.00,
    r"\bfrom\s+.+?\s+into\s+.+": 0.96,
    r"\bfrom\s+.+?\s+toward\s+.+": 0.90,
    r"\b.+?\s+instead\s+of\s+.+": 0.94,
    r"\b.+?\s+rather\s+than\s+.+": 0.88,
    r"\b.+?\s+in\s+place\s+of\s+.+": 0.92,
}


def _extract_text(
    statement_or_memory: Union[str, Memory, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if isinstance(statement_or_memory, Memory):
        text = str(
            statement_or_memory.text
            or statement_or_memory.value
            or ""
        )
        meta = statement_or_memory.metadata or {}

    elif isinstance(statement_or_memory, dict):
        text = str(
            statement_or_memory.get("text")
            or statement_or_memory.get("value")
            or ""
        )
        meta = statement_or_memory.get("metadata") or {}

    else:
        text = str(statement_or_memory or "")
        meta = {}

    return text, meta


def _find_patterns(
    text: str,
    patterns: dict[str, float],
) -> tuple[list[str], float]:
    markers = []
    max_strength = 0.0

    for pattern, weight in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            markers.append(pattern)
            max_strength = max(max_strength, weight)

    return markers, max_strength


def detect_update_signals(
    statement_or_memory: Union[str, Memory, dict[str, Any]],
) -> dict[str, Any]:
    """
    Runtime linguistic update detection.

    This analyzes the actual incoming statement. It does NOT make
    the final Ignore/Preserve/Resolve/Ask decision.
    """

    text, meta = _extract_text(statement_or_memory)
    normalized = " ".join(text.lower().split())

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
        {p: 1.0 for p in TEMPORARY_PATTERNS},
    )

    negation_markers, _ = _find_patterns(
        normalized,
        {p: 1.0 for p in NEGATION_PATTERNS},
    )

    transition_markers, transition_score = _find_patterns(
        normalized,
        TRANSITION_PATTERNS,
    )

    metadata_is_update = meta.get("is_update") is True
    metadata_is_replacement = meta.get("is_replacement") is True
    metadata_targeted = bool(
        meta.get("update_of")
        or meta.get("supersedes")
    )

    # Structured additive-memory evidence.
    # This is different from replacement:
    #   old list + new items -> merge/preserve
    additive_update = bool(
        meta.get("additive_update")
        or meta.get("is_additive_update")
        or meta.get("update_type") == "redesigned_list_update"
        or meta.get("operation") == "add"
        or meta.get("memory_updates")
    )

    score = max(
        strong_score,
        transition_score,
        weak_score,
    )

    if metadata_is_update:
        score = max(score, 0.95)

    if metadata_is_replacement:
        score = max(score, 0.98)

    if metadata_targeted:
        score = 1.00

    explicit_replacement_signal = (
        metadata_is_replacement
        or metadata_targeted
        or transition_score >= 0.94
        or strong_score >= 0.94
    )

    temporary_context = bool(temporary_markers)

    strong_permanent_change = (
        explicit_replacement_signal
        or transition_score >= 0.94
    )

    likely_temporary_change = (
        temporary_context
        and not strong_permanent_change
    )

    has_signal = (
        metadata_is_update
        or metadata_is_replacement
        or metadata_targeted
        or strong_score >= 0.70
        or transition_score >= 0.80
        or (
            weak_score >= 0.60
            and not likely_temporary_change
        )
    )

    markers = []
    markers.extend(
        f"strong:{m}"
        for m in strong_markers
    )
    markers.extend(
        f"weak:{m}"
        for m in weak_markers
    )
    markers.extend(
        f"transition:{m}"
        for m in transition_markers
    )

    return {
        "has_update_signal": has_signal,
        "signal_strength": score,
        "score": score,
        "markers": markers,
        "update_detected": has_signal,

        "explicit_replacement_signal":
            explicit_replacement_signal,

        "transition_detected":
            transition_score >= 0.80,

        "transition_strength":
            transition_score,

        "temporary_context":
            temporary_context,

        "temporary_markers":
            temporary_markers,

        "likely_temporary_change":
            likely_temporary_change,

        "negation_detected":
            bool(negation_markers),

        "negation_markers":
            negation_markers,

        "additive_update": additive_update,

        "text": text,
    }


def analyze_update(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:

    new_meta = new_memory.metadata or {}

    update_of = new_meta.get("update_of")

    supersedes = (
        new_meta.get("supersedes")
        or []
    )

    if isinstance(supersedes, str):
        supersedes = [supersedes]

    explicit_update = (
        new_meta.get("is_update") is True
    )

    explicit_supersession = (
        old_memory.memory_id in supersedes
        or update_of == old_memory.memory_id
    )

    explicit_replacement = (
        new_meta.get("is_replacement")
    )

    if (
        explicit_replacement is not None
        and not isinstance(
            explicit_replacement,
            bool,
        )
    ):
        raise TypeError(
            "metadata.is_replacement must be boolean or null"
        )

    new_signals = detect_update_signals(
        new_memory
    )

    linguistic_score = float(
        new_signals["signal_strength"]
    )

    additive_update = bool(
        new_signals.get("additive_update")
        or new_meta.get("additive_update")
        or new_meta.get("is_additive_update")
        or new_meta.get("update_type") == "redesigned_list_update"
        or new_meta.get("memory_updates")
    )

    temporal_progression = False

    if old_memory.time and new_memory.time:
        try:
            temporal_progression = (
                str(new_memory.time)
                > str(old_memory.time)
            )
        except Exception:
            temporal_progression = False

    same_subject = (
        bool(old_memory.subject)
        and bool(new_memory.subject)
        and old_memory.subject.strip().casefold()
        == new_memory.subject.strip().casefold()
    )

    same_attribute = (
        bool(old_memory.attribute)
        and bool(new_memory.attribute)
        and old_memory.attribute.strip().casefold()
        == new_memory.attribute.strip().casefold()
    )

    same_subj_attr = (
        same_subject
        and same_attribute
    )

    different_value = (
        old_memory.value.strip().casefold()
        != new_memory.value.strip().casefold()
    )

    targeted = (
        explicit_supersession
        or (
            explicit_update
            and update_of == old_memory.memory_id
        )
    )

    # IMPORTANT:
    # A linguistic update signal alone does NOT authorize replacement.
    # Words such as "currently", "started", "began", "recently", etc.
    # indicate change/current state, but they do not prove that an older
    # memory should be superseded.
    #
    # Replacement requires stronger evidence such as:
    #   - switched from A to B
    #   - changed from A to B
    #   - replaced A with B
    #   - no longer ...
    #   - instead of ...
    #   - explicit replacement metadata

    strong_replacement_signal = (
        new_signals["explicit_replacement_signal"]
        or new_signals["transition_detected"]
    )

    if (
        not targeted
        and same_subj_attr
        and different_value
        and not new_signals["likely_temporary_change"]
        and strong_replacement_signal
    ):
        targeted = True

    if explicit_replacement is not None:
        effective_replacement = explicit_replacement
    else:
        effective_replacement = (
            targeted
            and different_value
            and not new_signals["likely_temporary_change"]
            and strong_replacement_signal
        )

    return {
        "explicit_update":
            explicit_update,

        "explicit_supersession":
            explicit_supersession,

        "explicit_replacement":
            effective_replacement,

        "targeted_to_old_memory":
            targeted,

        "update_of":
            update_of,

        "supersedes":
            list(supersedes),

        "linguistic_signal":
            linguistic_score,

        "linguistic_markers":
            new_signals["markers"],

        "transition_detected":
            new_signals["transition_detected"],

        "temporary_context":
            new_signals["temporary_context"],

        "likely_temporary_change":
            new_signals["likely_temporary_change"],

        "negation_detected":
            new_signals["negation_detected"],

        "temporal_progression":
            temporal_progression,

        "same_subject_and_attribute":
            same_subj_attr,

        "same_subject":
            same_subject,

        "same_attribute":
            same_attribute,

        "different_value":
            different_value,

        "replacement_supported":
            effective_replacement,

        "additive_update":
            additive_update,

        "evidence_source":
            new_meta.get("extraction_source")
            or "linguistic_and_metadata",

        "evidence_confidence":
            new_meta.get("extraction_confidence")
            or max(
                linguistic_score,
                0.8 if targeted else 0.0,
            ),
    }


def detect_update(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:
    """Backward-compatible wrapper."""
    return analyze_update(
        old_memory,
        new_memory,
    )
