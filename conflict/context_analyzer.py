from __future__ import annotations

from datetime import datetime
from typing import Any

from memory.schema import Memory


_SCOPE_ALIASES = {
    "general": "general",
    "persistent": "general",
    "global": "general",
    "stable": "general",
    "specific_event": "event",
    "specific": "event",
    "event": "event",
    "episodic": "event",
    "temporary": "event",
    "context_specific": "context_specific",
    "contextual": "context_specific",
}


def normalize_scope(value: Any) -> str:
    """Canonicalize scope labels without using domain-specific vocabulary."""
    raw = " ".join(str(value or "").strip().casefold().split())
    return _SCOPE_ALIASES.get(raw, raw)


def normalize_context(value: Any) -> str:
    """Normalize context for comparison only; do not infer its meaning."""
    return " ".join(str(value or "").strip().casefold().split())


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def analyze_context(old_memory: Memory, new_memory: Memory) -> dict[str, Any]:
    """Compare scope/context/time generically.

    This stage does not decide whether two claims conflict. It only describes
    whether they appear to apply to the same scope/context and whether the new
    memory is event-scoped or persistent.
    """
    old_scope = normalize_scope(old_memory.scope)
    new_scope = normalize_scope(new_memory.scope)
    old_context = normalize_context(old_memory.context)
    new_context = normalize_context(new_memory.context)

    if old_context and new_context:
        if old_context == new_context:
            context_relation = "same"
        else:
            context_relation = "different"
    elif old_context or new_context:
        context_relation = "partial"
    else:
        context_relation = "unknown"

    if old_scope == new_scope:
        scope_relation = "same"
    elif old_scope == "general" and new_scope == "event":
        scope_relation = "general_to_event"
    elif old_scope == "event" and new_scope == "general":
        scope_relation = "event_to_general"
    elif old_scope or new_scope:
        scope_relation = "different"
    else:
        scope_relation = "unknown"

    old_time = _parse_time(old_memory.time)
    new_time = _parse_time(new_memory.time)
    if old_time and new_time:
        if new_time > old_time:
            temporal_relation = "later"
        elif new_time < old_time:
            temporal_relation = "earlier"
        else:
            temporal_relation = "same"
    else:
        temporal_relation = "unknown"

    same_subject = old_memory.subject.strip().casefold() == new_memory.subject.strip().casefold()
    same_attribute = old_memory.attribute.strip().casefold() == new_memory.attribute.strip().casefold()

    return {
        "old_scope": old_scope,
        "new_scope": new_scope,
        "scope_relation": scope_relation,
        "old_context": old_context,
        "new_context": new_context,
        "context_relation": context_relation,
        "old_time": old_memory.time,
        "new_time": new_memory.time,
        "temporal_relation": temporal_relation,
        "old_is_event": old_scope == "event",
        "new_is_event": new_scope == "event",
        "old_is_general": old_scope == "general",
        "new_is_general": new_scope == "general",
        "same_subject": same_subject,
        "same_attribute": same_attribute,
        "same_context": context_relation == "same",
        "potential_scope_exception": old_scope == "general" and new_scope == "event",
    }
