from __future__ import annotations

import re
from typing import Any
from memory.schema import Memory
from conflict.metadata import compare_metadata, normalize_field
from conflict.update_detector import analyze_update
from conflict.semantic_rules import extract_structural_signals


EVENT_CONTEXT_KEYWORDS = {
    "wedding", "party", "conference", "trip", "travel", "presentation",
    "meeting", "vacation", "lunch", "dinner", "yesterday", "last saturday",
    "last night", "temporary", "exception", "event"
}

EVENT_ACTION_PATTERNS = [
    r"\bate\b", r"\bvisited\b", r"\btravelled\b", r"\btraveled\b",
    r"\bran a charity\b", r"\battended\b", r"\bwent to\b",
    r"\bduring the presentation\b", r"\bfor tomorrow\b", r"\btemporarily\b"
]


def _is_specific_event(old_memory: Memory, new_memory: Memory) -> bool:
    """
    Check whether new_memory represents a specific one-off event/exception
    to an existing general preference, rule, or constraint.
    """
    old_scope = normalize_field(old_memory.scope)
    new_scope = normalize_field(new_memory.scope)
    old_ctx = normalize_field(old_memory.context)
    new_ctx = normalize_field(new_memory.context)
    new_text = str(new_memory.text or new_memory.value or "").casefold()

    # Scope-based check
    if old_scope in ("general", "persistent", "global") and new_scope in ("specific_event", "specific", "event", "temporary"):
        return True

    # Context-based check
    if any(kw in new_ctx for kw in EVENT_CONTEXT_KEYWORDS) and (not old_ctx or old_ctx in ("general", "diet", "lifestyle")):
        return True

    # Linguistic event verb check
    for pat in EVENT_ACTION_PATTERNS:
        if re.search(pat, new_text):
            # Check if old memory is a general stance (e.g., vegetarian, dark mode)
            old_text = str(old_memory.text or old_memory.value or "").casefold()
            if any(term in old_text for term in ("vegetarian", "vegan", "normally", "always", "generally", "prefers", "uses")):
                return True

    return False


def classify_relationship(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:
    metadata = compare_metadata(old_memory, new_memory)
    update = analyze_update(old_memory, new_memory)
    structural = extract_structural_signals(old_memory, new_memory)

    declared = structural.get("declared_relationship")

    # 1. Exact or normalized duplicate check
    old_norm_val = normalize_field(old_memory.value or old_memory.text)
    new_norm_val = normalize_field(new_memory.value or new_memory.text)
    old_norm_subj = normalize_field(old_memory.subject)
    new_norm_subj = normalize_field(new_memory.subject)
    old_norm_attr = normalize_field(old_memory.attribute)
    new_norm_attr = normalize_field(new_memory.attribute)

    is_duplicate = False
    if metadata.get("same_claim_id"):
        is_duplicate = True
    elif old_norm_val and new_norm_val and old_norm_val == new_norm_val:
        # Same value with compatible or missing subject/attribute
        if (not old_norm_subj or not new_norm_subj or old_norm_subj == new_norm_subj):
            if (not old_norm_attr or not new_norm_attr or old_norm_attr == new_norm_attr):
                is_duplicate = True

    # 2. Explicit constraint violation
    is_constraint_violation = (
        old_memory.metadata.get("is_constraint") is True
        and new_memory.metadata.get("violates_memory_id") == old_memory.memory_id
    )

    # 3. Determine relationship dynamically
    relationship = "unresolved"

    if declared and declared not in ("unresolved", "unknown"):
        relationship = declared
    elif is_duplicate:
        relationship = "duplicate"
    elif is_constraint_violation:
        relationship = "constraint_violation"
    elif _is_specific_event(old_memory, new_memory):
        relationship = "specific_event"
    elif (
        old_norm_subj and new_norm_subj and old_norm_subj != new_norm_subj
    ):
        # Different people or entities (e.g., Caroline vs Melanie, User vs System)
        relationship = "independent"
    elif (
        old_norm_attr and new_norm_attr and old_norm_attr != new_norm_attr
    ):
        # Different attributes of the same entity (e.g., programming language vs diet)
        relationship = "independent"
    elif (
        metadata["same_subject_and_attribute"]
        and old_norm_val != new_norm_val
        and (update["targeted_to_old_memory"] or update.get("linguistic_signal", 0) >= 0.70)
    ):
        # Same entity and attribute with explicit or linguistic update markers
        relationship = "update"
    elif (
        metadata["same_subject_and_attribute"]
        and old_norm_val != new_norm_val
        and not update["targeted_to_old_memory"]
        and update.get("linguistic_signal", 0) < 0.60
    ):
        # Same subject and attribute, different values, but no update signal -> conflict
        relationship = "conflict"
    elif (
        # Different distinct contexts (e.g. DSA coursework vs ACMA project)
        old_memory.context and new_memory.context
        and normalize_field(old_memory.context) != normalize_field(new_memory.context)
        and old_norm_val != new_norm_val
    ):
        relationship = "independent"
    else:
        relationship = "unresolved"

    return {
        "relationship": relationship,
        "metadata": metadata,
        "update": update,
        "structural": structural,
        "old_scope": old_memory.scope,
        "new_scope": new_memory.scope,
        "old_context": old_memory.context,
        "new_context": new_memory.context,
        "old_time": old_memory.time,
        "new_time": new_memory.time,
        "same_subject_and_attribute": metadata["same_subject_and_attribute"],
        "explicit_update_target": update["targeted_to_old_memory"],
        "explicit_constraint_violation": is_constraint_violation,
    }