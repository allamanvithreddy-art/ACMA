from __future__ import annotations

from typing import Any

from memory.schema import Memory
from conflict.context_analyzer import analyze_context, normalize_context, normalize_scope
from conflict.metadata import compare_metadata, normalize_field
from conflict.semantic_rules import extract_structural_signals
from conflict.update_detector import analyze_update


def _same_claim(old: Memory, new: Memory, metadata: dict[str, Any]) -> bool:
    if metadata.get("same_claim_id"):
        return True

    if not metadata.get("same_subject") or not metadata.get("same_attribute"):
        return False

    if normalize_field(old.value) != normalize_field(new.value):
        return False

    old_scope = normalize_scope(old.scope)
    new_scope = normalize_scope(new.scope)
    old_context = normalize_context(old.context)
    new_context = normalize_context(new.context)

    return old_scope == new_scope and old_context == new_context


def _same_subject_attribute(old: Memory, new: Memory) -> bool:
    return bool(
        normalize_field(old.subject)
        and normalize_field(new.subject)
        and normalize_field(old.attribute)
        and normalize_field(new.attribute)
        and normalize_field(old.subject) == normalize_field(new.subject)
        and normalize_field(old.attribute) == normalize_field(new.attribute)
    )


def classify_relationship(old_memory: Memory, new_memory: Memory) -> dict[str, Any]:
    """Return a generic structural relationship.

    No domain-specific words (wedding, travel, diet, etc.) are used here.
    Semantic contradiction is deliberately left to NLI when structure is
    insufficient to establish a safe deterministic result.
    """
    metadata = compare_metadata(old_memory, new_memory)
    context = analyze_context(old_memory, new_memory)
    update = analyze_update(old_memory, new_memory)
    structural = extract_structural_signals(old_memory, new_memory)

    same_subject = metadata["same_subject"]
    same_attribute = metadata["same_attribute"]
    same_value = metadata["same_value"]
    explicit_constraint = bool(
        old_memory.metadata.get("is_constraint") is True
        and new_memory.metadata.get("violates_memory_id") == old_memory.memory_id
    )

    relationship = "unresolved"

    if _same_claim(old_memory, new_memory, metadata):
        relationship = "duplicate"
    elif explicit_constraint:
        relationship = "constraint_violation"
    elif update["targeted_to_old_memory"]:
        relationship = "update"
    elif not same_subject:
        relationship = "independent"
    elif not same_attribute:
        relationship = "independent"
    elif same_value:
        relationship = "compatible"
    elif same_subject and same_attribute:
        # Same entity + same attribute + different value is a potential conflict.
        # Context/scope may explain the difference, but do not declare it safe.
        relationship = "conflict"

    if relationship == "unresolved" and structural.get("declared_relationship"):
        declared = str(structural["declared_relationship"]).casefold()
        if declared in {"duplicate", "independent", "compatible", "paraphrase"}:
            # Upstream metadata can confirm benign relationships, but only when
            # the structural fields do not contradict that declaration.
            if declared == "independent" and (same_subject and same_attribute):
                pass
            else:
                relationship = declared

    return {
        "relationship": relationship,
        "metadata": metadata,
        "context": context,
        "update": update,
        "structural": structural,
        "same_subject_and_attribute": _same_subject_attribute(old_memory, new_memory),
        "old_scope": old_memory.scope,
        "new_scope": new_memory.scope,
        "old_context": old_memory.context,
        "new_context": new_memory.context,
        "old_time": old_memory.time,
        "new_time": new_memory.time,
        "explicit_update_target": update["targeted_to_old_memory"],
        "explicit_constraint_violation": explicit_constraint,
    }
