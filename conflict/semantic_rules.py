from __future__ import annotations

from typing import Any

from memory.schema import Memory


def extract_structural_signals(old_memory: Memory, new_memory: Memory) -> dict[str, Any]:
    old_meta = old_memory.metadata or {}
    new_meta = new_memory.metadata or {}
    return {
        "old_is_constraint": old_meta.get("is_constraint"),
        "new_is_constraint": new_meta.get("is_constraint"),
        "new_violates_memory_id": new_meta.get("violates_memory_id"),
        "new_update_of": new_meta.get("update_of"),
        "new_supersedes": list(new_meta.get("supersedes") or []),
        "declared_relationship": new_meta.get("relationship"),
        "extraction_source": new_meta.get("extraction_source"),
        "extraction_confidence": new_meta.get("extraction_confidence"),
    }


def is_explicit_constraint_violation(old_memory: Memory, new_memory: Memory) -> bool:
    return bool(
        old_memory.metadata.get("is_constraint") is True
        and new_memory.metadata.get("violates_memory_id") == old_memory.memory_id
    )


def is_explicit_event(memory: Memory) -> bool:
    return bool((memory.metadata or {}).get("is_event") is True)


def is_explicit_update(memory: Memory) -> bool:
    return bool((memory.metadata or {}).get("is_update") is True)
