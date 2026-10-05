from __future__ import annotations

from typing import Any

from memory.schema import Memory


def normalize_field(value: Any) -> str:
    """Normalize formatting for comparison; do not infer semantics."""
    return " ".join(str(value or "").casefold().split())


def compare_metadata(old_memory: Memory, new_memory: Memory) -> dict[str, Any]:
    fields = ("subject", "attribute", "scope", "context", "value")
    comparison = {
        f"same_{name}": normalize_field(getattr(old_memory, name)) == normalize_field(getattr(new_memory, name))
        for name in fields
    }
    for name in fields:
        comparison[f"same_{name}"] = bool(normalize_field(getattr(old_memory, name))) and comparison[f"same_{name}"]

    comparison["same_subject_and_attribute"] = comparison["same_subject"] and comparison["same_attribute"]

    old_meta = old_memory.metadata or {}
    new_meta = new_memory.metadata or {}
    comparison["same_claim_id"] = bool(old_meta.get("claim_id")) and old_meta.get("claim_id") == new_meta.get("claim_id")
    comparison["new_update_of_old"] = new_meta.get("update_of") == old_memory.memory_id
    comparison["new_supersedes_old"] = old_memory.memory_id in (new_meta.get("supersedes") or [])
    comparison["related"] = comparison["same_subject_and_attribute"] or comparison["new_update_of_old"] or comparison["new_supersedes_old"]
    comparison["old_value"] = old_memory.value
    comparison["new_value"] = new_memory.value
    return comparison


def metadata_agreement(old_memory: Memory, new_memory: Memory) -> dict[str, Any]:
    return compare_metadata(old_memory, new_memory)
