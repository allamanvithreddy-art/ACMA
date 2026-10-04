from __future__ import annotations

from typing import Any
from memory.schema import Memory


def normalize_field(value: Any) -> str:
    """Normalize formatting, not meaning."""
    return " ".join(str(value or "").casefold().split())


def compare_metadata(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:
    fields = ("subject", "attribute", "scope", "context", "value")

    comparison = {
        f"same_{name}": (
            normalize_field(getattr(old_memory, name))
            == normalize_field(getattr(new_memory, name))
        )
        for name in fields
    }

    # Empty fields are not evidence of equality.
    comparison["same_subject"] = (
        bool(normalize_field(old_memory.subject))
        and comparison["same_subject"]
    )
    comparison["same_attribute"] = (
        bool(normalize_field(old_memory.attribute))
        and comparison["same_attribute"]
    )
    comparison["same_value"] = (
        bool(normalize_field(old_memory.value))
        and comparison["same_value"]
    )

    comparison["same_context"] = (
        bool(normalize_field(old_memory.context))
        and comparison["same_context"]
    )
    comparison["same_scope"] = (
        bool(normalize_field(old_memory.scope))
        and comparison["same_scope"]
    )

    comparison["same_subject_and_attribute"] = (
        comparison["same_subject"] and comparison["same_attribute"]
    )

    comparison["old_scope"] = old_memory.scope
    comparison["new_scope"] = new_memory.scope
    comparison["old_context"] = old_memory.context
    comparison["new_context"] = new_memory.context
    comparison["old_value"] = old_memory.value
    comparison["new_value"] = new_memory.value

    # Explicit evidence references must come from the extractor or caller.
    old_meta = old_memory.metadata
    new_meta = new_memory.metadata

    comparison["same_claim_id"] = (
        bool(old_meta.get("claim_id"))
        and old_meta.get("claim_id") == new_meta.get("claim_id")
    )
    comparison["new_update_of_old"] = (
        new_meta.get("update_of") == old_memory.memory_id
    )
    comparison["new_supersedes_old"] = (
        old_memory.memory_id
        in (new_meta.get("supersedes") or [])
    )

    comparison["related"] = (
        comparison["same_subject_and_attribute"]
        or comparison["new_update_of_old"]
        or comparison["new_supersedes_old"]
    )

    return comparison


def metadata_agreement(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:
    """Return interpretable field comparisons, not a fabricated accuracy score."""
    return compare_metadata(old_memory, new_memory)