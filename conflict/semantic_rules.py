from __future__ import annotations

from typing import Any

from memory.schema import Memory


def extract_structural_signals(
    old_memory: Memory,
    new_memory: Memory,
) -> dict[str, Any]:
    """
    Extract only structural lineage information.

    Semantic interpretation belongs to relationship analysis and NLI.
    This module must not depend on benchmark operation labels.
    """

    old_meta = old_memory.metadata or {}
    new_meta = new_memory.metadata or {}

    supersedes = new_meta.get("supersedes") or []
    if isinstance(supersedes, str):
        supersedes = [supersedes]

    return {
        "new_update_of": new_meta.get("update_of"),
        "new_supersedes": list(supersedes),
        "declared_relationship": new_meta.get("relationship"),
        "claim_id": new_meta.get("claim_id"),
        "same_claim_id": bool(
            old_meta.get("claim_id")
            and old_meta.get("claim_id") == new_meta.get("claim_id")
        ),
        "extraction_source": new_meta.get("extraction_source"),
        "extraction_confidence": new_meta.get("extraction_confidence"),
    }
