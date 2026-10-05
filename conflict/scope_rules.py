from __future__ import annotations

from memory.schema import Memory
from conflict.context_analyzer import analyze_context
from conflict.relationship import classify_relationship


# Backward-compatible module path. The old implementation depended on
# domain-specific event keywords; all of that logic has been removed.


def analyze_scope(old_memory: Memory, new_memory: Memory) -> dict:
    return analyze_context(old_memory, new_memory)


__all__ = ["classify_relationship", "analyze_scope"]
