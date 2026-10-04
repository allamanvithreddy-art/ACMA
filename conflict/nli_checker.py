from __future__ import annotations

from typing import Any, Union
from memory.schema import Memory
from conflict.nli_engine import NLIEngine


def compare_statements(
    premise: Union[str, Memory, dict[str, Any]],
    hypothesis: Union[str, Memory, dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare two statements using the shared NLIEngine instance.
    """
    engine = NLIEngine.get_default()
    return engine.compare(premise, hypothesis)


__all__ = ["NLIEngine", "compare_statements"]