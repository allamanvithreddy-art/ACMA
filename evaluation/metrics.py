from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Any


class PipelineMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counts = Counter()
        self._latencies_ms: list[float] = []

    def record(self, result: dict[str, Any]) -> None:
        safety = result.get("safety") or {}
        decision = result.get("decision") or {}

        with self._lock:
            self._counts["candidates_evaluated"] += 1

            if safety.get("next_stage") == "nli":
                self._counts["nli_calls"] += 1
            elif safety.get("next_stage") == "decision":
                self._counts["nli_bypasses"] += 1

            if decision.get("action"):
                self._counts[f"decision_{decision['action'].casefold()}"] += 1

            if result.get("llm_fallback_called"):
                self._counts["llm_fallback_calls"] += 1

            if result.get("latency_ms") is not None:
                self._latencies_ms.append(float(result["latency_ms"]))

    def record_ground_truth(
        self,
        *,
        stage: str,
        expected: Any,
        actual: Any,
    ) -> None:
        """
        Call this only when expected labels come from a real annotated set.
        """
        with self._lock:
            self._counts[f"{stage}_ground_truth_cases"] += 1
            if expected == actual:
                self._counts[f"{stage}_ground_truth_correct"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            latencies = list(self._latencies_ms)

        def accuracy(stage: str) -> float | None:
            total = counts.get(f"{stage}_ground_truth_cases", 0)
            if not total:
                return None
            correct = counts.get(f"{stage}_ground_truth_correct", 0)
            return correct / total

        candidates = counts.get("candidates_evaluated", 0)
        nli_calls = counts.get("nli_calls", 0)
        bypasses = counts.get("nli_bypasses", 0)

        return {
            "counts": counts,
            "nli_bypass_rate": bypasses / candidates if candidates else None,
            "nli_call_rate": nli_calls / candidates if candidates else None,
            "relationship_accuracy": accuracy("relationship"),
            "safety_gate_accuracy": accuracy("safety_gate"),
            "nli_accuracy": accuracy("nli"),
            "decision_policy_accuracy": accuracy("decision_policy"),
            "end_to_end_accuracy": accuracy("end_to_end"),
            "latency_ms": {
                "count": len(latencies),
                "mean": (
                    sum(latencies) / len(latencies) if latencies else None
                ),
                "max": max(latencies) if latencies else None,
            },
        }