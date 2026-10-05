from __future__ import annotations

import time
from typing import Any, Union

from memory.schema import Memory, MemoryQuery
from memory.working_memory import WorkingMemory
from memory.evolution import MemoryEvolution
from conflict.nli_engine import NLIEngine
from conflict.relationship import classify_relationship
from conflict.safety_gate import evaluate_rule_safety
from conflict.evidence_score import analyze_evidence
from conflict.decision_policy import decide_action


class ACPipeline:
    def __init__(self, nli_engine: NLIEngine | None = None, working_memory: WorkingMemory | None = None) -> None:
        self.nli_engine = nli_engine or NLIEngine.get_default()
        self.working_memory = working_memory or WorkingMemory()
        self._total_candidates = 0
        self._total_nli_calls = 0

    def evaluate_candidate(self, old_memory: Memory, new_memory: Memory, retrieval_score: float | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        relationship = classify_relationship(old_memory, new_memory)
        safety_result = evaluate_rule_safety(
            old_memory,
            new_memory,
            retrieval_score,
            relationship_info=relationship,
        )
        safety = safety_result.to_dict()

        nli_result = None
        if safety_result.next_stage == "nli":
            nli_result = self.nli_engine.compare(old_memory, new_memory)
            self._total_nli_calls += 1

        evidence = analyze_evidence(safety=safety, relationship=relationship, nli=nli_result)
        decision = decide_action(safety=safety, evidence=evidence)
        self._total_candidates += 1

        return {
            "old_memory_id": old_memory.memory_id,
            "new_memory_id": new_memory.memory_id,
            "old_memory": old_memory.to_dict(),
            "new_memory": new_memory.to_dict(),
            "retrieval_score": retrieval_score,
            "relationship": relationship["relationship"],
            "relationship_details": relationship,
            "safety": safety,
            "nli": nli_result,
            "evidence": evidence,
            "decision": decision,
            "action": decision.action,
            "reason": decision.reason,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
        }

    def evaluate_candidates(self, new_memory: Memory, candidates: list[tuple[Memory, float]]) -> list[dict[str, Any]]:
        return [self.evaluate_candidate(old, new_memory, score) for old, score in candidates]

    @staticmethod
    def aggregate_event_decision(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidate_results:
            return {
                "action": "Preserve",
                "reason": "No relevant prior memory was retrieved; store the incoming memory.",
                "target_candidates": [],
                "requires_user_confirmation": False,
            }

        asks = [r for r in candidate_results if r["action"] == "Ask"]
        if asks:
            return {
                "action": "Ask",
                "reason": "At least one relevant memory remains ambiguous or contradictory without a safe replacement signal.",
                "target_candidates": [r["old_memory_id"] for r in asks],
                "requires_user_confirmation": True,
            }

        resolves = [r for r in candidate_results if r["action"] == "Resolve"]
        if len(resolves) == 1:
            return {
                "action": "Resolve",
                "reason": resolves[0]["reason"],
                "target_candidates": [resolves[0]["old_memory_id"]],
                "requires_user_confirmation": False,
            }
        if len(resolves) > 1:
            return {
                "action": "Ask",
                "reason": "Multiple existing memories appear to be replacement targets; automatic resolution would be ambiguous.",
                "target_candidates": [r["old_memory_id"] for r in resolves],
                "requires_user_confirmation": True,
            }

        duplicates = [r for r in candidate_results if r["action"] == "Ignore"]
        non_duplicates = [r for r in candidate_results if r["action"] not in {"Ignore", "Preserve"}]
        if duplicates and not non_duplicates:
            return {
                "action": "Ignore",
                "reason": "The incoming memory is already represented by an existing memory.",
                "target_candidates": [r["old_memory_id"] for r in duplicates],
                "requires_user_confirmation": False,
            }

        return {
            "action": "Preserve",
            "reason": "All relevant memories are compatible or independent; retain the incoming memory.",
            "target_candidates": [r["old_memory_id"] for r in candidate_results],
            "requires_user_confirmation": False,
        }

    def evaluate_retrieved(
        self,
        *,
        new_memory: Memory,
        retrieved: list[tuple[Memory, float]],
    ) -> dict[str, Any]:
        """
        Compatibility/evaluation entry point for an already completed RAG step.

        The actual reasoning path remains:
            relationship -> safety gate -> NLI when needed
            -> evidence -> decision policy
        """
        candidate_results = self.evaluate_candidates(
            new_memory,
            retrieved,
        )

        if not candidate_results:
            return {
                "chosen": None,
                "action": "Preserve",
                "confidence": 1.0,
                "nli_calls": 0,
                "candidates": [],
            }

        relation_priority = {
            "duplicate": 6,
            "update": 6,
            "constraint_violation": 6,
            "conflict": 5,
            "unresolved": 4,
            "compatible": 2,
            "paraphrase": 2,
            "independent": 1,
        }

        action_priority = {
            "Resolve": 5,
            "Ask": 4,
            "Ignore": 3,
            "Preserve": 1,
        }

        def priority(result: dict[str, Any]) -> tuple[float, float, float]:
            relationship = result.get("relationship", "unresolved")
            action = result.get("action", "Ask")

            evidence = result.get("evidence", {})
            evidence_score = float(
                evidence.get("score", 0.0)
                if isinstance(evidence, dict)
                else evidence
            )

            retrieval = float(result.get("retrieval_score") or 0.0)

            return (
                relation_priority.get(relationship, 0),
                action_priority.get(action, 0),
                evidence_score + retrieval,
            )

        chosen = max(candidate_results, key=priority)

        evidence = chosen.get("evidence", {})
        confidence = float(
            evidence.get("score", 0.0)
            if isinstance(evidence, dict)
            else evidence
        )

        nli_calls = sum(
            1 for result in candidate_results
            if result.get("nli") is not None
        )

        return {
            "chosen": chosen,
            "action": chosen["action"],
            "confidence": confidence,
            "nli_calls": nli_calls,
            "candidates": candidate_results,
        }

    def process_event(
        self,
        event: Union[Memory, dict[str, Any]],
        *,
        store: Any,
        user_id: str = "default_user",
        session_id: str = "default_session",
        top_k: int = 5,
        similarity_threshold: float = 0.30,
        apply_storage: bool = True,
    ) -> dict[str, Any]:
        memory = Memory.from_dict(event) if isinstance(event, dict) else event
        memory.user_id = str(user_id)

        # Working Memory is context, not long-term storage. Add the event after
        # taking a snapshot so the event cannot retrieve itself as a prior claim.
        prior_working = self.working_memory.get_recent_memories(
            user_id=user_id,
            session_id=session_id,
            max_items=20,
        )
        working_context = self.working_memory.get_task_context(user_id=user_id, session_id=session_id)
        self.working_memory.add_memory(memory, user_id=user_id, session_id=session_id)

        query = MemoryQuery.from_memory(memory)
        candidates = store.retrieve_related_memories(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            user_id=user_id,
        )

        # Merge lightweight current-session structural matches with long-term RAG.
        working_matches = self.working_memory.search(memory, user_id=user_id, session_id=session_id, top_k=min(top_k, 5))
        merged: dict[str, tuple[Memory, float]] = {m.memory_id: (m, score) for m, score in candidates}
        for working_memory, score in working_matches:
            if working_memory.memory_id == memory.memory_id:
                continue
            previous = merged.get(working_memory.memory_id)
            if previous is None or score > previous[1]:
                merged[working_memory.memory_id] = (working_memory, score)
        candidates = sorted(merged.values(), key=lambda item: item[1], reverse=True)[:top_k]

        evaluations = self.evaluate_candidates(memory, candidates)
        event_decision = self.aggregate_event_decision(evaluations)

        storage_result = None
        if apply_storage:
            evolution = MemoryEvolution(store)
            action = event_decision["action"]
            if action == "Ignore":
                storage_result = {"stored": False, "action": "Ignore"}
            elif action == "Preserve":
                storage_result = evolution.apply_decision(None, memory, event_decision)
            elif action == "Ask":
                storage_result = evolution.apply_decision(None, memory, event_decision)
            elif action == "Resolve":
                target_id = event_decision["target_candidates"][0]
                old_memory = store.get_memory(target_id, user_id=user_id)
                storage_result = evolution.apply_decision(old_memory, memory, event_decision)

        return {
            "memory": memory.to_dict(),
            "user_id": user_id,
            "session_id": session_id,
            "working_memory_context": {
                "recent_count": len(prior_working),
                "recent_memories": [m.to_dict() for m in prior_working],
                "task_context": working_context,
            },
            "retrieved_candidates_count": len(candidates),
            "candidate_evaluations": evaluations,
            "event_decision": event_decision,
            "storage_result": storage_result,
        }

    def statistics(self) -> dict[str, Any]:
        return {
            "total_candidates": self._total_candidates,
            "total_nli_calls": self._total_nli_calls,
            "nli_rate": self._total_nli_calls / self._total_candidates if self._total_candidates else 0.0,
        }
