from __future__ import annotations

import time
from typing import Any, Union

from memory.schema import Memory, MemoryQuery
from memory.working_memory import WorkingMemory
from memory.evolution import MemoryEvolution
from conflict.nli_engine import NLIEngine
from conflict.scope_rules import classify_relationship
from conflict.safety_gate import evaluate_rule_safety
from conflict.evidence_score import analyze_evidence
from conflict.decision_policy import decide_action, DecisionResult


class ACPipeline:
    """
    Adaptive Cognitive Memory Architecture (ACMA) Conflict & Integration Pipeline.
    Connects Normalization, Working Memory, RAG Retrieval, Safety Gate, NLI Engine,
    Evidence Scoring, and Decision Policy.
    """
    def __init__(
        self,
        nli_engine: NLIEngine | None = None,
        working_memory: WorkingMemory | None = None,
    ) -> None:
        self.nli_engine = nli_engine or NLIEngine.get_default()
        self.working_memory = working_memory or WorkingMemory()
        self._total_candidates: int = 0
        self._total_nli_calls: int = 0

    def evaluate_candidate(
        self,
        old_memory: Memory,
        new_memory: Memory,
        retrieval_score: float | None = None,
    ) -> dict[str, Any]:
        start_t = time.perf_counter()

        # 1. Structural and Relationship Classification
        relationship = classify_relationship(old_memory, new_memory)

        # 2. Safety Gate evaluation
        safety_res = evaluate_rule_safety(
            old_memory=old_memory,
            new_memory=new_memory,
            similarity=retrieval_score,
        )
        safety = safety_res.to_dict()

        # 3. NLI Engine (only if routed by Safety Gate)
        nli_result = None
        if safety_res.next_stage == "nli":
            nli_result = self.nli_engine.compare(
                old_memory,
                new_memory,
            )

        # 4. Evidence Analysis
        evidence = analyze_evidence(
            safety=safety,
            relationship=relationship,
            nli=nli_result,
        )

        # 5. Decision Policy
        decision = decide_action(
            safety=safety,
            evidence=evidence,
        )

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "old_memory_id": old_memory.memory_id,
            "new_memory_id": new_memory.memory_id,
            "retrieval_score": retrieval_score,
            "relationship": relationship.get("relationship", "unresolved"),
            "relationship_details": relationship,
            "safety": safety,
            "nli": nli_result,
            "evidence": evidence,
            "decision": decision,
            "action": decision.get("action", str(decision)),
            "reason": decision.get("reason", ""),
            "latency_ms": latency_ms,
        }

    def evaluate_candidates(
        self,
        new_memory: Memory,
        candidates: list[tuple[Memory, float]],
    ) -> list[dict[str, Any]]:
        results = []
        for old_memory, retrieval_score in candidates:
            results.append(
                self.evaluate_candidate(
                    old_memory=old_memory,
                    new_memory=new_memory,
                    retrieval_score=retrieval_score,
                )
            )
        return results

    def evaluate_retrieved(
        self,
        new_memory: Memory,
        retrieved: list[tuple[Memory, float]],
    ) -> dict[str, Any]:
        """
        Evaluate retrieved candidates and return chosen top candidate along with event statistics.
        """
        candidates = self.evaluate_candidates(new_memory, retrieved)
        event_decision = self.aggregate_event_decision(candidates)

        chosen = None
        if candidates:
            matches = [c for c in candidates if c["action"] == event_decision["action"]]
            chosen = matches[0] if matches else candidates[0]
            # Ensure chosen relationship is represented as dict for callers accessing chosen["relationship"]["relationship"]
            if not isinstance(chosen.get("relationship"), dict):
                rel_val = chosen.get("relationship", "unresolved")
                chosen["relationship"] = {"relationship": rel_val}

        nli_calls = sum(1 for c in candidates if c.get("nli") is not None)
        self._total_candidates += len(candidates)
        self._total_nli_calls += nli_calls

        return {
            "chosen": chosen,
            "candidates": candidates,
            "action": event_decision["action"],
            "reason": event_decision.get("reason", ""),
            "confidence": 0.95 if event_decision["action"] in ("Ignore", "Preserve", "Resolve") else 0.5,
            "nli_calls": nli_calls,
            "candidate_count": len(candidates),
        }

    def statistics(self) -> dict[str, Any]:
        rate = (
            float(self._total_nli_calls) / float(self._total_candidates)
            if self._total_candidates > 0
            else 0.0
        )
        return {
            "total_candidates": self._total_candidates,
            "total_nli_calls": self._total_nli_calls,
            "nli_rate": rate,
        }


    def aggregate_event_decision(
        self,
        candidate_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Aggregate per-candidate decisions into a consistent event-level decision.
        Priority:
          1. Ignore (duplicate detected -> ignore incoming event)
          2. Resolve (valid update -> supersede old memory)
          3. Ask (conflict / ambiguity -> request clarification)
          4. Preserve / Store (all compatible / independent -> store as active)
        """
        if not candidate_results:
            return {
                "action": "Preserve",
                "reason": "No prior memories retrieved; store incoming memory.",
                "target_candidates": [],
                "requires_user_confirmation": False,
            }

        # Check for Ignore (duplicate)
        duplicates = [r for r in candidate_results if r["action"] == "Ignore"]
        if duplicates:
            return {
                "action": "Ignore",
                "reason": "Incoming memory is a confirmed duplicate of existing memory.",
                "target_candidates": [r["old_memory_id"] for r in duplicates],
                "requires_user_confirmation": False,
            }

        # Check for Resolve (update)
        resolutions = [r for r in candidate_results if r["action"] == "Resolve"]
        if resolutions:
            return {
                "action": "Resolve",
                "reason": "Incoming memory supersedes existing memory with verified update evidence.",
                "target_candidates": [r["old_memory_id"] for r in resolutions],
                "requires_user_confirmation": False,
            }

        # Check for Ask (unresolved conflict)
        asks = [r for r in candidate_results if r["action"] == "Ask"]
        if asks:
            return {
                "action": "Ask",
                "reason": "Unresolved conflict or ambiguity detected; user clarification required.",
                "target_candidates": [r["old_memory_id"] for r in asks],
                "requires_user_confirmation": True,
            }

        # Otherwise all Preserve
        return {
            "action": "Preserve",
            "reason": "All retrieved memories are independent or compatible; retain new memory.",
            "target_candidates": [r["old_memory_id"] for r in candidate_results],
            "requires_user_confirmation": False,
        }

    def process_event(
        self,
        event: Union[Memory, dict[str, Any]],
        *,
        store: Any = None,
        user_id: str = "default_user",
        session_id: str = "default_session",
        top_k: int = 5,
        similarity_threshold: float = 0.30,
        apply_storage: bool = True,
    ) -> dict[str, Any]:
        """
        Execute the complete ACMA pipeline for an incoming memory event:
        Working Memory -> RAG Retrieval -> Safety Gate -> NLI -> Decision -> Storage.
        """
        if isinstance(event, dict):
            memory = Memory.from_dict(event)
        else:
            memory = event

        # Step 1: Add to Working Memory (buffer for current session)
        wm_item = self.working_memory.add_memory(
            memory=memory,
            user_id=user_id,
            session_id=session_id,
        )

        # Step 2: RAG Retrieval from Persistent Store (if provided)
        candidates = []
        if store is not None:
            query = MemoryQuery.from_memory(memory)
            candidates = store.retrieve_related_memories(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

        # Step 3: Candidate Evaluations
        candidate_evaluations = self.evaluate_candidates(
            new_memory=memory,
            candidates=candidates,
        )

        # Step 4: Aggregate Event Decision
        event_decision = self.aggregate_event_decision(candidate_evaluations)

        # Step 5: Apply Storage Evolution (if requested and store available)
        storage_result = None
        if apply_storage and store is not None:
            evolution = MemoryEvolution(store)
            action = event_decision["action"]

            if action == "Ignore":
                storage_result = {"stored": False, "action": "Ignore"}
            elif action == "Resolve":
                # Supersede target candidates
                target_ids = event_decision.get("target_candidates", [])
                for target_id in target_ids:
                    old_mem = store.get_memory(target_id)
                    if old_mem:
                        storage_result = evolution.apply_decision(
                            old_memory=old_mem,
                            new_memory=memory,
                            decision=event_decision,
                        )
            elif action == "Ask":
                storage_result = evolution.apply_decision(
                    old_memory=None,
                    new_memory=memory,
                    decision={"action": "Ask", "reason": event_decision["reason"]},
                )
            elif action == "Preserve":
                storage_result = evolution.apply_decision(
                    old_memory=None,
                    new_memory=memory,
                    decision={"action": "Preserve", "reason": event_decision["reason"]},
                )

        return {
            "memory": memory.to_dict(),
            "user_id": user_id,
            "session_id": session_id,
            "working_memory_item_id": wm_item.item_id,
            "retrieved_candidates_count": len(candidates),
            "candidate_evaluations": candidate_evaluations,
            "event_decision": event_decision,
            "storage_result": storage_result,
        }