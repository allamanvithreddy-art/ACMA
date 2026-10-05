from __future__ import annotations

import tempfile
from pathlib import Path

from conflict.pipeline import ACPipeline
from memory.schema import Memory
from memory.store import MemoryStore
from memory.working_memory import WorkingMemory


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="acma_human_case_") as tmp:
        db = Path(tmp) / "memory.db"
        store = MemoryStore(file_path=db)
        pipeline = ACPipeline(working_memory=WorkingMemory())

        old = Memory(
            memory_id="old_vegetarian",
            user_id="demo_user",
            subject="user",
            attribute="diet",
            value="I am vegetarian",
            scope="general",
            context="general",
            source="user",
            confidence=1.0,
            importance=0.8,
        )
        store.save_memory(old)

        new = Memory(
            memory_id="new_chicken_event",
            user_id="demo_user",
            subject="user",
            attribute="diet",
            value="I ate chicken at a wedding",
            scope="specific_event",
            context="wedding",
            source="user",
            confidence=1.0,
            importance=0.7,
        )

        result = pipeline.process_event(
            new,
            store=store,
            user_id="demo_user",
            session_id="demo_session",
            top_k=5,
            similarity_threshold=0.0,
            apply_storage=False,
        )

        print("relationship:", result["candidate_evaluations"][0]["relationship"] if result["candidate_evaluations"] else None)
        print("safety:", result["candidate_evaluations"][0]["safety"] if result["candidate_evaluations"] else None)
        print("nli:", result["candidate_evaluations"][0]["nli"] if result["candidate_evaluations"] else None)
        print("decision:", result["event_decision"])

        action = result["event_decision"]["action"]
        if action != "Ask":
            raise SystemExit(f"FAIL: expected Ask, got {action!r}")
        print("PASS: vegetarian + chicken-at-event => Ask")


if __name__ == "__main__":
    main()
