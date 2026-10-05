from __future__ import annotations

from memory.schema import Memory
from memory.store import MemoryStore


class MemoryEvolution:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def apply_decision(self, old_memory: Memory | None, new_memory: Memory, decision: dict) -> dict:
        action = str(decision.get("action", ""))
        reason = str(decision.get("reason", ""))
        if action not in {"Ignore", "Preserve", "Resolve", "Ask"}:
            raise ValueError(f"Unsupported memory action: {action!r}")

        if action == "Ignore":
            return {"action": action, "stored": False, "memory_id": old_memory.memory_id if old_memory else None, "reason": reason}

        new_memory.last_decision = action
        new_memory.last_decision_reason = reason

        if action == "Ask":
            new_memory.status = "pending"
            new_memory.update_timestamp()
            self.store.save_memory(new_memory)
            return {"action": action, "stored": True, "status": new_memory.status, "memory_id": new_memory.memory_id, "reason": reason}

        if action == "Preserve":
            new_memory.status = "active"
            new_memory.update_timestamp()
            self.store.save_memory(new_memory)
            return {"action": action, "stored": True, "status": new_memory.status, "memory_id": new_memory.memory_id, "reason": reason}

        if old_memory is None:
            raise ValueError("Resolve requires an existing memory target")
        if old_memory.user_id != new_memory.user_id:
            raise ValueError("Cannot resolve memories belonging to different users")

        if old_memory.memory_id not in new_memory.supersedes:
            new_memory.supersedes.append(old_memory.memory_id)

        old_memory.status = "superseded"
        old_memory.superseded_by = new_memory.memory_id
        old_memory.update_timestamp()

        new_memory.status = "active"
        new_memory.version = old_memory.version + 1
        new_memory.update_timestamp()

        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self.store._upsert_connection(connection, old_memory)
                self.store._upsert_connection(connection, new_memory)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        # Update embedding cache for the newly active memory.
        self.store.save_memory(new_memory)
        return {
            "action": action,
            "stored": True,
            "old_memory_id": old_memory.memory_id,
            "new_memory_id": new_memory.memory_id,
            "status": new_memory.status,
            "reason": reason,
        }
