from __future__ import annotations

from memory.schema import Memory
from memory.store import MemoryStore


class MemoryEvolution:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def apply_decision(
        self,
        old_memory: Memory | None,
        new_memory: Memory,
        decision: dict,
    ) -> dict:
        action = decision.get("action")
        reason = decision.get("reason", "")

        if action not in {"Ignore", "Preserve", "Resolve", "Ask"}:
            raise ValueError(f"Unsupported memory action: {action!r}")

        if action == "Ignore":
            return {
                "action": action,
                "stored": False,
                "memory_id": old_memory.memory_id if old_memory else None,
                "reason": reason,
            }

        new_memory.last_decision = action
        new_memory.last_decision_reason = reason

        if action == "Ask":
            new_memory.status = "pending"
            new_memory.update_timestamp()
            self.store.save_memory(new_memory)
            return {
                "action": action,
                "stored": True,
                "status": new_memory.status,
                "memory_id": new_memory.memory_id,
                "reason": reason,
            }

        if action == "Preserve":
            new_memory.status = "active"
            new_memory.update_timestamp()
            self.store.save_memory(new_memory)
            return {
                "action": action,
                "stored": True,
                "status": new_memory.status,
                "memory_id": new_memory.memory_id,
                "reason": reason,
            }

        # Resolve is allowed only when the decision policy has explicitly
        # established that the incoming memory replaces this old memory.
        if old_memory is None:
            raise ValueError("Resolve requires the old memory being replaced")

        if old_memory.memory_id not in new_memory.supersedes:
            new_memory.supersedes.append(old_memory.memory_id)

        old_memory.status = "superseded"
        old_memory.superseded_by = new_memory.memory_id
        old_memory.update_timestamp()

        new_memory.status = "active"
        new_memory.version = old_memory.version + 1
        new_memory.update_timestamp()

        # Both changes are written in a transaction so a partial replacement
        # is not intentionally created by this method.
        with self.store._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._write_with_connection(connection, old_memory)
                self._write_with_connection(connection, new_memory)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        return {
            "action": action,
            "stored": True,
            "old_memory_id": old_memory.memory_id,
            "new_memory_id": new_memory.memory_id,
            "status": new_memory.status,
            "reason": reason,
        }

    @staticmethod
    def _write_with_connection(connection, memory: Memory) -> None:
        import json

        columns = (
            "memory_id", "text", "subject", "attribute", "value",
            "scope", "context", "time", "source", "confidence",
            "importance", "metadata_json", "status", "version",
            "supersedes_json", "superseded_by", "created_at",
            "updated_at", "last_decision", "last_decision_reason",
            "last_decision_confidence",
        )
        values = (
            memory.memory_id, memory.text, memory.subject,
            memory.attribute, memory.value, memory.scope, memory.context,
            memory.time, memory.source, memory.confidence, memory.importance,
            json.dumps(memory.metadata, ensure_ascii=False), memory.status,
            memory.version, json.dumps(memory.supersedes, ensure_ascii=False),
            memory.superseded_by, memory.created_at, memory.updated_at,
            memory.last_decision, memory.last_decision_reason,
            memory.last_decision_confidence,
        )

        connection.execute(
            f"""
            INSERT INTO memories ({", ".join(columns)})
            VALUES ({", ".join("?" for _ in columns)})
            ON CONFLICT(memory_id) DO UPDATE SET
                status=excluded.status,
                version=excluded.version,
                supersedes_json=excluded.supersedes_json,
                superseded_by=excluded.superseded_by,
                updated_at=excluded.updated_at,
                last_decision=excluded.last_decision,
                last_decision_reason=excluded.last_decision_reason
            """,
            values,
        )