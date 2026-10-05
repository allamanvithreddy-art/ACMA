from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Callable, Union
from uuid import uuid4

from memory.schema import Memory, MemoryQuery


@dataclass
class WorkingMemoryItem:
    item_id: str = field(default_factory=lambda: str(uuid4()))
    memory: Memory = field(default_factory=Memory)
    user_id: str = "default_user"
    session_id: str = "default_session"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    turn_index: int = 0
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_time: datetime | None = None, current_turn: int | None = None, max_turns: int | None = None) -> bool:
        now = current_time or datetime.now(timezone.utc)
        if self.expires_at is not None and now >= self.expires_at:
            return True
        if max_turns is not None and current_turn is not None and current_turn - self.turn_index > max_turns:
            return True
        return False

    def touch(self) -> None:
        self.last_accessed_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "memory": self.memory.to_dict(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "turn_index": self.turn_index,
            "importance": self.importance,
            "metadata": dict(self.metadata),
        }


class SessionWorkingContext:
    def __init__(self, user_id: str, session_id: str, capacity: int = 30, default_ttl_seconds: float = 3600.0, default_max_turns: int = 15) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.capacity = capacity
        self.default_ttl_seconds = default_ttl_seconds
        self.default_max_turns = default_max_turns
        self.items: list[WorkingMemoryItem] = []
        self.task_context: dict[str, Any] = {}
        self.turn_counter = 0
        self._lock = RLock()

    def _prune(self) -> None:
        now = datetime.now(timezone.utc)
        self.items = [
            item for item in self.items
            if not item.is_expired(current_time=now, current_turn=self.turn_counter, max_turns=self.default_max_turns)
        ]

    def add(self, memory: Memory, ttl_seconds: float | None = None, importance: float | None = None, metadata: dict[str, Any] | None = None) -> WorkingMemoryItem:
        with self._lock:
            self._prune()
            ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl) if ttl > 0 else None
            item = WorkingMemoryItem(
                memory=memory,
                user_id=self.user_id,
                session_id=self.session_id,
                expires_at=expires_at,
                turn_index=self.turn_counter,
                importance=importance if importance is not None else (memory.importance or 0.5),
                metadata=dict(metadata or {}),
            )
            if len(self.items) >= self.capacity:
                self.items.sort(key=lambda entry: (entry.importance, entry.turn_index))
                self.items.pop(0)
            self.items.append(item)
            return item

    def recent(self, max_items: int = 10) -> list[Memory]:
        with self._lock:
            self._prune()
            ordered = sorted(self.items, key=lambda entry: entry.created_at, reverse=True)
            for item in ordered[:max_items]:
                item.touch()
            return [item.memory for item in ordered[:max_items]]

    def search(self, query: Memory | MemoryQuery, top_k: int = 5, threshold: float = 0.45, similarity_fn: Callable[[Memory, Memory | MemoryQuery], float] | None = None) -> list[tuple[Memory, float]]:
        with self._lock:
            self._prune()
            q_text = str(getattr(query, "text", "") or getattr(query, "value", "")).casefold()
            q_subject = str(getattr(query, "subject", "")).strip().casefold()
            q_attribute = str(getattr(query, "attribute", "")).strip().casefold()
            results: list[tuple[Memory, float]] = []
            for item in self.items:
                memory = item.memory
                score = 0.0
                if q_subject and memory.subject.strip().casefold() == q_subject:
                    score += 0.35
                if q_attribute and memory.attribute.strip().casefold() == q_attribute:
                    score += 0.45
                text = str(memory.text or memory.value).casefold()
                if q_text and text:
                    if q_text == text:
                        score = 1.0
                    elif q_text in text or text in q_text:
                        score = max(score, 0.75)
                if similarity_fn is not None and score < 1.0:
                    score = max(score, float(similarity_fn(memory, query)))
                if score >= threshold:
                    item.touch()
                    results.append((memory, min(score, 1.0)))
            results.sort(key=lambda item: item[1], reverse=True)
            return results[:top_k]

    def step_turn(self) -> int:
        with self._lock:
            self.turn_counter += 1
            self._prune()
            return self.turn_counter

    def set_task_context(self, key: str, value: Any) -> None:
        with self._lock:
            self.task_context[str(key)] = value

    def get_task_context(self, key: str | None = None) -> Any:
        with self._lock:
            return dict(self.task_context) if key is None else self.task_context.get(key)

    def clear(self) -> None:
        with self._lock:
            self.items.clear()
            self.task_context.clear()


class WorkingMemory:
    def __init__(self, default_capacity: int = 30, default_ttl_seconds: float = 3600.0, default_max_turns: int = 15) -> None:
        self.default_capacity = default_capacity
        self.default_ttl_seconds = default_ttl_seconds
        self.default_max_turns = default_max_turns
        self._sessions: dict[tuple[str, str], SessionWorkingContext] = {}
        self._lock = RLock()

    def _get_session(self, user_id: str, session_id: str) -> SessionWorkingContext:
        key = (str(user_id).strip(), str(session_id).strip())
        with self._lock:
            return self._sessions.setdefault(
                key,
                SessionWorkingContext(
                    key[0], key[1], self.default_capacity, self.default_ttl_seconds, self.default_max_turns,
                ),
            )

    def add_memory(self, memory: Memory | dict[str, Any], user_id: str = "default_user", session_id: str = "default_session", ttl_seconds: float | None = None, importance: float | None = None, metadata: dict[str, Any] | None = None) -> WorkingMemoryItem:
        memory_obj = Memory.from_dict(memory) if isinstance(memory, dict) else memory
        memory_obj.user_id = str(user_id)
        return self._get_session(user_id, session_id).add(memory_obj, ttl_seconds, importance, metadata)

    def get_recent_memories(self, user_id: str = "default_user", session_id: str = "default_session", max_items: int = 10) -> list[Memory]:
        return self._get_session(user_id, session_id).recent(max_items)

    def search(self, query: Memory | MemoryQuery, user_id: str = "default_user", session_id: str = "default_session", top_k: int = 5, threshold: float = 0.45, similarity_fn: Callable[[Memory, Memory | MemoryQuery], float] | None = None) -> list[tuple[Memory, float]]:
        return self._get_session(user_id, session_id).search(query, top_k=top_k, threshold=threshold, similarity_fn=similarity_fn)

    def set_task_context(self, key: str, value: Any, user_id: str = "default_user", session_id: str = "default_session") -> None:
        self._get_session(user_id, session_id).set_task_context(key, value)

    def get_task_context(self, key: str | None = None, user_id: str = "default_user", session_id: str = "default_session") -> Any:
        return self._get_session(user_id, session_id).get_task_context(key)

    def step_turn(self, user_id: str = "default_user", session_id: str = "default_session") -> int:
        return self._get_session(user_id, session_id).step_turn()

    def clear_session(self, user_id: str = "default_user", session_id: str = "default_session") -> None:
        key = (str(user_id), str(session_id))
        with self._lock:
            context = self._sessions.pop(key, None)
        if context:
            context.clear()
