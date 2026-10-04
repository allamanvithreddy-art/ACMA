from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from threading import RLock
from typing import Any, Union
from uuid import uuid4

from memory.schema import Memory, MemoryQuery, utc_now


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
        if max_turns is not None and current_turn is not None:
            if (current_turn - self.turn_index) > max_turns:
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
    def __init__(
        self,
        user_id: str,
        session_id: str,
        capacity: int = 30,
        default_ttl_seconds: float = 3600.0,
        default_max_turns: int = 15,
    ) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.capacity = capacity
        self.default_ttl_seconds = default_ttl_seconds
        self.default_max_turns = default_max_turns
        self.items: list[WorkingMemoryItem] = []
        self.task_context: dict[str, Any] = {}
        self.turn_counter: int = 0
        self.created_at = datetime.now(timezone.utc)
        self.last_active_at = datetime.now(timezone.utc)
        self._lock = RLock()

    def add(
        self,
        memory: Union[Memory, dict[str, Any]],
        ttl_seconds: float | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryItem:
        with self._lock:
            self.last_active_at = datetime.now(timezone.utc)
            if isinstance(memory, dict):
                mem_obj = Memory.from_dict(memory)
            else:
                mem_obj = memory

            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=ttl)
                if ttl > 0
                else None
            )

            item = WorkingMemoryItem(
                memory=mem_obj,
                user_id=self.user_id,
                session_id=self.session_id,
                expires_at=expires_at,
                turn_index=self.turn_counter,
                importance=importance if importance is not None else (mem_obj.importance or 0.5),
                metadata=dict(metadata or {}),
            )

            # Evict expired items first
            self._prune_expired_locked()

            # Capacity management (FIFO for lowest importance items)
            if len(self.items) >= self.capacity:
                # Remove oldest item with lowest importance
                self.items.sort(key=lambda it: (it.importance, -it.turn_index))
                self.items.pop(0)

            self.items.append(item)
            return item

    def get_recent(self, max_items: int = 10, include_expired: bool = False) -> list[Memory]:
        with self._lock:
            self._prune_expired_locked()
            items = self.items if include_expired else [it for it in self.items if not it.is_expired(current_turn=self.turn_counter, max_turns=self.default_max_turns)]
            for it in items:
                it.touch()
            # Return most recent first
            sorted_items = sorted(items, key=lambda it: it.created_at, reverse=True)
            return [it.memory for it in sorted_items[:max_items]]

    def find_matches(self, query: Union[Memory, MemoryQuery], similarity_fn=None, threshold: float = 0.5) -> list[tuple[Memory, float]]:
        with self._lock:
            self._prune_expired_locked()
            active_items = [it for it in self.items if not it.is_expired(current_turn=self.turn_counter, max_turns=self.default_max_turns)]
            results = []
            q_text = (getattr(query, "text", "") or getattr(query, "value", "")).strip().casefold()
            q_subj = getattr(query, "subject", "").strip().casefold()
            q_attr = getattr(query, "attribute", "").strip().casefold()

            for item in active_items:
                mem = item.memory
                score = 0.0
                m_text = (mem.text or mem.value).strip().casefold()
                m_subj = mem.subject.strip().casefold()
                m_attr = mem.attribute.strip().casefold()

                if q_subj and m_subj and q_subj == m_subj:
                    score += 0.3
                if q_attr and m_attr and q_attr == m_attr:
                    score += 0.4
                if q_text and m_text:
                    if q_text == m_text:
                        score = 1.0
                    elif q_text in m_text or m_text in q_text:
                        score = max(score, 0.75)

                if similarity_fn is not None and score < 1.0:
                    try:
                        sim = similarity_fn(mem, query)
                        score = max(score, float(sim))
                    except Exception:
                        pass

                if score >= threshold:
                    item.touch()
                    results.append((mem, score))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

    def step_turn(self) -> int:
        with self._lock:
            self.turn_counter += 1
            self.last_active_at = datetime.now(timezone.utc)
            self._prune_expired_locked()
            return self.turn_counter

    def set_task_context(self, key: str, value: Any) -> None:
        with self._lock:
            self.task_context[key] = value
            self.last_active_at = datetime.now(timezone.utc)

    def get_task_context(self, key: str | None = None) -> Any:
        with self._lock:
            if key is None:
                return dict(self.task_context)
            return self.task_context.get(key)

    def clear(self) -> None:
        with self._lock:
            self.items.clear()
            self.task_context.clear()

    def _prune_expired_locked(self) -> int:
        now = datetime.now(timezone.utc)
        before = len(self.items)
        self.items = [
            it for it in self.items
            if not it.is_expired(current_time=now, current_turn=self.turn_counter, max_turns=self.default_max_turns)
        ]
        return before - len(self.items)


class WorkingMemory:
    """
    Multi-tenant, multi-session Working Memory manager for ACMA.
    Maintains temporary task and conversation context with automatic expiration,
    preventing context leakage between users or sessions.
    """
    def __init__(
        self,
        default_capacity: int = 30,
        default_ttl_seconds: float = 3600.0,
        default_max_turns: int = 15,
    ) -> None:
        self.default_capacity = default_capacity
        self.default_ttl_seconds = default_ttl_seconds
        self.default_max_turns = default_max_turns
        self._sessions: dict[tuple[str, str], SessionWorkingContext] = {}
        self._lock = RLock()

    def _get_session(self, user_id: str, session_id: str) -> SessionWorkingContext:
        key = (str(user_id).strip(), str(session_id).strip())
        with self._lock:
            if key not in self._sessions:
                self._sessions[key] = SessionWorkingContext(
                    user_id=key[0],
                    session_id=key[1],
                    capacity=self.default_capacity,
                    default_ttl_seconds=self.default_ttl_seconds,
                    default_max_turns=self.default_max_turns,
                )
            return self._sessions[key]

    def add_memory(
        self,
        memory: Union[Memory, dict[str, Any]],
        user_id: str = "default_user",
        session_id: str = "default_session",
        ttl_seconds: float | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryItem:
        session = self._get_session(user_id, session_id)
        return session.add(
            memory=memory,
            ttl_seconds=ttl_seconds,
            importance=importance,
            metadata=metadata,
        )

    def get_recent_memories(
        self,
        user_id: str = "default_user",
        session_id: str = "default_session",
        max_items: int = 10,
    ) -> list[Memory]:
        session = self._get_session(user_id, session_id)
        return session.get_recent(max_items=max_items)

    def search(
        self,
        query: Union[Memory, MemoryQuery],
        user_id: str = "default_user",
        session_id: str = "default_session",
        top_k: int = 5,
        threshold: float = 0.4,
    ) -> list[tuple[Memory, float]]:
        session = self._get_session(user_id, session_id)
        matches = session.find_matches(query, threshold=threshold)
        return matches[:top_k]

    def set_task_context(
        self,
        key: str,
        value: Any,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> None:
        session = self._get_session(user_id, session_id)
        session.set_task_context(key, value)

    def get_task_context(
        self,
        key: str | None = None,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> Any:
        session = self._get_session(user_id, session_id)
        return session.get_task_context(key)

    def step_turn(
        self,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> int:
        session = self._get_session(user_id, session_id)
        return session.step_turn()

    def clear_session(
        self,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> None:
        key = (str(user_id).strip(), str(session_id).strip())
        with self._lock:
            if key in self._sessions:
                self._sessions[key].clear()
                del self._sessions[key]

    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
