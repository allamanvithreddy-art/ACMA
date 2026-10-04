from __future__ import annotations

from memory.schema import Memory, MemoryQuery
from memory.store import MemoryStore


class MemoryRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(
        self,
        query: Memory | MemoryQuery,
        *,
        top_k: int = 5,
    ) -> list[tuple[Memory, float]]:
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")

        return self.store.retrieve_related_memories(
            query=query,
            top_k=top_k,
        )