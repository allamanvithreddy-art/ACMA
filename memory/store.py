from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Generator, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from memory.schema import Memory, MemoryQuery


class MemoryStore:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    _shared_embedding_models: dict[str, SentenceTransformer] = {}
    _shared_model_lock = RLock()

    def __init__(self, file_path: str | Path = "data/acma_memory.db", embedding_model: str | None = None) -> None:
        self.file_path = Path(file_path)
        self.embedding_model_name = embedding_model or self.EMBEDDING_MODEL
        self._lock = RLock()
        self._embedding_cache: dict[str, np.ndarray] = {}

        json_source = None
        if self.file_path.suffix.lower() in {".json", ".jsonl"}:
            json_source = self.file_path
            self.file_path = self.file_path.with_suffix(".db")

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        if json_source and json_source.exists():
            self._import_from_json(json_source)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.file_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.file_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default_user',
                    text TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    attribute TEXT NOT NULL,
                    value TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    context TEXT NOT NULL,
                    time TEXT,
                    source TEXT NOT NULL,
                    confidence REAL,
                    importance REAL,
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    supersedes_json TEXT NOT NULL,
                    superseded_by TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_decision TEXT,
                    last_decision_reason TEXT,
                    last_decision_confidence REAL
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(memories)").fetchall()}
            if "user_id" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default_user'")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_status_user ON memories(status, user_id)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_subject_attribute_user ON memories(subject, attribute, user_id)")

    @classmethod
    def _get_embedding_model(cls, model_name: str) -> SentenceTransformer:
        with cls._shared_model_lock:
            if model_name not in cls._shared_embedding_models:
                cls._shared_embedding_models[model_name] = SentenceTransformer(model_name)
            return cls._shared_embedding_models[model_name]

    def _load_embedding_model(self) -> SentenceTransformer:
        return self._get_embedding_model(self.embedding_model_name)

    @staticmethod
    def memory_to_text(memory: Union[Memory, MemoryQuery]) -> str:
        fields = [memory.text, memory.subject, memory.attribute, memory.value, memory.scope, memory.context, memory.time or ""]
        return "\n".join(str(f).strip() for f in fields if str(f).strip())

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json"))
        data["supersedes"] = json.loads(data.pop("supersedes_json"))
        return Memory.from_dict(data)

    def load_memories(self, user_id: str | None = None) -> list[Memory]:
        sql = "SELECT * FROM memories"
        params: tuple = ()
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params = (str(user_id),)
        sql += " ORDER BY created_at"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_active_memories(self, user_id: str | None = None) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE status = 'active'"
        params: tuple = ()
        if user_id is not None:
            sql += " AND user_id = ?"
            params = (str(user_id),)
        sql += " ORDER BY created_at"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_memory(self, memory_id: str, user_id: str | None = None) -> Memory | None:
        sql = "SELECT * FROM memories WHERE memory_id = ?"
        params: tuple = (memory_id,)
        if user_id is not None:
            sql += " AND user_id = ?"
            params += (str(user_id),)
        with self._connection() as connection:
            row = connection.execute(sql, params).fetchone()
        return self._row_to_memory(row) if row else None

    def _upsert_connection(self, connection: sqlite3.Connection, memory: Memory) -> None:
        columns = (
            "memory_id", "user_id", "text", "subject", "attribute", "value", "scope", "context", "time",
            "source", "confidence", "importance", "metadata_json", "status", "version", "supersedes_json",
            "superseded_by", "created_at", "updated_at", "last_decision", "last_decision_reason", "last_decision_confidence",
        )
        values = (
            memory.memory_id, memory.user_id, memory.text, memory.subject, memory.attribute, memory.value, memory.scope,
            memory.context, memory.time, memory.source, memory.confidence, memory.importance,
            json.dumps(memory.metadata, ensure_ascii=False), memory.status, memory.version,
            json.dumps(memory.supersedes, ensure_ascii=False), memory.superseded_by, memory.created_at,
            memory.updated_at, memory.last_decision, memory.last_decision_reason, memory.last_decision_confidence,
        )
        connection.execute(
            f"""
            INSERT INTO memories ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})
            ON CONFLICT(memory_id) DO UPDATE SET
                user_id=excluded.user_id,
                text=excluded.text,
                subject=excluded.subject,
                attribute=excluded.attribute,
                value=excluded.value,
                scope=excluded.scope,
                context=excluded.context,
                time=excluded.time,
                source=excluded.source,
                confidence=excluded.confidence,
                importance=excluded.importance,
                metadata_json=excluded.metadata_json,
                status=excluded.status,
                version=excluded.version,
                supersedes_json=excluded.supersedes_json,
                superseded_by=excluded.superseded_by,
                updated_at=excluded.updated_at,
                last_decision=excluded.last_decision,
                last_decision_reason=excluded.last_decision_reason,
                last_decision_confidence=excluded.last_decision_confidence
            """,
            values,
        )

    def save_memory(self, memory: Memory) -> None:
        with self._connection() as connection:
            self._upsert_connection(connection, memory)

        if memory.status == "active":
            self._cache_embedding(memory)
        else:
            self._embedding_cache.pop(memory.memory_id, None)

    def _cache_embedding(self, memory: Memory) -> None:
        # Avoid forcing model initialization during a write-only operation.
        # The first retrieval will lazily load the model and build the cache.
        with self._shared_model_lock:
            model = self._shared_embedding_models.get(self.embedding_model_name)
        if model is None:
            return
        embedding = model.encode(
            [self.memory_to_text(memory)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        with self._lock:
            self._embedding_cache[memory.memory_id] = np.asarray(embedding, dtype=np.float32)

    def _ensure_embedding_cache(self, memories: list[Memory]) -> None:
        missing = [m for m in memories if m.memory_id not in self._embedding_cache]
        if not missing:
            return
        model = self._load_embedding_model()
        texts = [self.memory_to_text(m) for m in missing]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        with self._lock:
            for memory, embedding in zip(missing, embeddings):
                self._embedding_cache[memory.memory_id] = np.asarray(embedding, dtype=np.float32)

    def retrieve_related_memories(
        self,
        query: Union[Memory, MemoryQuery],
        top_k: int = 5,
        similarity_threshold: float | None = None,
        user_id: str | None = None,
    ) -> list[tuple[Memory, float]]:
        if top_k <= 0:
            return []
        memories = self.get_active_memories(user_id=user_id)
        if not memories:
            return []

        query_text = self.memory_to_text(query)
        if not query_text.strip():
            return []

        self._ensure_embedding_cache(memories)
        model = self._load_embedding_model()
        query_embedding = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        matrix = np.vstack([self._embedding_cache[m.memory_id] for m in memories])
        similarities = matrix @ np.asarray(query_embedding, dtype=np.float32)
        order = np.argsort(similarities)[::-1]

        results: list[tuple[Memory, float]] = []
        for index in order:
            score = float(similarities[int(index)])
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            results.append((memories[int(index)], score))
            if len(results) >= top_k:
                break
        return results

    def refresh_cache(self) -> None:
        with self._lock:
            self._embedding_cache.clear()
        self._ensure_embedding_cache(self.get_active_memories())

    def _build_embedding_cache(self) -> None:
        self.refresh_cache()

    def _import_from_json(self, json_path: Path) -> int:
        with open(json_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            return 0
        count = 0
        for item in raw:
            if isinstance(item, dict):
                self.save_memory(Memory.from_dict(item))
                count += 1
        return count
