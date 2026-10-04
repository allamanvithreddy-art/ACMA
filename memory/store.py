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
    _shared_embedding_model: SentenceTransformer | None = None
    _shared_model_lock = RLock()

    def __init__(
        self,
        file_path: str | Path = "data/acma_memory.db",
        embedding_model: str | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.embedding_model_name = embedding_model or self.EMBEDDING_MODEL
        self._lock = RLock()

        json_source_to_import = None
        if self.file_path.suffix.lower() in {".json", ".jsonl"}:
            json_source_to_import = self.file_path
            self.file_path = self.file_path.with_suffix(".db")

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

        if json_source_to_import is not None and json_source_to_import.exists():
            self._import_from_json(json_source_to_import)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(
            self.file_path,
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        """Backwards compatible raw connection getter."""
        connection = sqlite3.connect(
            self.file_path,
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
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
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_status
                ON memories(status)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_subject_attribute
                ON memories(subject, attribute)
                """
            )

    @classmethod
    def _get_shared_embedding_model(cls, model_name: str) -> SentenceTransformer:
        with cls._shared_model_lock:
            if cls._shared_embedding_model is None:
                cls._shared_embedding_model = SentenceTransformer(model_name)
            return cls._shared_embedding_model

    def _load_embedding_model(self) -> SentenceTransformer:
        return self._get_shared_embedding_model(self.embedding_model_name)

    @staticmethod
    def memory_to_text(
        memory: Union[Memory, MemoryQuery],
    ) -> str:
        fields = [
            memory.text,
            memory.subject,
            memory.attribute,
            memory.value,
            memory.scope,
            memory.context,
            memory.time or "",
        ]
        return "\n".join(str(field).strip() for field in fields if field)

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json"))
        data["supersedes"] = json.loads(data.pop("supersedes_json"))
        return Memory.from_dict(data)

    def load_memories(self) -> list[Memory]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY created_at"
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_active_memories(self) -> list[Memory]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE status = 'active'
                ORDER BY created_at
                """
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_memory(self, memory_id: str) -> Memory | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def save_memory(self, memory: Memory) -> None:
        columns = (
            "memory_id", "text", "subject", "attribute", "value",
            "scope", "context", "time", "source", "confidence",
            "importance", "metadata_json", "status", "version",
            "supersedes_json", "superseded_by", "created_at",
            "updated_at", "last_decision", "last_decision_reason",
            "last_decision_confidence",
        )
        values = (
            memory.memory_id,
            memory.text,
            memory.subject,
            memory.attribute,
            memory.value,
            memory.scope,
            memory.context,
            memory.time,
            memory.source,
            memory.confidence,
            memory.importance,
            json.dumps(memory.metadata, ensure_ascii=False),
            memory.status,
            memory.version,
            json.dumps(memory.supersedes, ensure_ascii=False),
            memory.superseded_by,
            memory.created_at,
            memory.updated_at,
            memory.last_decision,
            memory.last_decision_reason,
            memory.last_decision_confidence,
        )

        placeholders = ", ".join("?" for _ in columns)
        column_names = ", ".join(columns)

        with self._connection() as connection:
            connection.execute(
                f"""
                INSERT INTO memories ({column_names})
                VALUES ({placeholders})
                ON CONFLICT(memory_id) DO UPDATE SET
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

    def retrieve_related_memories(
        self,
        query: Union[Memory, MemoryQuery],
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> list[tuple[Memory, float]]:
        if top_k <= 0:
            return []

        memories = self.get_active_memories()
        if not memories:
            return []

        model = self._load_embedding_model()
        query_text = self.memory_to_text(query)

        if not query_text.strip():
            return []

        memory_texts = [self.memory_to_text(memory) for memory in memories]

        embeddings = model.encode(
            memory_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        similarities = np.asarray(embeddings) @ np.asarray(query_embedding)
        ranked = np.argsort(similarities)[::-1]

        results = []
        for index in ranked:
            score = float(similarities[index])
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            results.append((memories[int(index)], score))
            if len(results) >= top_k:
                break

        return results

    def refresh_cache(self) -> None:
        return None

    def _build_embedding_cache(self) -> None:
        memories = self.get_active_memories()
        if memories:
            model = self._load_embedding_model()
            texts = [self.memory_to_text(m) for m in memories]
            model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    def _import_from_json(self, json_path: Path) -> int:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, list):
                return 0
            count = 0
            for item in raw:
                if isinstance(item, dict):
                    mem = Memory.from_dict(item)
                    self.save_memory(mem)
                    count += 1
            return count
        except Exception:
            return 0