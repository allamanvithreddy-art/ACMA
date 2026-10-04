PRAGMA foreign_keys = ON;

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
);

CREATE INDEX IF NOT EXISTS idx_memories_status
    ON memories(status);

CREATE INDEX IF NOT EXISTS idx_memories_subject_attribute
    ON memories(subject, attribute);