from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from conflict.pipeline import ACPipeline
from evaluation.metrics import PipelineMetrics
from memory.store import MemoryStore
from memory.understanding import MemoryUnderstanding
from memory.working_memory import WorkingMemory


class MemoryEventRequest(BaseModel):
    memory_id: str | None = None
    text: str | None = None
    subject: str
    attribute: str
    value: str
    scope: str | None = None
    context: str | None = None
    time: str | None = None
    source: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    importance: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessEventRequest(BaseModel):
    memory: MemoryEventRequest
    user_id: str = "default_user"
    session_id: str = "default_session"
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.30, ge=-1.0, le=1.0)
    apply_storage: bool = True


class TaskContextRequest(BaseModel):
    key: str
    value: Any


working_memory = WorkingMemory()
store = MemoryStore()
understanding = MemoryUnderstanding()
pipeline = ACPipeline(working_memory=working_memory)
metrics = PipelineMetrics()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Embeddings are loaded lazily by MemoryStore on first retrieval/save.
    yield


app = FastAPI(
    title="ACMA Real-Time Memory API",
    version="3.0.0",
    description="Adaptive Cognitive Memory Architecture with conservative memory conflict resolution.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/memories/process")
def process_memory_event(request: ProcessEventRequest) -> dict[str, Any]:
    try:
        memory = understanding.from_structured_event(
            request.memory.model_dump(exclude_none=True),
            user_id=request.user_id,
        )
        result = pipeline.process_event(
            memory,
            store=store,
            user_id=request.user_id,
            session_id=request.session_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            apply_storage=request.apply_storage,
        )
        for candidate in result.get("candidate_evaluations", []):
            metrics.record(candidate)
        return result
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/memories")
def create_memory(request: ProcessEventRequest) -> dict[str, Any]:
    """Compatibility endpoint: normal memory ingestion uses the full ACMA pipeline."""
    return process_memory_event(request)


@app.get("/memories")
def list_memories(user_id: str = "default_user") -> dict[str, Any]:
    memories = store.get_active_memories(user_id=user_id)
    return {"count": len(memories), "memories": [memory.to_dict() for memory in memories]}


@app.get("/working-memory/{user_id}/{session_id}")
def get_working_memory(user_id: str, session_id: str) -> dict[str, Any]:
    memories = working_memory.get_recent_memories(user_id=user_id, session_id=session_id)
    context = working_memory.get_task_context(user_id=user_id, session_id=session_id)
    return {"user_id": user_id, "session_id": session_id, "item_count": len(memories), "recent_memories": [m.to_dict() for m in memories], "task_context": context}


@app.post("/task-context/{user_id}/{session_id}")
def set_task_context(user_id: str, session_id: str, request: TaskContextRequest) -> dict[str, Any]:
    working_memory.set_task_context(request.key, request.value, user_id=user_id, session_id=session_id)
    return {"user_id": user_id, "session_id": session_id, "task_context": working_memory.get_task_context(user_id=user_id, session_id=session_id)}


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    return metrics.snapshot()
