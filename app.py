from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memory.understanding import MemoryUnderstanding
from memory.store import MemoryStore
from memory.working_memory import WorkingMemory
from retrieval.retriever import MemoryRetriever
from conflict.pipeline import ACPipeline
from evaluation.metrics import PipelineMetrics


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


class EvaluationRequest(BaseModel):
    memory: MemoryEventRequest
    top_k: int = Field(default=5, ge=1, le=100)
    user_id: str = "default_user"
    session_id: str = "default_session"


class ProcessEventRequest(BaseModel):
    memory: MemoryEventRequest
    user_id: str = "default_user"
    session_id: str = "default_session"
    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    apply_storage: bool = True


class TaskContextRequest(BaseModel):
    key: str
    value: Any


working_memory = WorkingMemory()
store = MemoryStore()
understanding = MemoryUnderstanding()
retriever = MemoryRetriever(store)
pipeline = ACPipeline(working_memory=working_memory)
metrics = PipelineMetrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="ACMA Real-Time Memory API",
    version="2.0.0",
    description="Adaptive Cognitive Memory Architecture API with Working Memory, Safety Gate, NLI, and Decision Policy.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/memories")
def create_memory(request: MemoryEventRequest) -> dict[str, Any]:
    try:
        memory = understanding.from_structured_event(
            request.model_dump(exclude_none=True)
        )
        store.save_memory(memory)
        return {"stored": True, "memory": memory.to_dict()}
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/memories")
def list_memories() -> dict[str, Any]:
    memories = store.get_active_memories()
    return {
        "count": len(memories),
        "memories": [memory.to_dict() for memory in memories],
    }


@app.post("/memories/evaluate")
def evaluate_memory(request: EvaluationRequest) -> dict[str, Any]:
    try:
        new_memory = understanding.from_structured_event(
            request.memory.model_dump(exclude_none=True)
        )
        candidates = retriever.retrieve(
            new_memory,
            top_k=request.top_k,
        )
        results = pipeline.evaluate_candidates(
            new_memory=new_memory,
            candidates=candidates,
        )
        for result in results:
            metrics.record(result)

        event_decision = pipeline.aggregate_event_decision(results)

        return {
            "new_memory": new_memory.to_dict(),
            "candidate_count": len(results),
            "results": results,
            "event_decision": event_decision,
            "metrics": metrics.snapshot(),
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/memories/process")
def process_memory_event(request: ProcessEventRequest) -> dict[str, Any]:
    """
    End-to-end event execution through Working Memory, Retrieval, Safety Gate,
    NLI, Decision Policy, and Memory Evolution storage.
    """
    try:
        new_memory = understanding.from_structured_event(
            request.memory.model_dump(exclude_none=True)
        )
        result = pipeline.process_event(
            event=new_memory,
            store=store,
            user_id=request.user_id,
            session_id=request.session_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            apply_storage=request.apply_storage,
        )
        for cand in result.get("candidate_evaluations", []):
            metrics.record(cand)

        return result
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/working-memory/{user_id}/{session_id}")
def get_working_memory(user_id: str, session_id: str) -> dict[str, Any]:
    memories = working_memory.get_recent_memories(user_id=user_id, session_id=session_id)
    task_ctx = working_memory.get_task_context(user_id=user_id, session_id=session_id)
    return {
        "user_id": user_id,
        "session_id": session_id,
        "item_count": len(memories),
        "recent_memories": [m.to_dict() for m in memories],
        "task_context": task_ctx,
    }


@app.post("/task-context/{user_id}/{session_id}")
def set_task_context(
    user_id: str, session_id: str, request: TaskContextRequest
) -> dict[str, Any]:
    working_memory.set_task_context(
        key=request.key,
        value=request.value,
        user_id=user_id,
        session_id=session_id,
    )
    return {
        "user_id": user_id,
        "session_id": session_id,
        "task_context": working_memory.get_task_context(user_id=user_id, session_id=session_id),
    }


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    return metrics.snapshot()