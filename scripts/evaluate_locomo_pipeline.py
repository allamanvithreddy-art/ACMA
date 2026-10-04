from __future__ import annotations

import csv
import json
import sys
import tempfile
import traceback

from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "locomo_evaluation"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ACMA IMPORTS
# ============================================================

from memory.schema import Memory, MemoryQuery
from memory.store import MemoryStore
from conflict.pipeline import ACPipeline


# ============================================================
# FILE UTILITIES
# ============================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8", newline="") as file:
            file.write("")
        return

    columns = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row.keys()
        )
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            converted = {}

            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    converted[key] = json.dumps(
                        value,
                        ensure_ascii=False,
                        default=str,
                    )
                else:
                    converted[key] = value


            writer.writerow(converted)


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOCOMO DIALOGUE PROCESSING
# ============================================================

def build_dialogue_index(
    conversation: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], int]:
    dialogue_index = {}
    total_turns = 0

    sessions = conversation.get("sessions", [])

    for session_index, session in enumerate(sessions):
        dialogues = session.get("dialogues", [])

        for turn_index, dialogue in enumerate(dialogues):
            total_turns += 1

            dialogue_id = dialogue.get("dia_id")

            if not dialogue_id:
                continue

            dialogue_index[str(dialogue_id)] = {
                "text": str(dialogue.get("text", "")),
                "speaker": str(dialogue.get("speaker", "")),
                "session_index": session_index,
                "turn_index": turn_index,
                "session_id": session.get(
                    "session_id",
                    f"session_{session_index + 1}",
                ),
            }

    return dialogue_index, total_turns


def sort_memory_key(
    record: dict[str, Any],
    dialogue_index: dict[str, dict[str, Any]],
) -> tuple[int, int, str]:
    source_id = str(record.get("source_dia_id", ""))

    source = dialogue_index.get(source_id)

    if source is not None:
        return (
            int(source["session_index"]),
            int(source["turn_index"]),
            str(record.get("memory_id", "")),
        )

    return (
        10**9,
        10**9,
        str(record.get("memory_id", "")),
    )


def prepare_locomo_memories(
    conversation: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    dialogue_index, _ = build_dialogue_index(conversation)

    raw_memories = metadata.get("memories")

    if not isinstance(raw_memories, list):
        raise ValueError(
            "locomo_metadata.json must contain a 'memories' list."
        )

    ordered_memories = sorted(
        raw_memories,
        key=lambda item: sort_memory_key(
            item,
            dialogue_index,
        ),
    )

    prepared = []

    for index, record in enumerate(ordered_memories, start=1):
        source_id = str(record.get("source_dia_id", ""))

        source_dialogue = dialogue_index.get(source_id, {})

        source_text = str(
            source_dialogue.get("text", "")
        ).strip()

        curated_value = str(
            record.get("value", "")
        ).strip()

        text = source_text or curated_value

        if not text:
            continue

        memory_metadata = dict(record.get("metadata") or {})

        memory_metadata.update(
            {
                "source_dia_id": source_id,
                "speaker": record.get(
                    "speaker",
                    source_dialogue.get("speaker", ""),
                ),
                "conversation_id": conversation.get(
                    "conversation_id",
                    "",
                ),
                "curated_value": curated_value,
            }
        )

        prepared.append(
            {
                "memory_id": str(
                    record.get(
                        "memory_id",
                        f"locomo_m{index:03d}",
                    )
                ),
                "text": text,
                "subject": str(record.get("subject", "")),
                "attribute": str(record.get("attribute", "")),
                "value": curated_value or text,
                "scope": str(record.get("scope", "")),
                "context": str(record.get("context", "")),
                "time": record.get("time"),
                "source": str(record.get("source", "locomo")),
                "metadata": memory_metadata,
                "status": "active",
                "version": 1,
                "supersedes": [],
                "superseded_by": None,
            }
        )

    return prepared


# ============================================================
# MEMORY AND RETRIEVAL
# ============================================================

def create_memory(record: dict[str, Any]) -> Memory:
    return Memory.from_dict(record)


def create_query(memory: Memory) -> MemoryQuery:
    return MemoryQuery(
        text=memory.text,
        subject=memory.subject,
        attribute=memory.attribute,
        value=memory.value,
        scope=memory.scope,
        context=memory.context,
        time=memory.time,
    )


def create_temporary_database() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".db",
        prefix="acma_locomo_",
        dir=DATA_DIR,
        delete=False,
    ) as temp_file:
        return Path(temp_file.name)


def cleanup_database(path: Path | None) -> None:
    if path is None:
        return

    import gc
    gc.collect()

    for suffix in ("", "-wal", "-shm", "-journal"):
        target = Path(str(path) + suffix)

        try:
            if target.exists():
                target.unlink()
        except OSError as exc:
            pass


# ============================================================
# PIPELINE RESULT EXTRACTION
# ============================================================

def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_action(result: dict[str, Any]) -> str:
    decision = as_dict(result.get("decision"))

    action = decision.get("action")

    if action:
        return str(action)

    # Support a pipeline that returns action at the top level.
    action = result.get("action")

    if action:
        return str(action)

    return "Ask"


def extract_safety(result: dict[str, Any]) -> dict[str, Any]:
    return as_dict(result.get("safety"))


def extract_relationship(result: dict[str, Any]) -> str:
    safety = extract_safety(result)

    if safety.get("relationship"):
        return str(safety["relationship"])

    relationship = result.get("relationship")

    if isinstance(relationship, dict):
        return str(
            relationship.get(
                "relationship",
                "unknown",
            )
        )

    if relationship:
        return str(relationship)

    return "unknown"


def extract_nli(result: dict[str, Any]) -> dict[str, Any]:
    return as_dict(result.get("nli"))


def extract_reason(result: dict[str, Any]) -> str:
    decision = as_dict(result.get("decision"))

    reason = decision.get("reason")

    if reason:
        return str(reason)

    reason = result.get("reason")

    if reason:
        return str(reason)

    return ""


# ============================================================
# LOCOMO EVALUATION
# ============================================================

def run_locomo_evaluation() -> dict[str, Any]:
    conversation_path = DATA_DIR / "locomo_sample.json"
    metadata_path = DATA_DIR / "locomo_metadata.json"

    conversation = load_json(conversation_path)
    metadata = load_json(metadata_path)

    if not isinstance(conversation, dict):
        raise ValueError(
            "locomo_sample.json must contain a JSON object."
        )

    if not isinstance(metadata, dict):
        raise ValueError(
            "locomo_metadata.json must contain a JSON object."
        )

    memories = prepare_locomo_memories(
        conversation,
        metadata,
    )

    if not memories:
        raise ValueError(
            "No usable curated memories were found."
        )

    _, dialogue_turn_count = build_dialogue_index(conversation)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_section("ACMA LoCoMo Replay")

    print(
        f"Conversation: "
        f"{conversation.get('conversation_id', 'unknown')}"
    )
    print(
        f"Dialogue sessions: "
        f"{len(conversation.get('sessions', []))}"
    )
    print(f"Dialogue turns: {dialogue_turn_count}")
    print(f"Curated memories: {len(memories)}")
    print("Database: temporary SQLite")
    print("Evaluation: retrieval through Decision Policy")

    database_path = None
    store = None
    pipeline = None

    event_rows = []
    candidate_rows = []

    event_actions = Counter()
    candidate_actions = Counter()
    safety_routes = Counter()
    relationships = Counter()
    nli_labels = Counter()

    events_with_retrieval = 0
    events_without_retrieval = 0
    retrieved_candidate_count = 0
    nli_call_count = 0

    try:
        database_path = create_temporary_database()

        store = MemoryStore(
            file_path=database_path,
        )

        pipeline = ACPipeline()

        for index, record in enumerate(memories, start=1):
            new_memory = create_memory(record)
            query = create_query(new_memory)

            # Retrieve only memories stored during earlier iterations.
            retrieved = store.retrieve_related_memories(
                query=query,
                top_k=5,
                similarity_threshold=0.30,
            )

            retrieved_candidate_count += len(retrieved)

            print()
            print(
                f"[{index}/{len(memories)}] "
                f"{new_memory.memory_id}"
            )
            print(f"  Text: {new_memory.text[:120]}")
            print(f"  Retrieved candidates: {len(retrieved)}")

            if not retrieved:
                events_without_retrieval += 1

                action = "Store"
                reason = (
                    "No prior memories passed the retrieval threshold."
                )

                event_actions[action] += 1

                print("  Decision: Store")

                event_rows.append(
                    {
                        "event_index": index,
                        "memory_id": new_memory.memory_id,
                        "source_dia_id": new_memory.metadata.get(
                            "source_dia_id"
                        ),
                        "text": new_memory.text,
                        "retrieved_count": 0,
                        "top_candidate_id": None,
                        "top_retrieval_score": None,
                        "safety_route": None,
                        "relationship": None,
                        "nli_label": None,
                        "decision_action": action,
                        "decision_reason": reason,
                    }
                )

                # Save after evaluation so the current memory cannot
                # be retrieved as its own previous memory.
                store.save_memory(new_memory)
                continue

            events_with_retrieval += 1

            # ACPipeline.evaluate_candidates() returns one result
            # dictionary per retrieved (old_memory, retrieval_score) pair.
            candidate_results = pipeline.evaluate_candidates(
                new_memory=new_memory,
                candidates=retrieved,
            )

            if not isinstance(candidate_results, list):
                raise TypeError(
                    "ACPipeline.evaluate_candidates() must return a list."
                )

            normalized_results = [
                item for item in candidate_results
                if isinstance(item, dict)
            ]

            if len(normalized_results) != len(retrieved):
                raise RuntimeError(
                    "Pipeline result count does not match retrieved "
                    f"candidate count ({len(normalized_results)} vs "
                    f"{len(retrieved)})."
                )

            for candidate_index, candidate_result in enumerate(
                normalized_results
            ):
                action_for_candidate = extract_action(candidate_result)
                safety = extract_safety(candidate_result)
                nli = extract_nli(candidate_result)
                relationship = extract_relationship(candidate_result)

                candidate_actions[action_for_candidate] += 1

                route = str(
                    safety.get("next_stage", "unknown")
                )
                safety_routes[route] += 1
                relationships[relationship] += 1

                if nli:
                    nli_call_count += 1
                    nli_labels[
                        str(nli.get("label", "unknown"))
                    ] += 1

                # Associate each candidate result with its corresponding
                # retrieved memory when the pipeline returns per-candidate
                # results in the same order.
                retrieved_index = min(
                    candidate_index,
                    len(retrieved) - 1,
                )

                old_memory, retrieval_score = retrieved[retrieved_index]

                candidate_rows.append(
                    {
                        "event_index": index,
                        "new_memory_id": new_memory.memory_id,
                        "old_memory_id": old_memory.memory_id,
                        "new_memory_text": new_memory.text,
                        "old_memory_text": old_memory.text,
                        "retrieval_score": retrieval_score,
                        "safety_safe": safety.get("safe"),
                        "safety_route": safety.get("next_stage"),
                        "safety_reason": safety.get("reason"),
                        "relationship": relationship,
                        "nli_was_run": bool(nli),
                        "nli_label": nli.get("label"),
                        "nli_confidence": nli.get("confidence"),
                        "decision_action": action_for_candidate,
                        "decision_reason": extract_reason(
                            candidate_result
                        ),
                    }
                )

            # ACPipeline currently returns per-candidate decisions, not a
            # separate event-level decision. Use the top retrieved candidate
            # as the event-level summary and retain all candidate decisions
            # separately in candidate_rows.
            action = extract_action(normalized_results[0])

            event_actions[action] += 1

            top_memory, top_score = retrieved[0]

            top_result = normalized_results[0]
            top_safety = extract_safety(top_result)
            top_nli = extract_nli(top_result)
            top_relationship = extract_relationship(top_result)

            event_rows.append(
                {
                    "event_index": index,
                    "memory_id": new_memory.memory_id,
                    "source_dia_id": new_memory.metadata.get(
                        "source_dia_id"
                    ),
                    "text": new_memory.text,
                    "retrieved_count": len(retrieved),
                    "top_candidate_id": top_memory.memory_id,
                    "top_retrieval_score": top_score,
                    "safety_route": top_safety.get("next_stage"),
                    "relationship": top_relationship,
                    "nli_label": top_nli.get("label"),
                    "nli_confidence": top_nli.get("confidence"),
                    "decision_action": action,
                    "decision_reason": extract_reason(top_result),
                }
            )

            print(f"  Top candidate: {top_memory.memory_id}")
            print(f"  Retrieval score: {top_score:.4f}")
            print(
                f"  Safety route: "
                f"{top_safety.get('next_stage', 'unknown')}"
            )
            print(f"  Relationship: {top_relationship}")

            # FIXED: this conditional is now valid Python syntax.
            nli_label_display = (
                top_nli.get("label", "not run")
                if top_nli
                else "not run"
            )
            print(f"  NLI: {nli_label_display}")
            print(f"  Decision: {action}")

            # Store the incoming memory only after evaluation.
            store.save_memory(new_memory)

    finally:
        store = None
        pipeline = None
        cleanup_database(database_path)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    event_csv = OUTPUT_DIR / "locomo_event_results.csv"
    candidate_csv = OUTPUT_DIR / "locomo_candidate_results.csv"
    summary_json = OUTPUT_DIR / "locomo_summary.json"

    write_csv(event_csv, event_rows)
    write_csv(candidate_csv, candidate_rows)

    summary = {
        "experiment": "ACMA LoCoMo chronological replay",
        "conversation_id": conversation.get("conversation_id"),
        "dialogue_sessions": len(conversation.get("sessions", [])),
        "dialogue_turns": dialogue_turn_count,
        "curated_memories": len(memories),
        "events_evaluated": len(event_rows),
        "events_with_retrieval": events_with_retrieval,
        "events_without_retrieval": events_without_retrieval,
        "retrieved_candidates_total": retrieved_candidate_count,
        "candidate_evaluations": len(candidate_rows),
        "nli_calls": nli_call_count,
        "nli_labels": dict(nli_labels),
        "event_level_actions": dict(event_actions),
        "candidate_level_actions": dict(candidate_actions),
        "safety_gate_routes": dict(safety_routes),
        "relationships": dict(relationships),
        "limitations": [
            (
                "This replay evaluates the curated memory subset, "
                "not the complete LoCoMo benchmark."
            ),
            (
                "No ground-truth action accuracy is reported because "
                "expected actions are not provided for each replay event."
            ),
            (
                "Event-level and candidate-level decisions are reported "
                "separately."
            ),
        ],
        "outputs": {
            "event_results": str(event_csv),
            "candidate_results": str(candidate_csv),
            "summary": str(summary_json),
        },
    }

    save_json(summary_json, summary)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print_section("LoCoMo Replay Summary")

    print(f"Events evaluated: {len(event_rows)}")
    print(f"Events with retrieval: {events_with_retrieval}")
    print(f"Events without retrieval: {events_without_retrieval}")
    print(f"Candidate evaluations: {len(candidate_rows)}")
    print(f"Retrieved candidates: {retrieved_candidate_count}")
    print(f"NLI calls: {nli_call_count}")

    print()
    print("Event-level actions:")

    for action, count in sorted(event_actions.items()):
        print(f"  {action}: {count}")

    print()
    print("Candidate-level actions:")

    for action, count in sorted(candidate_actions.items()):
        print(f"  {action}: {count}")

    print()
    print("Safety Gate routes:")

    for route, count in sorted(safety_routes.items()):
        print(f"  {route}: {count}")

    print()
    print("NLI labels:")

    for label, count in sorted(nli_labels.items()):
        print(f"  {label}: {count}")

    print()
    print("Output files:")

    print(f"  {event_csv}")
    print(f"  {candidate_csv}")
    print(f"  {summary_json}")

    return summary


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    try:
        run_locomo_evaluation()

    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
        raise SystemExit(130)

    except Exception as exc:
        print_section("EVALUATION FAILED")
        print(f"{type(exc).__name__}: {exc}")
        print()
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
