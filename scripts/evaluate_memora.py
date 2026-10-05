from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.schema import Memory, MemoryQuery
from memory.store import MemoryStore
from conflict.nli_engine import NLIEngine
from conflict.pipeline import ACPipeline


ROOT = PROJECT_ROOT / "data" / "external" / "memora" / "data" / "quarterly"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "memora_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Case:
    case_id: str
    persona: str
    session_id: int
    update_type: str
    old_memory: Memory
    new_memory: Memory
    expected_action: str


def clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def conversation_memory_text(session: dict[str, Any]) -> str:
    parts = []

    for turn in session.get("conversation", []):
        if turn.get("share_memory") is True:
            message = clean(turn.get("message"))
            if message:
                parts.append(message)

    if parts:
        return " ".join(parts)

    for turn in session.get("conversation", []):
        speaker = clean(turn.get("speaker")).lower()
        if speaker.startswith("user"):
            message = clean(turn.get("message"))
            if message:
                parts.append(message)

    return " ".join(parts)


def make_memory(
    memory_id: str,
    persona: str,
    attribute: str,
    value: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        user_id=persona,
        text=text,
        subject=persona,
        attribute=attribute,
        value=value,
        scope="general",
        context="general",
        time=None,
        source="memora",
        confidence=1.0,
        importance=0.7,
        metadata=metadata or {},
    )


def get_attribute(details: dict[str, Any], session: dict[str, Any]) -> str:
    category = clean(details.get("category") or session.get("category"))
    subcategory = clean(details.get("subcategory"))

    if category and subcategory:
        return f"{category}:{subcategory}"

    if subcategory:
        return subcategory

    if category:
        return category

    return clean(session.get("session_type")) or "memory"


def build_value_update_case(session: dict[str, Any]) -> Case | None:
    details = session.get("operation_details") or {}

    old_item = details.get("old_item")
    new_item = details.get("item")

    if old_item is None or new_item is None:
        return None

    persona = clean(session.get("persona"))
    session_id = int(session.get("session_id", -1))
    attribute = get_attribute(details, session)
    conversation = conversation_memory_text(session)

    old_id = f"memora-old-{persona}-{session_id}"
    new_id = f"memora-new-{persona}-{session_id}"

    old_value = clean(old_item)
    new_value = clean(new_item)

    old_text = f"My previous choice in {attribute} was {old_value}."

    preference = clean(details.get("preference"))
    if preference:
        old_text = f"I {preference} {old_value}."

    new_text = conversation or (
        f"I changed my choice in {attribute} from "
        f"{old_value} to {new_value}."
    )

    old_memory = make_memory(
        old_id,
        persona,
        attribute,
        old_value,
        old_text,
    )

    new_memory = make_memory(
        new_id,
        persona,
        attribute,
        new_value,
        new_text,
        {
            "update_of": old_id,
            "supersedes": [old_id],
        },
    )

    return Case(
        case_id=f"value_update_{persona}_{session_id}",
        persona=persona,
        session_id=session_id,
        update_type="value_update",
        old_memory=old_memory,
        new_memory=new_memory,
        expected_action="Resolve",
    )


def build_preference_update_case(session: dict[str, Any]) -> Case | None:
    details = session.get("operation_details") or {}

    item = details.get("item")
    old_preference = details.get("old_preference")
    new_preference = details.get("preference")

    if item is None or old_preference is None or new_preference is None:
        return None

    persona = clean(session.get("persona"))
    session_id = int(session.get("session_id", -1))
    attribute = get_attribute(details, session)
    conversation = conversation_memory_text(session)

    old_id = f"memora-old-{persona}-{session_id}"
    new_id = f"memora-new-{persona}-{session_id}"

    old_value = f"{clean(old_preference)}:{clean(item)}"
    new_value = f"{clean(new_preference)}:{clean(item)}"

    old_memory = make_memory(
        old_id,
        persona,
        attribute,
        old_value,
        f"I {old_preference} {item}.",
    )

    new_memory = make_memory(
        new_id,
        persona,
        attribute,
        new_value,
        conversation or f"I now {new_preference} {item}.",
        {
            "update_of": old_id,
            "supersedes": [old_id],
        },
    )

    return Case(
        case_id=f"preference_update_{persona}_{session_id}",
        persona=persona,
        session_id=session_id,
        update_type="preference_update",
        old_memory=old_memory,
        new_memory=new_memory,
        expected_action="Resolve",
    )


def build_list_update_case(session: dict[str, Any]) -> Case | None:
    details = session.get("operation_details") or {}
    memory_updates = details.get("memory_updates")

    if not isinstance(memory_updates, list) or not memory_updates:
        return None

    additions = []

    for change in memory_updates:
        if not isinstance(change, dict):
            continue

        if change.get("action") == "added":
            item = change.get("added_item")
            if item is not None:
                additions.append(str(item))

    if not additions:
        return None

    fields = [
        str(change.get("field"))
        for change in memory_updates
        if isinstance(change, dict) and change.get("field")
    ]

    field = fields[0] if fields else "content"

    persona = clean(session.get("persona"))
    session_id = int(session.get("session_id", -1))
    conversation = conversation_memory_text(session)

    old_id = f"memora-old-{persona}-{session_id}"
    new_id = f"memora-new-{persona}-{session_id}"

    added_text = "; ".join(additions[:10])

    old_memory = make_memory(
        old_id,
        persona,
        f"content:{field}",
        "<existing-content>",
        f"The existing {field} information is stored.",
    )

    new_memory = make_memory(
        new_id,
        persona,
        f"content:{field}",
        f"<existing-content> + {added_text}",
        conversation or f"New items were added to {field}.",
        {},
    )

    return Case(
        case_id=f"list_update_{persona}_{session_id}",
        persona=persona,
        session_id=session_id,
        update_type="redesigned_list_update",
        old_memory=old_memory,
        new_memory=new_memory,
        expected_action="Preserve",
    )


def load_cases(limit: int | None = None) -> tuple[list[Case], Counter[str], int, int, int]:
    operation_counts = Counter()
    cases: list[Case] = []

    add_sessions = 0
    delete_sessions = 0
    skipped_updates = 0

    files = sorted(ROOT.glob("*/conversations/session_*.json"))

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            session = json.load(f)

        operation = session.get("operation")
        operation_counts[str(operation)] += 1

        if operation == "add":
            add_sessions += 1
            continue

        if operation == "delete":
            delete_sessions += 1
            continue

        if operation != "update":
            continue

        details = session.get("operation_details") or {}
        update_type = details.get("update_type")

        case = None

        if update_type == "value_update":
            case = build_value_update_case(session)

        elif update_type == "preference_update":
            case = build_preference_update_case(session)

        elif update_type == "redesigned_list_update":
            case = build_list_update_case(session)

        if case is None:
            skipped_updates += 1
        else:
            cases.append(case)

        if limit is not None and len(cases) >= limit:
            break

    return cases, operation_counts, add_sessions, delete_sessions, skipped_updates


def metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}

    ordered = sorted(values)
    p95_index = min(
        len(ordered) - 1,
        int(0.95 * (len(ordered) - 1)),
    )

    return {
        "mean_ms": round(statistics.fmean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "p95_ms": round(ordered[p95_index], 2),
        "max_ms": round(max(values), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of eligible Memora decision cases to run.",
    )
    args = parser.parse_args()

    cases, operation_counts, add_sessions, delete_sessions, skipped_updates = load_cases(
        args.limit
    )

    print("=" * 90)
    print("ACMA FINAL MEMORA EXTERNAL EVALUATION")
    print("=" * 90)

    print(f"Decision cases:               {len(cases)}")
    print(f"Add sessions observed:        {add_sessions}")
    print(f"Delete sessions observed:     {delete_sessions}")
    print(f"Unsupported updates skipped:  {skipped_updates}")
    print()

    print("Decision ground truth:")
    print("  value_update             -> Resolve")
    print("  preference_update        -> Resolve")
    print("  redesigned_list_update   -> Preserve")
    print()
    print("Delete events are reported separately and are NOT mapped to Resolve.")
    print()

    print("Loading ACMA NLI model...")

    pipeline = ACPipeline(
        nli_engine=NLIEngine.get_default()
    )

    # Real retrieval benchmark store.
    # Each case uses its own user namespace so the previous memory for that
    # case is retrieved by the actual MemoryStore embedding pipeline.
    runtime_store = MemoryStore(
        OUTPUT_DIR / "memora_runtime_eval.db"
    )

    rows: list[dict[str, Any]] = []

    route_counts = Counter()
    route_correct = Counter()
    nli_labels = Counter()

    expected_counts = Counter()
    predicted_counts = Counter()
    confusion = defaultdict(Counter)

    latency_values: list[float] = []

    errors = 0

    for index, case in enumerate(cases, start=1):
        try:
            case_user_id = f"memora:{case.persona}:{case.session_id}"

            case.old_memory.user_id = case_user_id
            case.new_memory.user_id = case_user_id

            runtime_store.save_memory(case.old_memory)

            query = MemoryQuery.from_memory(case.new_memory)

            retrieved = runtime_store.retrieve_related_memories(
                query,
                top_k=5,
                similarity_threshold=0.0,
                user_id=case_user_id,
            )

            if retrieved:
                candidate, retrieval_score = retrieved[0]
                result = pipeline.evaluate_candidate(
                    candidate,
                    case.new_memory,
                    retrieval_score=retrieval_score,
                )
            else:
                result = pipeline.evaluate_candidate(
                    case.old_memory,
                    case.new_memory,
                    retrieval_score=0.0,
                )

            predicted_action = str(result["action"])

            safety = result["safety"]

            if safety["next_stage"] == "decision":
                route = "SAFE->DECISION"
            else:
                route = "UNSAFE->NLI"

            nli = result.get("nli")

            correct = predicted_action == case.expected_action

            route_counts[route] += 1
            route_correct[route] += int(correct)

            expected_counts[case.expected_action] += 1
            predicted_counts[predicted_action] += 1
            confusion[case.expected_action][predicted_action] += 1

            if nli:
                nli_labels[str(nli.get("label", "unknown"))] += 1

            latency = float(result.get("latency_ms") or 0.0)
            latency_values.append(latency)

            rows.append({
                "case_id": case.case_id,
                "persona": case.persona,
                "session_id": case.session_id,
                "update_type": case.update_type,
                "expected_action": case.expected_action,
                "predicted_action": predicted_action,
                "correct": correct,
                "safety_route": route,
                "relationship": result.get("relationship"),
                "nli_label": nli.get("label") if nli else "",
                "nli_confidence": nli.get("confidence") if nli else "",
                "latency_ms": round(latency, 2),
                "reason": result.get("reason", ""),
                "old_memory": case.old_memory.text,
                "new_memory": case.new_memory.text,
            })

        except Exception as exc:
            errors += 1

            rows.append({
                "case_id": case.case_id,
                "persona": case.persona,
                "session_id": case.session_id,
                "update_type": case.update_type,
                "expected_action": case.expected_action,
                "predicted_action": "ERROR",
                "correct": False,
                "safety_route": "",
                "relationship": "",
                "nli_label": "",
                "nli_confidence": "",
                "latency_ms": "",
                "reason": f"{type(exc).__name__}: {exc}",
                "old_memory": case.old_memory.text,
                "new_memory": case.new_memory.text,
            })

        if index % 100 == 0 or index == len(cases):
            good = sum(1 for r in rows if r["correct"])
            accuracy = good / index * 100
            print(
                f"Evaluated {index}/{len(cases)} "
                f"| accuracy so far: {accuracy:.2f}%"
            )

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    accuracy = correct / total * 100 if total else 0.0

    safe_total = route_counts["SAFE->DECISION"]
    safe_correct = route_correct["SAFE->DECISION"]

    nli_total = route_counts["UNSAFE->NLI"]
    nli_correct = route_correct["UNSAFE->NLI"]

    actions = ["Ignore", "Preserve", "Resolve", "Ask"]

    per_action = {}

    for action in actions:
        tp = confusion[action][action]

        fp = sum(
            confusion[other][action]
            for other in actions
            if other != action
        )

        fn = sum(
            confusion[action][other]
            for other in actions
            if other != action
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0
        )

        per_action[action] = {
            "support": sum(confusion[action].values()),
            "precision_percent": round(precision * 100, 2),
            "recall_percent": round(recall * 100, 2),
            "f1_percent": round(f1 * 100, 2),
        }

    by_type = {}

    for update_type in sorted({c.update_type for c in cases}):
        subset = [
            row
            for row in rows
            if row["update_type"] == update_type
        ]

        good = sum(1 for row in subset if row["correct"])

        by_type[update_type] = {
            "cases": len(subset),
            "correct": good,
            "accuracy_percent": round(
                good / len(subset) * 100,
                2,
            ) if subset else 0.0,
        }

    summary = {
        "dataset": {
            "name": "Memora quarterly",
            "total_sessions_seen": sum(operation_counts.values()),
            "operation_counts": dict(operation_counts),
            "decision_cases": total,
            "add_sessions": add_sessions,
            "delete_sessions": delete_sessions,
            "unsupported_updates_skipped": skipped_updates,
            "ground_truth_policy": {
                "value_update": "Resolve",
                "preference_update": "Resolve",
                "redesigned_list_update": "Preserve",
                "delete": "separate lifecycle metric",
            },
        },
        "overall": {
            "correct": correct,
            "errors": errors,
            "accuracy_percent": round(accuracy, 2),
        },
        "safety_gate": {
            "safe_direct_cases": safe_total,
            "safe_direct_correct": safe_correct,
            "safe_direct_accuracy_percent": round(
                safe_correct / safe_total * 100,
                2,
            ) if safe_total else 0.0,
            "coverage_percent": round(
                safe_total / total * 100,
                2,
            ) if total else 0.0,
        },
        "nli_decision_path": {
            "nli_routed_cases": nli_total,
            "nli_route_correct": nli_correct,
            "nli_route_final_accuracy_percent": round(
                nli_correct / nli_total * 100,
                2,
            ) if nli_total else 0.0,
            "nli_calls": sum(nli_labels.values()),
            "labels": dict(nli_labels),
        },
        "decision_policy": {
            "expected": dict(expected_counts),
            "predicted": dict(predicted_counts),
            "per_action": per_action,
            "confusion_matrix": {
                expected_action: dict(
                    confusion[expected_action]
                )
                for expected_action in actions
            },
            "by_update_type": by_type,
        },
        "latency_ms": metric_stats(latency_values),
    }

    csv_path = OUTPUT_DIR / "memora_decision_results.csv"
    json_path = OUTPUT_DIR / "memora_decision_summary.json"

    if rows:
        with csv_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 90)
    print("FINAL MEMORA RESULTS")
    print("=" * 90)

    print(f"Decision cases:               {total}")
    print(f"Correct:                      {correct}")
    print(f"Overall accuracy:             {accuracy:.2f}%")
    print()

    print("SAFETY GATE")
    print(f"SAFE -> Decision:             {safe_total}")
    print(f"SAFE correct:                 {safe_correct}")

    if safe_total:
        print(
            f"SAFE accuracy:                "
            f"{safe_correct / safe_total * 100:.2f}%"
        )

    if total:
        print(
            f"SAFE coverage:                "
            f"{safe_total / total * 100:.2f}%"
        )

    print()

    print("NLI -> DECISION")
    print(f"NLI routed:                   {nli_total}")
    print(f"NLI calls:                    {sum(nli_labels.values())}")

    if nli_total:
        print(
            f"NLI-route final accuracy:     "
            f"{nli_correct / nli_total * 100:.2f}%"
        )

    print()

    print("BY UPDATE TYPE")

    for name, values in by_type.items():
        print(
            f"  {name:27s} "
            f"{values['correct']}/{values['cases']} "
            f"= {values['accuracy_percent']:.2f}%"
        )

    print()

    print(f"Delete events observed:      {delete_sessions}")

    print()
    print("OUTPUT FILES")
    print(f"CSV:                         {csv_path}")
    print(f"JSON:                        {json_path}")


if __name__ == "__main__":
    main()
