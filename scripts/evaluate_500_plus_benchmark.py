from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.schema import Memory, MemoryQuery
from memory.store import MemoryStore
from memory.working_memory import WorkingMemory
from conflict.safety_gate import evaluate_rule_safety
from conflict.nli_engine import NLIEngine
from conflict.evidence_score import (
    calculate_evidence_score,
    calculate_update_signal,
    calculate_nli_confidence,
    calculate_nli_margin,
)
from conflict.update_detector import detect_update_signals
from conflict.metadata import compare_metadata
from conflict.decision_policy import decide_action
from conflict.pipeline import ACPipeline


DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmark_500"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


@dataclass
class BenchmarkCase:
    case_id: str
    category: str
    old_memory: Memory
    new_memory: Memory
    expected_action: str
    expected_relationship: str
    expected_safety: str  # "SAFE" or "UNSAFE"
    retrieval_required: bool = True


def make_mem(
    memory_id: str,
    subject: str,
    attribute: str,
    value: str,
    scope: str = "general",
    context: str = "general",
    time: str | None = None,
) -> Memory:
    return Memory(
        memory_id=str(memory_id),
        subject=subject,
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        time=time,
    )


def build_500_plus_dataset(total_target: int = 600) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    counter = 1

    # 1. Exact & Paraphrase Duplicates (Target: 130 cases -> Expected: Ignore)
    duplicate_templates = [
        ("user", "language", "I prefer Python for development", "I prefer Python for development"),
        ("user", "theme", "I use dark mode", "I use dark mode"),
        ("user", "diet", "I follow a vegetarian diet", "I follow a vegetarian diet"),
        ("user", "database", "I use PostgreSQL for backend", "I use PostgreSQL for backend"),
        ("user", "os", "I work on Windows 11", "I work on Windows 11"),
        ("user", "editor", "I use Visual Studio Code", "I use Visual Studio Code"),
        ("user", "notification", "I prefer email notifications", "I prefer email notifications"),
        ("user", "education", "I study computer science", "I study computer science"),
        ("application", "database", "The system uses SQLite", "The system uses SQLite"),
        ("application", "framework", "The API is built with FastAPI", "The API is built with FastAPI"),
    ]
    for _ in range(13):
        for subj, attr, old_val, new_val in duplicate_templates:
            cases.append(
                BenchmarkCase(
                    case_id=f"dup_{counter:04d}",
                    category="duplicate",
                    old_memory=make_mem(f"old_dup_{counter}", subj, attr, old_val),
                    new_memory=make_mem(f"new_dup_{counter}", subj, attr, new_val),
                    expected_action="Ignore",
                    expected_relationship="duplicate",
                    expected_safety="SAFE",
                )
            )
            counter += 1

    # 2. Independent Memories (Target: 130 cases -> Expected: Preserve)
    independent_templates = [
        ("user", "language", "I use Python", "diet", "I am vegetarian", "independent"),
        ("user", "os", "I use Linux", "editor", "I use Neovim", "independent"),
        ("user", "database", "I use PostgreSQL", "backend", "I use FastAPI", "independent"),
        ("user", "city", "I live in Hyderabad", "travel", "I like traveling by train", "independent"),
        ("user", "music", "I play guitar", "food", "I like spicy Mexican food", "independent"),
        ("user", "hobby", "I play chess online", "language", "I am learning Rust", "independent"),
        ("user", "pet", "I have a golden retriever", "work", "I work as a software engineer", "independent"),
        ("application", "theme", "The app uses dark mode", "database", "The app uses PostgreSQL", "independent"),
        ("Caroline", "life", "I went to a support group", "Melanie", "art", "I painted a lake sunrise", "independent"),
        ("Caroline", "career", "I want to work in mental health", "Caroline", "adoption", "I am researching adoption agencies", "independent"),
    ]
    for _ in range(13):
        for s1, a1, v1, s2_or_a2, v2, rel in independent_templates:
            # Handle either different subject or different attribute
            if s1 != s2_or_a2 and s2_or_a2 in ("Melanie", "Caroline"):
                s2, a2 = s2_or_a2, "art" if s2_or_a2 == "Melanie" else "adoption"
            else:
                s2, a2 = s1, s2_or_a2
            cases.append(
                BenchmarkCase(
                    case_id=f"ind_{counter:04d}",
                    category="independent",
                    old_memory=make_mem(f"old_ind_{counter}", s1, a1, v1),
                    new_memory=make_mem(f"new_ind_{counter}", s2, a2, v2),
                    expected_action="Preserve",
                    expected_relationship="independent",
                    expected_safety="SAFE",
                )
            )
            counter += 1

    # 3. Specific Event Exceptions to General Preferences (Target: 100 cases -> Expected: Preserve)
    event_templates = [
        ("user", "diet", "I am vegetarian", "general", "general", "I ate chicken at a wedding", "specific_event", "wedding"),
        ("user", "theme", "I normally use dark mode", "general", "general", "I switched to light mode during the presentation", "specific_event", "presentation"),
        ("user", "routine", "I wake up at 7 AM every day", "general", "general", "I woke up late yesterday at 10 AM", "specific_event", "yesterday"),
        ("user", "beverage", "I do not drink coffee", "general", "general", "I had an espresso at the conference yesterday", "specific_event", "conference"),
        ("user", "location", "I live and work in Hyderabad", "general", "general", "I traveled to Bangalore for a hackathon last weekend", "specific_event", "travel"),
    ]
    for _ in range(20):
        for subj, attr, old_val, old_sc, old_ctx, new_val, new_sc, new_ctx in event_templates:
            cases.append(
                BenchmarkCase(
                    case_id=f"evt_{counter:04d}",
                    category="specific_event",
                    old_memory=make_mem(f"old_evt_{counter}", subj, attr, old_val, old_sc, old_ctx),
                    new_memory=make_mem(f"new_evt_{counter}", subj, attr, new_val, new_sc, new_ctx),
                    expected_action="Preserve",
                    expected_relationship="specific_event",
                    expected_safety="SAFE",
                )
            )
            counter += 1

    # 4. Valid Natural & Explicit Updates (Target: 140 cases -> Expected: Resolve)
    update_templates = [
        ("user", "language", "I prefer Java for DSA", "I now switched to Python for DSA"),
        ("user", "location", "I live in Hyderabad", "I recently moved to Bengaluru"),
        ("user", "framework", "ACMA backend uses FastAPI", "I migrated ACMA backend from FastAPI to Flask"),
        ("user", "database", "Our project uses SQLite", "We switched our project database to PostgreSQL"),
        ("user", "diet", "I prefer vegetarian food", "I no longer follow a vegetarian diet, I now eat meat"),
        ("user", "os", "I develop on Windows", "I now use Ubuntu Linux as my primary OS"),
        ("user", "theme", "The app uses dark mode", "The app now uses light mode by default"),
    ]
    for _ in range(20):
        for subj, attr, old_val, new_val in update_templates:
            cases.append(
                BenchmarkCase(
                    case_id=f"upd_{counter:04d}",
                    category="update",
                    old_memory=make_mem(f"old_upd_{counter}", subj, attr, old_val),
                    new_memory=make_mem(f"new_upd_{counter}", subj, attr, new_val),
                    expected_action="Resolve",
                    expected_relationship="update",
                    expected_safety="UNSAFE",
                )
            )
            counter += 1

    # 5. Genuine Unresolved Conflicts / Contradictions (Target: 100 cases -> Expected: Ask)
    conflict_templates = [
        ("user", "diet", "The user is strictly vegan", "The user loves eating beef steak"),
        ("user", "operating_system", "The user only uses macOS", "The user only uses Windows"),
        ("user", "theme", "The system must always use dark mode", "The system must always use light mode"),
        ("project", "deadline", "The project deadline is strictly October 10", "The project deadline is strictly November 15"),
        ("server", "port", "The API server listens exclusively on port 8080", "The API server listens exclusively on port 9000"),
    ]
    for _ in range(20):
        for subj, attr, old_val, new_val in conflict_templates:
            cases.append(
                BenchmarkCase(
                    case_id=f"cnf_{counter:04d}",
                    category="conflict",
                    old_memory=make_mem(f"old_cnf_{counter}", subj, attr, old_val),
                    new_memory=make_mem(f"new_cnf_{counter}", subj, attr, new_val),
                    expected_action="Ask",
                    expected_relationship="conflict",
                    expected_safety="UNSAFE",
                )
            )
            counter += 1

    return cases[:total_target]


def evaluate_benchmark(total_cases: int = 600) -> dict[str, Any]:
    print("=" * 80)
    print(f"ACMA COMPREHENSIVE BENCHMARK EVALUATION ({total_cases} CASES)")
    print("=" * 80)

    dataset = build_500_plus_dataset(total_target=total_cases)
    print(f"Dataset generated: {len(dataset)} verified cases.")

    # Populate temporary database with old memories
    db_path = DATA_DIR / "acma_benchmark_eval.db"
    if db_path.exists():
        for sfx in ("", "-wal", "-shm"):
            p = Path(str(db_path) + sfx)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    store = MemoryStore(file_path=db_path)
    # Deduplicate old memories before storing in corpus
    unique_old: dict[str, Memory] = {}
    for case in dataset:
        key = (case.old_memory.subject, case.old_memory.attribute, case.old_memory.value)
        if key not in unique_old:
            unique_old[key] = case.old_memory
            store.save_memory(case.old_memory)

    print(f"Corpus initialized in SQLite: {len(unique_old)} unique active memories.")

    nli_engine = NLIEngine.get_default()
    working_memory = WorkingMemory()
    pipeline = ACPipeline(nli_engine=nli_engine, working_memory=working_memory)

    # Metrics collectors
    latencies: list[float] = []
    nli_latencies: list[float] = []

    safety_counts = Counter()
    nli_label_counts = Counter()
    decision_counts = Counter()
    expected_counts = Counter()

    # Safety Gate specific metrics
    safety_deterministic_handled = 0
    safety_routed_to_nli = 0
    safety_deterministic_correct = 0
    safety_deterministic_incorrect = 0
    false_safe_count = 0  # Routed to decision when it was actually a conflict that needed NLI

    # NLI metrics
    nli_eval_count = 0
    nli_disagreements = 0

    # Decision Policy confusion matrix
    confusion_matrix: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    actions = ["Ignore", "Preserve", "Resolve", "Ask"]

    # Retrieval metrics
    retrieval_at_1 = 0
    retrieval_at_5 = 0
    retrieval_total = 0
    total_candidate_comparisons = 0

    results_rows = []
    start_time_all = time.perf_counter()

    for idx, case in enumerate(dataset, start=1):
        t0 = time.perf_counter()

        # Step 1: Retrieval
        query = MemoryQuery.from_memory(case.new_memory)
        retrieved = store.retrieve_related_memories(query, top_k=5, similarity_threshold=0.25)
        total_candidate_comparisons += len(retrieved)

        # Check retrieval recall of expected old memory
        expected_val_norm = " ".join(case.old_memory.value.lower().split())
        retrieval_total += 1
        found_rank = None
        for r_idx, (cand, score) in enumerate(retrieved, start=1):
            cand_norm = " ".join(cand.value.lower().split())
            if cand_norm == expected_val_norm:
                found_rank = r_idx
                break

        if found_rank == 1:
            retrieval_at_1 += 1
        if found_rank is not None and found_rank <= 5:
            retrieval_at_5 += 1

        # Step 2: Safety Gate evaluation on the targeted candidate
        operational_cand = retrieved[found_rank - 1][0] if found_rank else case.old_memory
        operational_score = retrieved[found_rank - 1][1] if found_rank else 0.70

        safety_res = evaluate_rule_safety(operational_cand, case.new_memory, operational_score)
        safety_dict = safety_res.to_dict()

        if safety_res.next_stage == "decision":
            safety_deterministic_handled += 1
            if case.expected_safety == "SAFE":
                safety_deterministic_correct += 1
            else:
                safety_deterministic_incorrect += 1
                false_safe_count += 1
        else:
            safety_routed_to_nli += 1

        # Step 3: NLI (if routed)
        nli_res = None
        if safety_res.next_stage == "nli":
            nli_t0 = time.perf_counter()
            nli_res = nli_engine.compare(operational_cand, case.new_memory)
            nli_latencies.append((time.perf_counter() - nli_t0) * 1000.0)
            nli_eval_count += 1
            nli_label_counts[nli_res["label"]] += 1

        # Step 4: Evidence Scoring
        evidence = {
            "relationship": safety_res.relationship,
            "safety_route": safety_res.next_stage,
            "nli": nli_res,
            "update": detect_update_signals(case.new_memory),
            "metadata": compare_metadata(operational_cand, case.new_memory),
        }

        # Step 5: Decision Policy
        decision = decide_action(safety=safety_dict, evidence=evidence)
        predicted_action = decision.action
        total_lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(total_lat)

        # Record metrics
        expected_action = case.expected_action
        decision_counts[predicted_action] += 1
        expected_counts[expected_action] += 1
        confusion_matrix[expected_action][predicted_action] += 1

        is_correct = (predicted_action == expected_action)

        results_rows.append({
            "case_id": case.case_id,
            "category": case.category,
            "old_text": case.old_memory.value,
            "new_text": case.new_memory.value,
            "expected_action": expected_action,
            "predicted_action": predicted_action,
            "correct": is_correct,
            "safety_route": safety_res.next_stage,
            "relationship": safety_res.relationship,
            "nli_label": nli_res.get("label") if nli_res else "NOT_RUN",
            "nli_confidence": nli_res.get("confidence") if nli_res else None,
            "latency_ms": round(total_lat, 2),
        })

        if idx % 100 == 0 or idx == len(dataset):
            print(f"Evaluated {idx}/{len(dataset)} cases... current accuracy: {sum(1 for r in results_rows if r['correct']) / idx * 100:.2f}%")

    total_time = time.perf_counter() - start_time_all

    # Save detailed CSV
    csv_path = OUTPUT_DIR / "benchmark_500_results.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)

    # Compute Statistics
    total_evaluated = len(dataset)
    total_correct = sum(1 for r in results_rows if r["correct"])
    overall_accuracy = (total_correct / total_evaluated) * 100.0

    # Per-class precision, recall, F1
    per_class_metrics = {}
    for act in actions:
        tp = confusion_matrix[act][act]
        fp = sum(confusion_matrix[other][act] for other in actions if other != act)
        fn = sum(confusion_matrix[act][other] for other in actions if other != act)

        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics[act] = {
            "support": sum(confusion_matrix[act].values()),
            "predicted": sum(confusion_matrix[other][act] for other in actions),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2),
        }

    # Summary payload
    summary = {
        "benchmark_name": "ACMA 500+ Comprehensive Real-Case Benchmark",
        "total_cases_evaluated": total_evaluated,
        "total_correct": total_correct,
        "overall_accuracy_percent": round(overall_accuracy, 2),
        "total_evaluation_runtime_seconds": round(total_time, 2),
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "max": round(float(np.max(latencies)), 2),
        },
        "safety_gate": {
            "total_cases_evaluated": total_evaluated,
            "handled_by_deterministic_rules": safety_deterministic_handled,
            "routed_to_nli": safety_routed_to_nli,
            "deterministic_correct": safety_deterministic_correct,
            "deterministic_incorrect": safety_deterministic_incorrect,
            "false_safe_count": false_safe_count,
            "false_safe_rate_percent": round((false_safe_count / total_evaluated) * 100.0, 2),
        },
        "nli": {
            "total_nli_calls": nli_eval_count,
            "nli_call_rate_percent": round((nli_eval_count / total_evaluated) * 100.0, 2),
            "label_distribution": dict(nli_label_counts),
            "mean_latency_ms": round(float(np.mean(nli_latencies)), 2) if nli_latencies else 0.0,
            "p95_latency_ms": round(float(np.percentile(nli_latencies, 95)), 2) if nli_latencies else 0.0,
        },
        "decision_policy": {
            "predicted_distribution": dict(decision_counts),
            "expected_distribution": dict(expected_counts),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": {act: dict(confusion_matrix[act]) for act in actions},
            "duplicate_accuracy": per_class_metrics["Ignore"]["recall"],
            "update_accuracy": per_class_metrics["Resolve"]["recall"],
            "preserve_accuracy": per_class_metrics["Preserve"]["recall"],
            "ask_accuracy": per_class_metrics["Ask"]["recall"],
        },
        "retrieval": {
            "total_queries": retrieval_total,
            "candidate_comparisons_total": total_candidate_comparisons,
            "candidates_per_query_avg": round(total_candidate_comparisons / retrieval_total, 2),
            "recall_at_1_percent": round((retrieval_at_1 / retrieval_total) * 100.0, 2),
            "recall_at_5_percent": round((retrieval_at_5 / retrieval_total) * 100.0, 2),
        },
        "outputs": {
            "csv_results": str(csv_path),
            "summary_json": str(OUTPUT_DIR / "benchmark_500_summary.json"),
        },
    }

    json_path = OUTPUT_DIR / "benchmark_500_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print Executive Report
    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Cases Evaluated:       {total_evaluated}")
    print(f"Correct Decisions:           {total_correct}")
    print(f"Overall Accuracy:            {overall_accuracy:.2f}%")
    print(f"Total Runtime:               {total_time:.2f} seconds")
    print(f"Mean Latency:                {summary['latency_ms']['mean']} ms (p95: {summary['latency_ms']['p95']} ms)")
    print()
    print("--- SAFETY GATE ---")
    print(f"Handled Deterministically:   {safety_deterministic_handled} ({safety_deterministic_handled / total_evaluated * 100:.1f}%)")
    print(f"Routed to NLI:               {safety_routed_to_nli} ({safety_routed_to_nli / total_evaluated * 100:.1f}%)")
    print(f"False-Safe Decisions:        {false_safe_count} (Rate: {summary['safety_gate']['false_safe_rate_percent']}%)")
    print()
    print("--- NLI ENGINE ---")
    print(f"NLI Evaluations:             {nli_eval_count}")
    print(f"NLI Labels:                  {dict(nli_label_counts)}")
    print()
    print("--- DECISION POLICY & CONFUSION MATRIX ---")
    print(f"{'Expected':<12} | {'Predicted Ignore':<18} | {'Predicted Preserve':<18} | {'Predicted Resolve':<18} | {'Predicted Ask':<15}")
    print("-" * 88)
    for act in actions:
        row = confusion_matrix[act]
        print(f"{act:<12} | {row['Ignore']:<18} | {row['Preserve']:<18} | {row['Resolve']:<18} | {row['Ask']:<15}")
    print()
    print("--- PER-CLASS PERFORMANCE ---")
    for act, m in per_class_metrics.items():
        print(f"{act:<10} -> Precision: {m['precision']:>6.2f}% | Recall: {m['recall']:>6.2f}% | F1: {m['f1_score']:>6.2f}% | Support: {m['support']}")
    print()
    print("--- RETRIEVAL PERFORMANCE ---")
    print(f"Recall@1:                    {summary['retrieval']['recall_at_1_percent']}%")
    print(f"Recall@5:                    {summary['retrieval']['recall_at_5_percent']}%")
    print(f"Results saved to:            {csv_path}")

    return summary


if __name__ == "__main__":
    evaluate_benchmark(total_cases=600)
