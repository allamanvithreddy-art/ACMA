import json
import random
from dataclasses import dataclass
from pathlib import Path

from memory.schema import Memory, MemoryQuery
from memory.store import MemoryStore

from conflict.safety_gate import evaluate_rule_safety
from conflict.evidence_score import (
    calculate_evidence_score,
    calculate_update_signal,
)
from conflict.update_detector import detect_update_signals
from conflict.metadata import metadata_agreement
from conflict.nli_engine import NLIEngine
from conflict.decision_policy import decide_action


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
TOTAL_CASES = 600

TOP_K = 5
RETRIEVAL_DIAGNOSTIC_K = 10

SIMILARITY_THRESHOLD = 0.30

BENCHMARK_MEMORY_FILE = (
    "data/full_pipeline_benchmark_memories.json"
)

# Number of actual RAG examples to display.
RAG_EXAMPLES_TO_SHOW = 15

random.seed(RANDOM_SEED)


# ============================================================
# BENCHMARK CASE
# ============================================================

@dataclass
class BenchmarkCase:

    case_id: str

    category: str

    old_memory: Memory

    new_memory: Memory

    expected_action: str

    retrieval_required: bool = True


# ============================================================
# MEMORY CREATION
# ============================================================

def make_memory(
    memory_id,
    subject,
    attribute,
    value,
    scope="general",
    context="general",
    time=None,
    confidence=0.9,
    importance=0.7,
):

    return Memory(
        memory_id=str(memory_id),
        subject=subject,
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        time=time,
        confidence=confidence,
        importance=importance,
        active=True,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    return " ".join(
        value.lower().strip().split()
    )


# ============================================================
# MEMORY KEY
# ============================================================

def memory_key(memory):

    return (
        memory.subject.strip().lower(),
        memory.attribute.strip().lower(),
        memory.value.strip().lower(),
        memory.scope.strip().lower(),
        memory.context.strip().lower(),
        memory.time,
    )


# ============================================================
# BUILD BENCHMARK
# ============================================================

def build_benchmark():

    cases = []

    counter = 1

    # --------------------------------------------------------
    # CONTROLLED CASES
    # --------------------------------------------------------

    controlled_cases = [

        (
            "duplicate",
            "user",
            "language",
            "Python is used for ACMA development",
            "general",
            "ACMA project",
            "Python is used for ACMA development",
            "Ignore",
        ),

        (
            "explicit_update",
            "user",
            "language",
            "Java is used for DSA coursework",
            "general",
            "DSA",
            "I now prefer Python for DSA coursework",
            "Resolve",
        ),

        (
            "specific_event",
            "user",
            "food",
            "User is vegetarian",
            "general",
            "diet",
            "User ate chicken at a wedding",
            "Preserve",
        ),

        (
            "different_attribute",
            "user",
            "language",
            "Python is used for ACMA development",
            "general",
            "ACMA project",
            "ACMA backend uses FastAPI",
            "Preserve",
        ),

        (
            "different_context",
            "user",
            "language",
            "Python is used for ACMA development",
            "general",
            "ACMA project",
            "Python is used for DSA coursework",
            "Preserve",
        ),

        (
            "preference_update",
            "user",
            "food",
            "User prefers vegetarian food",
            "general",
            "diet",
            "I now prefer non-vegetarian food",
            "Resolve",
        ),

        (
            "os_update",
            "user",
            "operating_system",
            "User uses Windows",
            "general",
            "development",
            "I now use Linux",
            "Resolve",
        ),

        (
            "notification_update",
            "user",
            "notification",
            "User prefers email notifications",
            "general",
            "communication",
            "I now prefer SMS notifications",
            "Resolve",
        ),

        (
            "context_exception",
            "application",
            "theme",
            "Application normally uses dark mode",
            "general",
            "application",
            "The application uses light mode during presentations",
            "Preserve",
        ),

        (
            "database_update",
            "application",
            "database",
            "Application uses SQLite",
            "general",
            "ACMA project",
            "We switched the application database to PostgreSQL",
            "Resolve",
        ),
    ]

    for item in controlled_cases:

        (
            category,
            subject,
            attribute,
            old_value,
            scope,
            context,
            new_value,
            expected_action,
        ) = item

        old_memory = make_memory(
            f"old_{counter}",
            subject,
            attribute,
            old_value,
            scope,
            context,
        )

        new_memory = make_memory(
            f"new_{counter}",
            subject,
            attribute,
            new_value,
            scope,
            context,
        )

        cases.append(
            BenchmarkCase(
                case_id=f"case_{counter}",
                category=category,
                old_memory=old_memory,
                new_memory=new_memory,
                expected_action=expected_action,
                retrieval_required=True,
            )
        )

        counter += 1

    # --------------------------------------------------------
    # PARAPHRASES
    # --------------------------------------------------------

    paraphrases = [

        (
            "user",
            "language",
            "Python is used for ACMA development",
            "The ACMA project is developed using Python",
            "ACMA project",
        ),

        (
            "user",
            "food",
            "User prefers vegetarian food",
            "The user prefers eating vegetarian meals",
            "diet",
        ),

        (
            "user",
            "notification",
            "User prefers email notifications",
            "The user likes receiving notifications through email",
            "communication",
        ),

        (
            "application",
            "theme",
            "Application normally uses dark mode",
            "The application generally operates using a dark theme",
            "application",
        ),

        (
            "user",
            "language",
            "Java is used for DSA coursework",
            "Java is the language used for the user's DSA coursework",
            "DSA",
        ),
    ]

    for (
        subject,
        attribute,
        old_value,
        new_value,
        context,
    ) in paraphrases:

        for _ in range(20):

            old_memory = make_memory(
                f"old_{counter}",
                subject,
                attribute,
                old_value,
                "general",
                context,
            )

            new_memory = make_memory(
                f"new_{counter}",
                subject,
                attribute,
                new_value,
                "general",
                context,
            )

            cases.append(
                BenchmarkCase(
                    case_id=f"case_{counter}",
                    category="paraphrase",
                    old_memory=old_memory,
                    new_memory=new_memory,
                    expected_action="Preserve",
                    retrieval_required=True,
                )
            )

            counter += 1

    # --------------------------------------------------------
    # NATURAL UPDATES
    # --------------------------------------------------------

    natural_updates = [

        (
            "project",
            "deadline",
            "The ACMA paper deadline is in December",
            "ACMA project",
            "I moved the ACMA paper deadline to January",
        ),

        (
            "user",
            "language",
            "The user uses Java for DSA",
            "DSA",
            "I switched from Java to C++ for DSA",
        ),

        (
            "user",
            "food",
            "The user prefers vegetarian meals",
            "diet",
            "I now eat non-vegetarian meals",
        ),

        (
            "application",
            "database",
            "The application uses SQLite",
            "ACMA project",
            "We migrated the application database to PostgreSQL",
        ),

        (
            "application",
            "backend",
            "The application backend uses FastAPI",
            "ACMA project",
            "The backend now uses Django",
        ),

        (
            "application",
            "theme",
            "The application normally uses dark mode",
            "application",
            "The application now uses light mode",
        ),
    ]

    for (
        subject,
        attribute,
        old_value,
        context,
        new_value,
    ) in natural_updates:

        for _ in range(20):

            old_memory = make_memory(
                f"old_{counter}",
                subject,
                attribute,
                old_value,
                "general",
                context,
            )

            new_memory = make_memory(
                f"new_{counter}",
                subject,
                attribute,
                new_value,
                "general",
                context,
            )

            cases.append(
                BenchmarkCase(
                    case_id=f"case_{counter}",
                    category="natural_update",
                    old_memory=old_memory,
                    new_memory=new_memory,
                    expected_action="Resolve",
                    retrieval_required=True,
                )
            )

            counter += 1

    # --------------------------------------------------------
    # DIFFERENT ATTRIBUTE
    # --------------------------------------------------------

    different_attributes = [

        (
            "user",
            "language",
            "The user uses Python",
            "backend",
            "The user's backend uses FastAPI",
            "development",
        ),

        (
            "user",
            "language",
            "The user uses Java for DSA",
            "food",
            "The user prefers vegetarian food",
            "DSA",
        ),

        (
            "application",
            "backend",
            "The application uses FastAPI",
            "database",
            "The application uses PostgreSQL",
            "ACMA project",
        ),

        (
            "user",
            "operating_system",
            "The user uses Windows",
            "language",
            "The user uses Python",
            "development",
        ),
    ]

    for (
        subject,
        old_attribute,
        old_value,
        new_attribute,
        new_value,
        context,
    ) in different_attributes:

        for _ in range(25):

            old_memory = make_memory(
                f"old_{counter}",
                subject,
                old_attribute,
                old_value,
                "general",
                context,
            )

            new_memory = make_memory(
                f"new_{counter}",
                subject,
                new_attribute,
                new_value,
                "general",
                context,
            )

            cases.append(
                BenchmarkCase(
                    case_id=f"case_{counter}",
                    category="different_attribute",
                    old_memory=old_memory,
                    new_memory=new_memory,
                    expected_action="Preserve",
                    retrieval_required=True,
                )
            )

            counter += 1

    # --------------------------------------------------------
    # RANDOM NATURAL CASES
    # --------------------------------------------------------

    random_pairs = [

        (
            "user",
            "language",
            "The user uses Python",
            "development",
        ),

        (
            "user",
            "language",
            "The user uses Java",
            "DSA",
        ),

        (
            "user",
            "food",
            "The user prefers vegetarian food",
            "diet",
        ),

        (
            "application",
            "theme",
            "The application normally uses dark mode",
            "application",
        ),

        (
            "application",
            "database",
            "The application uses PostgreSQL",
            "ACMA project",
        ),

        (
            "application",
            "backend",
            "The application uses FastAPI",
            "ACMA project",
        ),

        (
            "user",
            "notification",
            "The user prefers email notifications",
            "communication",
        ),
    ]

    while len(cases) < TOTAL_CASES:

        (
            subject,
            attribute,
            value,
            context,
        ) = random.choice(
            random_pairs
        )

        new_variations = [

            value,

            f"The user currently uses "
            f"{value.split('uses')[-1].strip()}",

            f"The user generally has this preference: "
            f"{value}",
        ]

        new_value = random.choice(
            new_variations
        )

        old_memory = make_memory(
            f"old_{counter}",
            subject,
            attribute,
            value,
            "general",
            context,
        )

        new_memory = make_memory(
            f"new_{counter}",
            subject,
            attribute,
            new_value,
            "general",
            context,
        )

        expected_action = (
            "Ignore"
            if normalize_text(
                old_memory.value
            )
            == normalize_text(
                new_memory.value
            )
            else "Preserve"
        )

        cases.append(
            BenchmarkCase(
                case_id=f"case_{counter}",
                category="natural",
                old_memory=old_memory,
                new_memory=new_memory,
                expected_action=expected_action,
                retrieval_required=True,
            )
        )

        counter += 1

    return cases[:TOTAL_CASES]


# ============================================================
# CANONICAL CORPUS
# ============================================================

def write_benchmark_memory_corpus(cases):

    """
    Store identical old memories only once.

    This prevents duplicate embeddings from artificially
    occupying the retrieval ranking.
    """

    unique_memories = {}

    for case in cases:

        memory = case.old_memory

        key = memory_key(memory)

        if key not in unique_memories:

            unique_memories[key] = memory

    memories = list(
        unique_memories.values()
    )

    for index, memory in enumerate(
        memories,
        start=1,
    ):

        memory.memory_id = (
            f"canonical_{index}"
        )

    Path(
        BENCHMARK_MEMORY_FILE
    ).write_text(
        json.dumps(
            [
                memory.to_dict()
                for memory in memories
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    return memories


# ============================================================
# RAG INSPECTION
# ============================================================

def print_rag_example(
    case,
    diagnostics,
):

    print()
    print("=" * 75)
    print("RAG INSPECTION")
    print("=" * 75)

    print(
        f"Case:     {case.case_id}"
    )

    print(
        f"Category: {case.category}"
    )

    print()
    print("NEW MEMORY / QUERY:")
    print(
        f"  {case.new_memory.value}"
    )

    print()
    print("EXPECTED OLD MEMORY:")
    print(
        f"  {case.old_memory.value}"
    )

    expected_key = memory_key(
        case.old_memory
    )

    expected_rank = None

    print()
    print("TOP RETRIEVED MEMORIES:")

    for rank, (
        memory,
        score,
    ) in enumerate(
        diagnostics,
        start=1,
    ):

        is_expected = (
            memory_key(memory)
            == expected_key
        )

        marker = "  <-- EXPECTED" if is_expected else ""

        if is_expected:
            expected_rank = rank

        print()
        print(
            f"  {rank}. "
            f"score={score:.4f}"
            f"{marker}"
        )

        print(
            f"     subject : {memory.subject}"
        )

        print(
            f"     attribute: {memory.attribute}"
        )

        print(
            f"     value   : {memory.value}"
        )

        print(
            f"     scope   : {memory.scope}"
        )

        print(
            f"     context : {memory.context}"
        )

    print()

    if expected_rank is None:

        print(
            "EXPECTED MEMORY: NOT IN TOP 10"
        )

    else:

        print(
            f"EXPECTED MEMORY RANK: "
            f"{expected_rank}"
        )


# ============================================================
# RUN PIPELINE
# ============================================================

def run_pipeline(
    store,
    nli_engine,
    case,
):

    new_memory = case.new_memory

    query = MemoryQuery(
        subject=new_memory.subject,
        attribute=new_memory.attribute,
        value=new_memory.value,
        scope=new_memory.scope,
        context=new_memory.context,
        time=new_memory.time,
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    retrieved = store.retrieve_related_memories(
        query,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    diagnostics = store.retrieve_related_memories(
        query,
        top_k=RETRIEVAL_DIAGNOSTIC_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    expected_key = memory_key(
        case.old_memory
    )

    diagnostic_rank = None

    for rank, (
        memory,
        score,
    ) in enumerate(
        diagnostics,
        start=1,
    ):

        if memory_key(memory) == expected_key:

            diagnostic_rank = rank
            break

    # --------------------------------------------------------
    # RAG FAILURE
    # --------------------------------------------------------

    if diagnostic_rank is None:

        return {
            "expected_action": case.expected_action,
            "predicted_action": "RAG_FAILURE",
            "correct": False,
            "retrieved": False,
            "diagnostic_rank": None,
            "stage": "rag",
            "diagnostics": diagnostics,
        }

    # --------------------------------------------------------
    # FIND EXPECTED MEMORY IN TOP K
    # --------------------------------------------------------

    operational_memory = None
    operational_similarity = None

    for memory, similarity in retrieved:

        if memory_key(memory) == expected_key:

            operational_memory = memory
            operational_similarity = similarity

            break

    if operational_memory is None:

        return {
            "expected_action": case.expected_action,
            "predicted_action": "RAG_FAILURE",
            "correct": False,
            "retrieved": True,
            "diagnostic_rank": diagnostic_rank,
            "stage": "rag_top_k",
            "diagnostics": diagnostics,
        }

    # --------------------------------------------------------
    # SAFETY GATE
    # --------------------------------------------------------

    safety = evaluate_rule_safety(
        operational_memory,
        new_memory,
        operational_similarity,
    )

    if safety.safe:

        predicted_action = safety.action_hint

        return {
            "expected_action": case.expected_action,
            "predicted_action": predicted_action,
            "correct": (
                predicted_action
                == case.expected_action
            ),
            "retrieved": True,
            "diagnostic_rank": diagnostic_rank,
            "stage": "rule",
            "safety": safety,
            "diagnostics": diagnostics,
        }

    # --------------------------------------------------------
    # NLI
    # --------------------------------------------------------

    nli_result = nli_engine.compare(
        operational_memory,
        new_memory,
    )

    nli_label = nli_result["label"]

    nli_confidence = (
        nli_result["confidence"]
    )

    nli_margin = (
        nli_result["margin"]
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata_score = metadata_agreement(
        operational_memory,
        new_memory,
    )

    # --------------------------------------------------------
    # UPDATE SIGNAL
    # --------------------------------------------------------

    update_signals = detect_update_signals(
        new_memory.value
    )

    update_signal = calculate_update_signal(
        update_signals
    )

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    evidence_score = calculate_evidence_score(
        nli_confidence=nli_confidence,
        nli_margin=nli_margin,
        similarity=operational_similarity,
        metadata_agreement=metadata_score,
        update_signal=update_signal,
        source_reliability=1.0,
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    predicted_action = decide_action(
        relationship=safety.relationship,
        evidence_score=evidence_score,
        nli_label=nli_label,
        nli_confidence=nli_confidence,
    )

    return {
        "expected_action": case.expected_action,
        "predicted_action": predicted_action,
        "correct": (
            predicted_action
            == case.expected_action
        ),
        "retrieved": True,
        "diagnostic_rank": diagnostic_rank,
        "stage": "decision",
        "relationship": safety.relationship,
        "nli_label": nli_label,
        "nli_confidence": nli_confidence,
        "nli_margin": nli_margin,
        "evidence_score": evidence_score,
        "retrieval_score": operational_similarity,
        "diagnostics": diagnostics,
    }


# ============================================================
# BENCHMARK
# ============================================================

def test_realistic_pipeline_benchmark():

    cases = build_benchmark()

    write_benchmark_memory_corpus(
        cases
    )

    print()
    print("=" * 75)
    print("ACMA REALISTIC PIPELINE BENCHMARK")
    print("=" * 75)

    print(
        f"Benchmark cases: {len(cases)}"
    )

    print()
    print(
        "Building RAG embedding cache..."
    )

    store = MemoryStore(
        BENCHMARK_MEMORY_FILE
    )

    # Force embedding cache creation before NLI.
    # This makes the RAG cost visible separately.
    store._build_embedding_cache()

    print(
        "RAG embedding cache ready."
    )

    print()
    print(
        "Loading NLI model..."
    )

    nli_engine = NLIEngine()

    print(
        "NLI engine ready."
    )

    print()
    print(
        "Running pipeline..."
    )

    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):

        result = run_pipeline(
            store,
            nli_engine,
            case,
        )

        results.append(
            (
                case,
                result,
            )
        )

        if (
            index == 1
            or index % 25 == 0
            or index == len(cases)
        ):

            print(
                f"Processed "
                f"{index}/{len(cases)} cases...",
                flush=True,
            )

    # ========================================================
    # RAG EXAMPLES
    # ========================================================

    print()
    print("=" * 75)
    print("INDIVIDUAL RAG BEHAVIOR")
    print("=" * 75)

    # Show representative examples from different categories.
    selected = []

    categories_seen = set()

    for case, result in results:

        if case.category not in categories_seen:

            selected.append(
                (case, result)
            )

            categories_seen.add(
                case.category
            )

        if len(selected) >= RAG_EXAMPLES_TO_SHOW:
            break

    for case, result in selected:

        print_rag_example(
            case,
            result.get(
                "diagnostics",
                []
            ),
        )

    # ========================================================
    # METRICS
    # ========================================================

    total = len(results)

    correct = sum(
        1
        for _, result in results
        if result["correct"]
    )

    accuracy = (
        correct / total
        if total
        else 0
    )

    # --------------------------------------------------------
    # RETRIEVAL METRICS
    # --------------------------------------------------------

    required_cases = [
        (case, result)
        for case, result in results
        if case.retrieval_required
    ]

    required_total = len(
        required_cases
    )

    recall_1 = sum(
        1
        for _, result in required_cases
        if result["diagnostic_rank"] == 1
    )

    recall_5 = sum(
        1
        for _, result in required_cases
        if (
            result["diagnostic_rank"]
            and result["diagnostic_rank"] <= 5
        )
    )

    recall_10 = sum(
        1
        for _, result in required_cases
        if (
            result["diagnostic_rank"]
            and result["diagnostic_rank"] <= 10
        )
    )

    operational_retrievals = sum(
        1
        for _, result in required_cases
        if result["retrieved"]
    )

    rag_failures = sum(
        1
        for _, result in results
        if result["predicted_action"]
        == "RAG_FAILURE"
    )

    # --------------------------------------------------------
    # STAGE COUNTS
    # --------------------------------------------------------

    rule_cases = sum(
        1
        for _, result in results
        if result["stage"] == "rule"
    )

    nli_cases = sum(
        1
        for _, result in results
        if result["stage"] == "decision"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 75)

    print()
    print("DECISION PERFORMANCE")
    print("-" * 75)

    print(
        f"Total cases:             {total}"
    )

    print(
        f"Correct decisions:       {correct}"
    )

    print(
        f"Incorrect decisions:     "
        f"{total - correct}"
    )

    print(
        f"Overall accuracy:        "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print("RAG RETRIEVAL")
    print("-" * 75)

    print(
        f"Retrieval-required cases: "
        f"{required_total}"
    )

    print(
        f"Operational retrievals:    "
        f"{operational_retrievals}"
    )

    if required_total:

        print(
            f"Recall@1:                 "
            f"{recall_1 / required_total * 100:.2f}%"
        )

        print(
            f"Recall@5:                 "
            f"{recall_5 / required_total * 100:.2f}%"
        )

        print(
            f"Recall@10:                "
            f"{recall_10 / required_total * 100:.2f}%"
        )

    print(
        f"RAG failures:             "
        f"{rag_failures}"
    )

    print()
    print("PIPELINE STAGES")
    print("-" * 75)

    print(
        f"Rule SAFE decisions:      "
        f"{rule_cases}"
    )

    print(
        f"NLI/decision cases:       "
        f"{nli_cases}"
    )

    # ========================================================
    # ACTION DISTRIBUTION
    # ========================================================

    print()
    print("FINAL ACTION DISTRIBUTION")
    print("-" * 75)

    action_counts = {}

    for _, result in results:

        action = result[
            "predicted_action"
        ]

        if action not in action_counts:

            action_counts[action] = {
                "total": 0,
                "correct": 0,
            }

        action_counts[action][
            "total"
        ] += 1

        if result["correct"]:

            action_counts[action][
                "correct"
            ] += 1

    for action, values in sorted(
        action_counts.items()
    ):

        action_total = values[
            "total"
        ]

        action_correct = values[
            "correct"
        ]

        action_accuracy = (
            action_correct
            / action_total
            * 100
            if action_total
            else 0
        )

        print(
            f"{action:<20}"
            f"{action_total:>5} cases | "
            f"{action_correct:>5} correct | "
            f"{action_accuracy:>7.2f}%"
        )

    # ========================================================
    # FAILURES
    # ========================================================

    failures = [
        (case, result)
        for case, result in results
        if not result["correct"]
    ]

    print()
    print("FIRST 20 DECISION FAILURES")
    print("-" * 75)

    for case, result in failures[:20]:

        print()
        print(
            f"{case.case_id} | "
            f"{case.category}"
        )

        print(
            f"Expected:  "
            f"{case.expected_action}"
        )

        print(
            f"Predicted: "
            f"{result['predicted_action']}"
        )

        print(
            f"RAG rank:  "
            f"{result['diagnostic_rank']}"
        )

        if "nli_label" in result:

            print(
                f"NLI:       "
                f"{result['nli_label']} "
                f"("
                f"{result['nli_confidence']:.4f}"
                f")"
            )

            print(
                f"Evidence:  "
                f"{result['evidence_score']:.4f}"
            )


if __name__ == "__main__":

    test_realistic_pipeline_benchmark()