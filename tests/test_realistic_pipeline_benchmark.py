import json
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

import pytest

from memory.schema import Memory, MemoryQuery
from memory.store import MemoryStore

from conflict.safety_gate import evaluate_rule_safety
from conflict.nli_engine import NLIEngine
from conflict.metadata import metadata_agreement
from conflict.update_detector import detect_update_signals
from conflict.evidence_score import calculate_evidence_score
from conflict.decision_policy import decide_action


# =============================================================================
# CONFIGURATION
# =============================================================================

TOTAL_CASES = 600

# IMPORTANT:
# Keep this at 50 while diagnosing the pipeline.
# After the pipeline is verified, change to TOTAL_CASES.
DIAGNOSTIC_CASES = 50

TOP_K = 5
RETRIEVAL_DIAGNOSTIC_K = 10
SIMILARITY_THRESHOLD = 0.30

BENCHMARK_MEMORY_FILE = Path(
    "data/full_pipeline_benchmark_memories.json"
)


# =============================================================================
# BENCHMARK CASE
# =============================================================================

@dataclass
class BenchmarkCase:

    case_id: str
    category: str

    old_memory: Memory
    new_memory: Memory

    expected_action: str

    retrieval_required: bool = True


# =============================================================================
# MEMORY FACTORY
# =============================================================================

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


# =============================================================================
# LOGICAL MEMORY KEY
# =============================================================================

def memory_key(memory):

    return (
        memory.subject.strip().lower(),
        memory.attribute.strip().lower(),
        memory.value.strip().lower(),
        memory.scope.strip().lower(),
        memory.context.strip().lower(),
        memory.time,
    )


# =============================================================================
# BUILD BENCHMARK
# =============================================================================

def build_benchmark():

    cases = []

    case_number = 1

    # =====================================================================
    # 1. EXACT DUPLICATES
    # =====================================================================

    duplicate_cases = [

        (
            "duplicate_java",
            make_memory(
                "old_java",
                "user",
                "language",
                "I use Java for DSA.",
                context="DSA",
            ),
            make_memory(
                "new_java",
                "user",
                "language",
                "I use Java for DSA.",
                context="DSA",
            ),
            "Ignore",
        ),

        (
            "duplicate_python",
            make_memory(
                "old_python",
                "user",
                "language",
                "Python is used for ACMA development.",
                context="ACMA",
            ),
            make_memory(
                "new_python",
                "user",
                "language",
                "Python is used for ACMA development.",
                context="ACMA",
            ),
            "Ignore",
        ),

        (
            "duplicate_fastapi",
            make_memory(
                "old_fastapi",
                "application",
                "backend",
                "The ACMA backend uses FastAPI.",
                context="ACMA",
            ),
            make_memory(
                "new_fastapi",
                "application",
                "backend",
                "The ACMA backend uses FastAPI.",
                context="ACMA",
            ),
            "Ignore",
        ),

        (
            "duplicate_vegetarian",
            make_memory(
                "old_veg",
                "user",
                "food",
                "I am vegetarian.",
                context="diet",
            ),
            make_memory(
                "new_veg",
                "user",
                "food",
                "I am vegetarian.",
                context="diet",
            ),
            "Ignore",
        ),
    ]

    for name, old, new, expected in duplicate_cases:

        cases.append(
            BenchmarkCase(
                case_id=f"case_{case_number}_{name}",
                category="duplicate",
                old_memory=old,
                new_memory=new,
                expected_action=expected,
            )
        )

        case_number += 1

    # =====================================================================
    # 2. EXPLICIT UPDATES
    # =====================================================================

    update_cases = [

        (
            "java_to_python",
            make_memory(
                "old_java_update",
                "user",
                "language",
                "I use Java for DSA.",
                context="DSA",
            ),
            make_memory(
                "new_python_update",
                "user",
                "language",
                "I now use Python for DSA.",
                context="DSA",
            ),
            "Resolve",
        ),

        (
            "windows_to_linux",
            make_memory(
                "old_windows",
                "user",
                "operating_system",
                "I use Windows.",
                context="computing",
            ),
            make_memory(
                "new_linux",
                "user",
                "operating_system",
                "I now use Linux.",
                context="computing",
            ),
            "Resolve",
        ),

        (
            "email_to_sms",
            make_memory(
                "old_email",
                "user",
                "notification",
                "I prefer email notifications.",
                context="communication",
            ),
            make_memory(
                "new_sms",
                "user",
                "notification",
                "I now prefer SMS notifications.",
                context="communication",
            ),
            "Resolve",
        ),

        (
            "dark_to_light",
            make_memory(
                "old_dark",
                "application",
                "theme",
                "The application normally uses dark mode.",
                context="application",
            ),
            make_memory(
                "new_light",
                "application",
                "theme",
                "The application now uses light mode.",
                context="application",
            ),
            "Resolve",
        ),

        (
            "sqlite_to_postgresql",
            make_memory(
                "old_sqlite",
                "application",
                "database",
                "The application uses SQLite.",
                context="database",
            ),
            make_memory(
                "new_postgres",
                "application",
                "database",
                "We migrated the application database to PostgreSQL.",
                context="database",
            ),
            "Resolve",
        ),

        (
            "fastapi_to_django",
            make_memory(
                "old_fastapi_update",
                "application",
                "backend",
                "The application backend uses FastAPI.",
                context="backend",
            ),
            make_memory(
                "new_django",
                "application",
                "backend",
                "The backend now uses Django.",
                context="backend",
            ),
            "Resolve",
        ),

        (
            "java_to_cpp",
            make_memory(
                "old_java_cpp",
                "user",
                "language",
                "The user uses Java for DSA.",
                context="DSA",
            ),
            make_memory(
                "new_cpp",
                "user",
                "language",
                "I switched from Java to C++ for DSA.",
                context="DSA",
            ),
            "Resolve",
        ),
    ]

    for name, old, new, expected in update_cases:

        cases.append(
            BenchmarkCase(
                case_id=f"case_{case_number}_{name}",
                category="explicit_update",
                old_memory=old,
                new_memory=new,
                expected_action=expected,
            )
        )

        case_number += 1

    # =====================================================================
    # 3. SPECIFIC EVENTS
    # =====================================================================

    event_cases = [

        (
            "vegetarian_chicken_wedding",
            make_memory(
                "old_vegetarian_event",
                "user",
                "food",
                "I am vegetarian.",
                scope="general",
                context="diet",
            ),
            make_memory(
                "new_chicken_event",
                "user",
                "food",
                "I ate chicken at a wedding.",
                scope="specific_event",
                context="wedding",
            ),
            "Preserve",
        ),

        (
            "vegetarian_chicken_party",
            make_memory(
                "old_vegetarian_party",
                "user",
                "food",
                "I am vegetarian.",
                scope="general",
                context="diet",
            ),
            make_memory(
                "new_chicken_party",
                "user",
                "food",
                "I ate chicken at a party.",
                scope="specific_event",
                context="party",
            ),
            "Preserve",
        ),

        (
            "dark_light_presentation",
            make_memory(
                "old_dark_presentation",
                "application",
                "theme",
                "The application normally uses dark mode.",
                scope="general",
                context="application",
            ),
            make_memory(
                "new_light_presentation",
                "application",
                "theme",
                "I use light mode for presentations.",
                scope="specific_event",
                context="presentation",
            ),
            "Preserve",
        ),
    ]

    for name, old, new, expected in event_cases:

        cases.append(
            BenchmarkCase(
                case_id=f"case_{case_number}_{name}",
                category="specific_event",
                old_memory=old,
                new_memory=new,
                expected_action=expected,
            )
        )

        case_number += 1

    # =====================================================================
    # 4. DIFFERENT ATTRIBUTES
    # =====================================================================

    different_attribute_cases = [

        (
            "python_fastapi",
            make_memory(
                "old_python_attribute",
                "user",
                "language",
                "Python is used for ACMA development.",
                context="ACMA",
            ),
            make_memory(
                "new_fastapi_attribute",
                "user",
                "backend",
                "The ACMA backend uses FastAPI.",
                context="ACMA",
            ),
            "Preserve",
        ),

        (
            "java_food",
            make_memory(
                "old_java_attribute",
                "user",
                "language",
                "I use Java for DSA.",
                context="DSA",
            ),
            make_memory(
                "new_food_attribute",
                "user",
                "food",
                "I prefer vegetarian food.",
                context="diet",
            ),
            "Preserve",
        ),

        (
            "python_theme",
            make_memory(
                "old_python_theme",
                "user",
                "language",
                "Python is used for ACMA development.",
                context="ACMA",
            ),
            make_memory(
                "new_theme_attribute",
                "application",
                "theme",
                "The application normally uses dark mode.",
                context="application",
            ),
            "Preserve",
        ),
    ]

    for name, old, new, expected in different_attribute_cases:

        cases.append(
            BenchmarkCase(
                case_id=f"case_{case_number}_{name}",
                category="different_attribute",
                old_memory=old,
                new_memory=new,
                expected_action=expected,
            )
        )

        case_number += 1

    # =====================================================================
    # 5. DIFFERENT CONTEXT
    # =====================================================================

    different_context_cases = [

        (
            "python_acma_dsa",
            make_memory(
                "old_python_acma",
                "user",
                "language",
                "Python is used for ACMA development.",
                context="ACMA",
            ),
            make_memory(
                "new_python_dsa",
                "user",
                "language",
                "Python is used for DSA coursework.",
                context="DSA",
            ),
            "Preserve",
        ),

        (
            "java_acma_dsa",
            make_memory(
                "old_java_dsa",
                "user",
                "language",
                "Java is used for DSA coursework.",
                context="DSA",
            ),
            make_memory(
                "new_java_acma",
                "user",
                "language",
                "Java is used for an ACMA experiment.",
                context="ACMA",
            ),
            "Preserve",
        ),
    ]

    for name, old, new, expected in different_context_cases:

        cases.append(
            BenchmarkCase(
                case_id=f"case_{case_number}_{name}",
                category="different_context",
                old_memory=old,
                new_memory=new,
                expected_action=expected,
            )
        )

        case_number += 1

    # =====================================================================
    # 6. NATURAL LANGUAGE UPDATES
    # =====================================================================

    natural_update_cases = [

        (
            "vegetarian_to_nonveg",
            make_memory(
                "old_vegetarian",
                "user",
                "food",
                "I am vegetarian.",
                context="diet",
            ),
            make_memory(
                "new_nonveg",
                "user",
                "food",
                "I no longer prefer vegetarian food.",
                context="diet",
            ),
            "Resolve",
        ),

        (
            "python_to_java",
            make_memory(
                "old_python",
                "user",
                "language",
                "I use Python for DSA.",
                context="DSA",
            ),
            make_memory(
                "new_java",
                "user",
                "language",
                "I switched to Java for DSA.",
                context="DSA",
            ),
            "Resolve",
        ),

        (
            "windows_to_linux_natural",
            make_memory(
                "old_windows_natural",
                "user",
                "operating_system",
                "I use Windows.",
                context="computing",
            ),
            make_memory(
                "new_linux_natural",
                "user",
                "operating_system",
                "I switched to Linux.",
                context="computing",
            ),
            "Resolve",
        ),
    ]

    for name, old, new, expected in natural_update_cases:

        cases.append(
            BenchmarkCase(
                case_id=f"case_{case_number}_{name}",
                category="natural_update",
                old_memory=old,
                new_memory=new,
                expected_action=expected,
            )
        )

        case_number += 1

    # =====================================================================
    # 7. PARAPHRASES
    # =====================================================================

    paraphrase_cases = [

        (
            "python_paraphrase",
            make_memory(
                "old_python_para",
                "user",
                "language",
                "Python is used for ACMA development.",
                context="ACMA",
            ),
            make_memory(
                "new_python_para",
                "user",
                "language",
                "The ACMA project is developed using Python.",
                context="ACMA",
            ),
            "Preserve",
        ),

        (
            "vegetarian_paraphrase",
            make_memory(
                "old_veg_para",
                "user",
                "food",
                "I am vegetarian.",
                context="diet",
            ),
            make_memory(
                "new_veg_para",
                "user",
                "food",
                "I follow a vegetarian diet.",
                context="diet",
            ),
            "Preserve",
        ),

        (
            "fastapi_paraphrase",
            make_memory(
                "old_fastapi_para",
                "application",
                "backend",
                "The ACMA backend uses FastAPI.",
                context="ACMA",
            ),
            make_memory(
                "new_fastapi_para",
                "application",
                "backend",
                "ACMA's backend is implemented with FastAPI.",
                context="ACMA",
            ),
            "Preserve",
        ),
    ]

    for name, old, new, expected in paraphrase_cases:

        cases.append(
            BenchmarkCase(
                case_id=f"case_{case_number}_{name}",
                category="paraphrase",
                old_memory=old,
                new_memory=new,
                expected_action=expected,
            )
        )

        case_number += 1

    # =====================================================================
    # 8. FILL THE REST OF THE BENCHMARK
    # =====================================================================

    base_memories = [

        make_memory(
            "base_1",
            "user",
            "language",
            "Python is used for ACMA development.",
            context="ACMA",
        ),

        make_memory(
            "base_2",
            "user",
            "language",
            "Java is used for DSA coursework.",
            context="DSA",
        ),

        make_memory(
            "base_3",
            "user",
            "food",
            "User is vegetarian.",
            context="diet",
        ),

        make_memory(
            "base_4",
            "user",
            "operating_system",
            "User uses Windows.",
            context="computing",
        ),

        make_memory(
            "base_5",
            "application",
            "backend",
            "Application backend uses FastAPI.",
            context="backend",
        ),

        make_memory(
            "base_6",
            "application",
            "database",
            "Application uses SQLite.",
            context="database",
        ),

        make_memory(
            "base_7",
            "user",
            "notification",
            "User prefers email notifications.",
            context="communication",
        ),

        make_memory(
            "base_8",
            "application",
            "theme",
            "Application normally uses dark mode.",
            context="application",
        ),
    ]

    fill_index = 0

    while len(cases) < TOTAL_CASES:

        old = base_memories[
            fill_index % len(base_memories)
        ]

        mode = fill_index % 4

        if mode == 0:

            new = make_memory(
                f"fill_new_{fill_index}",
                old.subject,
                old.attribute,
                old.value,
                scope=old.scope,
                context=old.context,
            )

            expected = "Ignore"
            category = "generated_duplicate"

        elif mode == 1:

            new = make_memory(
                f"fill_new_{fill_index}",
                old.subject,
                old.attribute,
                old.value + " currently.",
                scope=old.scope,
                context=old.context,
            )

            expected = "Preserve"
            category = "generated_paraphrase"

        elif mode == 2:

            new = make_memory(
                f"fill_new_{fill_index}",
                old.subject,
                old.attribute,
                old.value,
                scope=old.scope,
                context=f"alternate_{fill_index}",
            )

            expected = "Preserve"
            category = "generated_context"

        else:

            new = make_memory(
                f"fill_new_{fill_index}",
                old.subject,
                old.attribute,
                f"I now use an alternative value instead of {old.value}.",
                scope=old.scope,
                context=old.context,
            )

            expected = "Resolve"
            category = "generated_update"

        cases.append(
            BenchmarkCase(
                case_id=f"case_{case_number}_fill_{fill_index}",
                category=category,
                old_memory=old,
                new_memory=new,
                expected_action=expected,
            )
        )

        case_number += 1
        fill_index += 1

    return cases[:TOTAL_CASES]


# =============================================================================
# WRITE CORPUS
# =============================================================================

def write_benchmark_memory_corpus(cases):

    unique = {}

    for case in cases:

        key = memory_key(
            case.old_memory
        )

        if key not in unique:

            unique[key] = case.old_memory

    memories = list(
        unique.values()
    )

    for index, memory in enumerate(
        memories,
        start=1,
    ):

        memory.memory_id = (
            f"canonical_{index}"
        )

    BENCHMARK_MEMORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        BENCHMARK_MEMORY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [
                memory.to_dict()
                for memory in memories
            ],
            file,
            indent=2,
        )

    return memories


# =============================================================================
# CALCULATE UPDATE SIGNAL
# =============================================================================

def calculate_update_signal(signals):

    if signals["explicit_update"]:

        return 1.0

    if signals["temporal_change"]:

        return 0.5

    return 0.0


# =============================================================================
# RUN ONE PIPELINE CASE
# =============================================================================

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

    # =================================================================
    # 1. RAG
    # =================================================================

    retrieved = store.retrieve_related_memories(
        query,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    diagnostic = store.retrieve_related_memories(
        query,
        top_k=RETRIEVAL_DIAGNOSTIC_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    expected_key = memory_key(
        case.old_memory
    )

    diagnostic_rank = None
    diagnostic_similarity = None

    for rank, (memory, score) in enumerate(
        diagnostic,
        start=1,
    ):

        if memory_key(memory) == expected_key:

            diagnostic_rank = rank
            diagnostic_similarity = score

            break

    # =================================================================
    # RAG NOT FOUND
    # =================================================================

    if diagnostic_rank is None:

        return {
            "expected": case.expected_action,
            "predicted": "RAG_FAILURE",
            "correct": False,

            "stage": "RAG",

            "rank": None,
            "similarity": None,

            "relationship": None,

            "safe": None,

            "nli_label": None,
            "nli_confidence": None,
            "nli_margin": None,

            "metadata": None,
            "update_signal": None,
            "evidence": None,
        }

    # =================================================================
    # FIND EXPECTED MEMORY IN OPERATIONAL TOP-K
    # =================================================================

    operational_memory = None
    operational_similarity = None

    for memory, similarity in retrieved:

        if memory_key(memory) == expected_key:

            operational_memory = memory
            operational_similarity = similarity

            break

    if operational_memory is None:

        return {
            "expected": case.expected_action,
            "predicted": "RAG_TOP_K_FAILURE",
            "correct": False,

            "stage": "RAG_TOP_K",

            "rank": diagnostic_rank,
            "similarity": diagnostic_similarity,

            "relationship": None,

            "safe": None,

            "nli_label": None,
            "nli_confidence": None,
            "nli_margin": None,

            "metadata": None,
            "update_signal": None,
            "evidence": None,
        }

    # =================================================================
    # 2. SAFETY GATE
    # =================================================================

    safety = evaluate_rule_safety(
        operational_memory,
        new_memory,
        operational_similarity,
    )

    # =================================================================
    # SAFE RULE PATH
    # =================================================================

    if safety.safe:

        predicted = safety.action_hint

        return {
            "expected": case.expected_action,
            "predicted": predicted,

            "correct": (
                predicted
                == case.expected_action
            ),

            "stage": "RULE",

            "rank": diagnostic_rank,
            "similarity": operational_similarity,

            "relationship": safety.relationship,

            "safe": True,

            "safety_confidence":
                safety.safety_confidence,

            "safety_reason":
                safety.reason,

            "nli_label": None,
            "nli_confidence": None,
            "nli_margin": None,

            "metadata": None,
            "update_signal": None,
            "evidence": None,
        }

    # =================================================================
    # 3. NLI
    # =================================================================

    nli_result = nli_engine.compare(
        operational_memory,
        new_memory,
    )

    # =================================================================
    # 4. METADATA
    # =================================================================

    metadata_score = metadata_agreement(
        operational_memory,
        new_memory,
    )

    # =================================================================
    # 5. UPDATE SIGNAL
    # =================================================================

    update_signals = detect_update_signals(
        new_memory.value
    )

    update_signal = calculate_update_signal(
        update_signals
    )

    # =================================================================
    # 6. EVIDENCE
    # =================================================================

    evidence = calculate_evidence_score(
        nli_confidence=nli_result[
            "confidence"
        ],

        nli_margin=nli_result[
            "margin"
        ],

        similarity=operational_similarity,

        metadata_agreement=metadata_score,

        update_signal=update_signal,

        source_reliability=1.0,
    )

    # =================================================================
    # 7. DECISION POLICY
    # =================================================================
    #
    # IMPORTANT:
    # Pass EVERY signal calculated above.
    #
    # Previously only relationship, evidence_score,
    # nli_label and nli_confidence were passed.
    #
    # That caused the remaining parameters of decide_action()
    # to silently use their default values:
    #
    #     nli_margin = 0.0
    #     update_signal = 0.0
    #     metadata_agreement = 0.0
    #     similarity = 0.0
    #
    # This made strong NLI/update cases appear weak to the policy.
    #

    predicted = decide_action(
        relationship=safety.relationship,
        evidence_score=evidence,
        nli_label=nli_result[
            "label"
        ],
        nli_confidence=nli_result[
            "confidence"
        ],
        nli_margin=nli_result[
            "margin"
        ],
        update_signal=update_signal,
        metadata_agreement=metadata_score,
        similarity=operational_similarity,
        old_value=operational_memory.value,
        new_value=new_memory.value,
        same_subject=(
            operational_memory.subject
            == new_memory.subject
        ),
        same_attribute=(
            operational_memory.attribute
            == new_memory.attribute
        ),
        same_context=(
            operational_memory.context
            == new_memory.context
        ),
        explicit_update=update_signals.get(
            "explicit_update", False
        ),
        temporal_change=update_signals.get(
            "temporal_change", False
        ),
    )

    return {
        "expected": case.expected_action,
        "predicted": predicted,

        "correct": (
            predicted
            == case.expected_action
        ),

        "stage": "DECISION",

        "rank": diagnostic_rank,
        "similarity": operational_similarity,

        "relationship":
            safety.relationship,

        "safe": False,

        "safety_confidence":
            safety.safety_confidence,

        "safety_reason":
            safety.reason,

        "nli_label":
            nli_result["label"],

        "nli_confidence":
            nli_result["confidence"],

        "nli_margin":
            nli_result["margin"],

        "nli_scores":
            nli_result.get("scores"),

        "metadata":
            metadata_score,

        "update_signal":
            update_signal,

        "evidence":
            evidence,
    }


# =============================================================================
# PRINT DETAILED CASE
# =============================================================================

def print_case(
    index,
    case,
    result,
):

    print()
    print("-" * 100)

    print(
        f"CASE {index}: "
        f"{case.case_id}"
    )

    print(
        f"Category:     {case.category}"
    )

    print(
        f"Expected:     {result['expected']}"
    )

    print(
        f"Predicted:    {result['predicted']}"
    )

    print(
        f"Correct:      {result['correct']}"
    )

    print(
        f"Stage:        {result['stage']}"
    )

    print()

    print(
        f"RAG rank:     {result['rank']}"
    )

    print(
        f"RAG score:    {result['similarity']}"
    )

    print()

    print(
        f"Relationship: {result['relationship']}"
    )

    print(
        f"Safe:         {result['safe']}"
    )

    if result.get("safety_confidence") is not None:

        print(
            f"Safety conf:  "
            f"{result['safety_confidence']}"
        )

    if result.get("safety_reason"):

        print(
            f"Safety reason:"
            f" {result['safety_reason']}"
        )

    if result["nli_label"] is not None:

        print()

        print("NLI:")

        print(
            f"  Label:      "
            f"{result['nli_label']}"
        )

        print(
            f"  Confidence: "
            f"{result['nli_confidence']}"
        )

        print(
            f"  Margin:     "
            f"{result['nli_margin']}"
        )

        print(
            f"  Scores:     "
            f"{result.get('nli_scores')}"
        )

        print()

        print("Evidence:")

        print(
            f"  Metadata:   "
            f"{result['metadata']}"
        )

        print(
            f"  Update:     "
            f"{result['update_signal']}"
        )

        print(
            f"  Evidence:   "
            f"{result['evidence']}"
        )

    print()

    print(
        "OLD:"
    )

    print(
        f"  {case.old_memory.value}"
    )

    print()

    print(
        "NEW:"
    )

    print(
        f"  {case.new_memory.value}"
    )


# =============================================================================
# MAIN TEST
# =============================================================================

def test_realistic_pipeline_benchmark():

    print()
    print("#" * 100)
    print("ACMA REALISTIC END-TO-END PIPELINE DIAGNOSTIC")
    print("#" * 100)

    print()
    print(
        "Pipeline:"
    )

    print(
        "RAG -> Safety Gate -> NLI -> "
        "Evidence Score -> Decision Policy"
    )

    print()
    print(
        "Diagnostic mode:"
    )

    print(
        f"Running first {DIAGNOSTIC_CASES} "
        f"cases out of {TOTAL_CASES}"
    )

    # =================================================================
    # BUILD
    # =================================================================

    cases = build_benchmark()

    print()
    print(
        f"Benchmark generated: "
        f"{len(cases)} cases"
    )

    # =================================================================
    # CORPUS
    # =================================================================

    corpus = write_benchmark_memory_corpus(
        cases
    )

    print(
        f"Canonical corpus: "
        f"{len(corpus)} memories"
    )

    # =================================================================
    # STORE
    # =================================================================

    store = MemoryStore(
        file_path=str(
            BENCHMARK_MEMORY_FILE
        )
    )

    # =================================================================
    # NLI
    # =================================================================

    print()
    print(
        "Loading NLI model..."
    )

    nli_engine = NLIEngine()

    # =================================================================
    # RUN
    # =================================================================

    diagnostic_cases = cases[
        :DIAGNOSTIC_CASES
    ]

    results = []

    for index, case in enumerate(
        diagnostic_cases,
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

        if index <= 20:

            print_case(
                index,
                case,
                result,
            )

        if index % 10 == 0:

            print()
            print(
                f"PROGRESS: "
                f"{index}/{len(diagnostic_cases)}"
            )

    # =================================================================
    # SUMMARY COUNTERS
    # =================================================================

    total = len(results)

    correct = sum(
        1
        for _, result in results
        if result["correct"]
    )

    incorrect = total - correct

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    # =================================================================
    # STAGE COUNTS
    # =================================================================

    stage_counts = Counter(
        result["stage"]
        for _, result in results
    )

    # =================================================================
    # NLI
    # =================================================================

    nli_results = [
        result
        for _, result in results
        if result["nli_label"]
        is not None
    ]

    nli_labels = Counter(
        result["nli_label"]
        for result in nli_results
    )

    # =================================================================
    # FINAL ACTIONS
    # =================================================================

    predicted_actions = Counter(
        result["predicted"]
        for _, result in results
    )

    expected_actions = Counter(
        result["expected"]
        for _, result in results
    )

    # =================================================================
    # RELATIONSHIPS
    # =================================================================

    relationships = Counter(
        result["relationship"]
        for _, result in results
        if result["relationship"]
        is not None
    )

    # =================================================================
    # NLI CONFIDENCE
    # =================================================================

    if nli_results:

        average_nli_confidence = (
            sum(
                result["nli_confidence"]
                for result in nli_results
            )
            / len(nli_results)
        )

        average_nli_margin = (
            sum(
                result["nli_margin"]
                for result in nli_results
            )
            / len(nli_results)
        )

    else:

        average_nli_confidence = 0
        average_nli_margin = 0

    # =================================================================
    # RAG METRICS
    # =================================================================

    retrieval_cases = [
        result
        for _, result in results
        if result["rank"] is not None
    ]

    recall_1 = sum(
        1
        for result in retrieval_cases
        if result["rank"] <= 1
    )

    recall_5 = sum(
        1
        for result in retrieval_cases
        if result["rank"] <= 5
    )

    recall_10 = sum(
        1
        for result in retrieval_cases
        if result["rank"] <= 10
    )

    # =================================================================
    # PRINT SUMMARY
    # =================================================================

    print()
    print()
    print("#" * 100)
    print("FINAL DIAGNOSTIC RESULTS")
    print("#" * 100)

    print()

    print(
        f"Total cases:              "
        f"{total}"
    )

    print(
        f"Correct:                  "
        f"{correct}"
    )

    print(
        f"Incorrect:                "
        f"{incorrect}"
    )

    print(
        f"Accuracy:                 "
        f"{accuracy:.2f}%"
    )

    # =================================================================
    # RAG
    # =================================================================

    print()
    print("=" * 100)
    print("1. RAG")
    print("=" * 100)

    if retrieval_cases:

        print(
            f"Recall@1:                 "
            f"{recall_1 / len(retrieval_cases) * 100:.2f}%"
        )

        print(
            f"Recall@5:                 "
            f"{recall_5 / len(retrieval_cases) * 100:.2f}%"
        )

        print(
            f"Recall@10:                "
            f"{recall_10 / len(retrieval_cases) * 100:.2f}%"
        )

    print(
        f"RAG failures:             "
        f"{stage_counts['RAG']}"
    )

    print(
        f"RAG top-K failures:       "
        f"{stage_counts['RAG_TOP_K']}"
    )

    # =================================================================
    # SAFETY / ROUTING
    # =================================================================

    print()
    print("=" * 100)
    print("2. SAFETY GATE / ROUTING")
    print("=" * 100)

    print(
        f"Rule decisions:           "
        f"{stage_counts['RULE']}"
    )

    print(
        f"Cases reaching NLI:       "
        f"{stage_counts['DECISION']}"
    )

    # =================================================================
    # NLI
    # =================================================================

    print()
    print("=" * 100)
    print("3. NLI")
    print("=" * 100)

    print(
        f"NLI calls:                "
        f"{len(nli_results)}"
    )

    for label in [
        "contradiction",
        "entailment",
        "neutral",
    ]:

        print(
            f"{label:<25}"
            f"{nli_labels[label]}"
        )

    print(
        f"Average confidence:       "
        f"{average_nli_confidence:.4f}"
    )

    print(
        f"Average margin:            "
        f"{average_nli_margin:.4f}"
    )

    # =================================================================
    # RELATIONSHIPS
    # =================================================================

    print()
    print("=" * 100)
    print("4. RELATIONSHIPS")
    print("=" * 100)

    for relationship, count in (
        relationships.most_common()
    ):

        print(
            f"{relationship:<30}"
            f"{count}"
        )

    # =================================================================
    # EXPECTED ACTIONS
    # =================================================================

    print()
    print("=" * 100)
    print("5. EXPECTED ACTIONS")
    print("=" * 100)

    for action, count in (
        expected_actions.most_common()
    ):

        print(
            f"{action:<25}"
            f"{count}"
        )

    # =================================================================
    # PREDICTED ACTIONS
    # =================================================================

    print()
    print("=" * 100)
    print("6. PREDICTED ACTIONS")
    print("=" * 100)

    for action, count in (
        predicted_actions.most_common()
    ):

        print(
            f"{action:<25}"
            f"{count}"
        )

    # =================================================================
    # LLM FALLBACK
    # =================================================================

    fallback_cases = [
        (case, result)
        for case, result in results
        if result["predicted"]
        == "LLM_FALLBACK"
    ]

    print()
    print("=" * 100)
    print("7. LLM FALLBACK")
    print("=" * 100)

    print(
        f"LLM fallback cases:      "
        f"{len(fallback_cases)}"
    )

    for case, result in fallback_cases[:15]:

        print()

        print(
            f"{case.case_id}"
        )

        print(
            f"Expected:     "
            f"{result['expected']}"
        )

        print(
            f"Relationship: "
            f"{result['relationship']}"
        )

        print(
            f"NLI:          "
            f"{result['nli_label']}"
        )

        print(
            f"NLI conf:     "
            f"{result['nli_confidence']}"
        )

        print(
            f"NLI margin:   "
            f"{result['nli_margin']}"
        )

        print(
            f"Evidence:     "
            f"{result['evidence']}"
        )

    # =================================================================
    # FAILURES
    # =================================================================

    failures = [
        (case, result)
        for case, result in results
        if not result["correct"]
    ]

    print()
    print("=" * 100)
    print("8. FIRST FAILURES")
    print("=" * 100)

    if not failures:

        print(
            "No failures in diagnostic sample."
        )

    else:

        for case, result in failures[:30]:

            print()
            print("-" * 100)

            print(
                f"Case:         "
                f"{case.case_id}"
            )

            print(
                f"Category:     "
                f"{case.category}"
            )

            print(
                f"Expected:     "
                f"{result['expected']}"
            )

            print(
                f"Predicted:    "
                f"{result['predicted']}"
            )

            print(
                f"Stage:        "
                f"{result['stage']}"
            )

            print(
                f"RAG rank:     "
                f"{result['rank']}"
            )

            print(
                f"RAG score:    "
                f"{result['similarity']}"
            )

            print(
                f"Relationship: "
                f"{result['relationship']}"
            )

            if result["nli_label"]:

                print(
                    f"NLI:          "
                    f"{result['nli_label']}"
                )

                print(
                    f"NLI conf:     "
                    f"{result['nli_confidence']}"
                )

                print(
                    f"NLI margin:   "
                    f"{result['nli_margin']}"
                )

            if result["evidence"] is not None:

                print(
                    f"Metadata:     "
                    f"{result['metadata']}"
                )

                print(
                    f"Update:       "
                    f"{result['update_signal']}"
                )

                print(
                    f"Evidence:     "
                    f"{result['evidence']}"
                )

            print()

            print(
                f"OLD: "
                f"{case.old_memory.value}"
            )

            print(
                f"NEW: "
                f"{case.new_memory.value}"
            )

    # =================================================================
    # SUSPICIOUS NLI
    # =================================================================

    suspicious_nli = [
        (case, result)
        for case, result in results
        if result["nli_label"]
        is not None
        and (
            result["nli_confidence"] < 0.80
            or result["nli_margin"] < 0.50
        )
    ]

    print()
    print("=" * 100)
    print("9. SUSPICIOUS / LOW-CONFIDENCE NLI")
    print("=" * 100)

    print(
        f"Suspicious NLI cases:    "
        f"{len(suspicious_nli)}"
    )

    for case, result in suspicious_nli[:20]:

        print()

        print(
            f"Case:         {case.case_id}"
        )

        print(
            f"Category:     {case.category}"
        )

        print(
            f"Expected:     {result['expected']}"
        )

        print(
            f"NLI:          "
            f"{result['nli_label']}"
        )

        print(
            f"Confidence:   "
            f"{result['nli_confidence']}"
        )

        print(
            f"Margin:       "
            f"{result['nli_margin']}"
        )

        print(
            f"Evidence:     "
            f"{result['evidence']}"
        )

        print(
            f"Predicted:    "
            f"{result['predicted']}"
        )

    # =================================================================
    # IMPORTANT
    # =================================================================

    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)

    # We intentionally do NOT require high accuracy yet.
    # This test is diagnosing the pipeline.
    assert total > 0