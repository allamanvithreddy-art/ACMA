import json
from pathlib import Path

import pytest

from memory.store import MemoryStore
from memory.schema import Memory, MemoryQuery
from conflict.safety_gate import evaluate_rule_safety


# ============================================================
# REALISTIC LARGE RAG -> SAFETY GATE STRESS TEST
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TEST_MEMORY_FILE = BASE_DIR / "data" / "stress_test_memories.json"


def make_memory(
    memory_id,
    attribute,
    value,
    scope="general",
    context="general",
    time=None,
    confidence=0.9,
    importance=0.5,
):
    return Memory(
        memory_id=memory_id,
        subject="user",
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        time=time,
        source="user",
        confidence=confidence,
        importance=importance,
    )


# ============================================================
# REALISTIC STORED MEMORIES
# ============================================================

STORED_MEMORIES = [

    # ---------------- ACMA ----------------

    make_memory(
        "m01",
        "project",
        "User is developing the ACMA memory architecture.",
        context="ACMA project",
    ),

    make_memory(
        "m02",
        "backend",
        "FastAPI is used for the ACMA backend.",
        context="ACMA project",
    ),

    make_memory(
        "m03",
        "programming_language",
        "Python is used for the ACMA project.",
        context="ACMA project",
    ),

    make_memory(
        "m04",
        "database",
        "SQLite is used for ACMA metadata storage.",
        context="ACMA project",
    ),

    make_memory(
        "m05",
        "retrieval",
        "Sentence Transformers are used for semantic memory retrieval.",
        context="ACMA project",
    ),

    make_memory(
        "m06",
        "memory_type",
        "ACMA uses working, episodic and semantic memory concepts.",
        context="ACMA architecture",
    ),

    make_memory(
        "m07",
        "nli",
        "NLI is used to verify potentially conflicting memories.",
        context="ACMA conflict resolution",
    ),

    make_memory(
        "m08",
        "conflict_resolution",
        "ACMA uses Resolve, Preserve and Ask decisions.",
        context="ACMA architecture",
    ),

    make_memory(
        "m09",
        "vector_retrieval",
        "Semantic similarity is used to retrieve related memories.",
        context="ACMA retrieval",
    ),

    make_memory(
        "m10",
        "paper",
        "The ACMA project is being prepared as a conference paper.",
        context="ACMA research",
    ),

    make_memory(
        "m11",
        "deadline",
        "The ACMA paper deadline is September 20.",
        context="ACMA paper",
        time="2026-09-20",
    ),

    make_memory(
        "m12",
        "interface_theme",
        "The ACMA application normally uses dark mode.",
        context="ACMA interface",
    ),

    # ---------------- DSA ----------------

    make_memory(
        "m13",
        "programming_language",
        "Java is used for DSA coursework.",
        context="DSA coursework",
    ),

    make_memory(
        "m14",
        "learning",
        "User is learning arrays and ArrayList in Java.",
        context="DSA coursework",
    ),

    make_memory(
        "m15",
        "learning",
        "User is learning HashMap traversal in Java.",
        context="DSA coursework",
    ),

    make_memory(
        "m16",
        "learning",
        "User is practicing binary search trees.",
        context="DSA coursework",
    ),

    make_memory(
        "m17",
        "learning",
        "User is practicing linked list problems.",
        context="DSA coursework",
    ),

    # ---------------- ATTENDANCE ----------------

    make_memory(
        "m18",
        "project",
        "User is developing a Student Attendance ERP.",
        context="attendance project",
    ),

    make_memory(
        "m19",
        "face_detection",
        "SCRFD is used for face detection.",
        context="attendance project",
    ),

    make_memory(
        "m20",
        "face_recognition",
        "InsightFace embeddings are used for face recognition.",
        context="attendance project",
    ),

    make_memory(
        "m21",
        "database",
        "MySQL is used for attendance records.",
        context="attendance project",
    ),

    make_memory(
        "m22",
        "backend",
        "Node.js and Express are used for the attendance backend.",
        context="attendance project",
    ),

    # ---------------- GENERAL SOFTWARE ----------------

    make_memory(
        "m23",
        "database",
        "PostgreSQL is being studied for future projects.",
        context="database learning",
    ),

    make_memory(
        "m24",
        "framework",
        "React is used for the attendance portal interface.",
        context="attendance project",
    ),

    make_memory(
        "m25",
        "api",
        "The attendance portal backend runs on port 5000.",
        context="attendance project",
    ),

    make_memory(
        "m26",
        "authentication",
        "Students log in using their roll number.",
        context="attendance portal",
    ),

    make_memory(
        "m27",
        "attendance_rule",
        "Parents receive an alert when attendance falls below 75 percent.",
        context="attendance portal",
    ),

    # ---------------- FOOD / EVENTS ----------------

    make_memory(
        "m28",
        "food_preference",
        "User generally follows a vegetarian diet.",
        context="general preference",
    ),

    make_memory(
        "m29",
        "food_event",
        "User ate chicken at a wedding.",
        scope="specific_event",
        context="wedding",
    ),

    # ---------------- LOCATION ----------------

    make_memory(
        "m30",
        "location",
        "User usually studies at home.",
        context="general routine",
    ),

    make_memory(
        "m31",
        "location",
        "User studied at the university library yesterday.",
        scope="specific_event",
        context="library visit",
    ),
]


# ============================================================
# NEW INCOMING MEMORIES
# ============================================================

CASES = [

    # SAFE / DUPLICATE
    {
        "name": "exact ACMA project duplicate",
        "new": make_memory(
            "new01",
            "project",
            "User is developing the ACMA memory architecture.",
            context="ACMA project",
        ),
        "expected": "rule_action",
    },

    # UNSAFE / EVENT
    {
        "name": "vegetarian vs chicken event",
        "new": make_memory(
            "new02",
            "food_event",
            "User ate chicken at a wedding.",
            scope="specific_event",
            context="wedding",
        ),
        "expected": "nli",
    },

    # UNSAFE / UPDATE
    {
        "name": "FastAPI changed to Flask",
        "new": make_memory(
            "new03",
            "backend",
            "User has switched the ACMA backend from FastAPI to Flask.",
            context="ACMA project",
        ),
        "expected": "nli",
    },

    # UNSAFE / UPDATE
    {
        "name": "Python preference changed",
        "new": make_memory(
            "new04",
            "programming_language",
            "User no longer wants to use Python for ACMA.",
            context="ACMA project",
        ),
        "expected": "nli",
    },

    # UNSAFE / DEADLINE
    {
        "name": "deadline changed",
        "new": make_memory(
            "new05",
            "deadline",
            "The ACMA paper deadline has been moved to September 27.",
            context="ACMA paper",
            time="2026-09-27",
        ),
        "expected": "nli",
    },

    # UNSAFE / TEMPORARY EXCEPTION
    {
        "name": "dark mode temporary exception",
        "new": make_memory(
            "new06",
            "interface_theme",
            "For tomorrow's presentation, use light mode.",
            scope="specific_event",
            context="presentation",
        ),
        "expected": "nli",
    },

    # SAFE / DIFFERENT ATTRIBUTE
    {
        "name": "ACMA NLI attribute",
        "new": make_memory(
            "new07",
            "testing",
            "The ACMA backend is being tested locally.",
            context="ACMA project",
        ),
        "expected": "rule_action",
    },

    # SAFE / DIFFERENT SUBJECT-LIKE CONTEXT
    {
        "name": "attendance face detection",
        "new": make_memory(
            "new08",
            "face_detection",
            "SCRFD is used for face detection.",
            context="attendance project",
        ),
        "expected": "rule_action",
    },

    # UNSAFE / NEGATION
    {
        "name": "MySQL changed",
        "new": make_memory(
            "new09",
            "database",
            "MySQL is no longer used for attendance records.",
            context="attendance project",
        ),
        "expected": "nli",
    },

    # SAFE / SAME VALUE
    {
        "name": "Java DSA duplicate",
        "new": make_memory(
            "new10",
            "programming_language",
            "Java is used for DSA coursework.",
            context="DSA coursework",
        ),
        "expected": "rule_action",
    },

    # UNSAFE / POSSIBLE CONFLICT
    {
        "name": "React replaced",
        "new": make_memory(
            "new11",
            "framework",
            "Angular is now used for the attendance portal interface.",
            context="attendance project",
        ),
        "expected": "nli",
    },

    # SAFE / DIFFERENT ATTRIBUTE
    {
        "name": "HashMap learning",
        "new": make_memory(
            "new12",
            "learning",
            "User is learning HashMap traversal in Java.",
            context="DSA coursework",
        ),
        "expected": "rule_action",
    },

    # UNSAFE / TEMPORAL
    {
        "name": "temporary library visit",
        "new": make_memory(
            "new13",
            "location",
            "User is studying at the university library today.",
            scope="specific_event",
            context="library visit",
            time="2026-09-22",
        ),
        "expected": "nli",
    },

    # SAFE / DIFFERENT ATTRIBUTE
    {
        "name": "attendance login",
        "new": make_memory(
            "new14",
            "authentication",
            "Students log in using their roll number.",
            context="attendance portal",
        ),
        "expected": "rule_action",
    },

    # UNSAFE / CONFLICT
    {
        "name": "ACMA database changed",
        "new": make_memory(
            "new15",
            "database",
            "PostgreSQL is now used for ACMA metadata storage.",
            context="ACMA project",
        ),
        "expected": "nli",
    },

    # SAFE / UNRELATED
    {
        "name": "DSA linked list",
        "new": make_memory(
            "new16",
            "learning",
            "User is practicing linked list problems.",
            context="DSA coursework",
        ),
        "expected": "rule_action",
    },
]


# ============================================================
# WRITE REALISTIC MEMORY DATA
# ============================================================

def create_memory_file():
    TEST_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TEST_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            [memory.to_dict() for memory in STORED_MEMORIES],
            f,
            indent=2,
        )


# ============================================================
# STRESS TEST
# ============================================================

def test_large_rag_to_safety_gate():

    create_memory_file()

    store = MemoryStore(TEST_MEMORY_FILE)

    total = 0
    passed = 0

    print("\n")
    print("=" * 90)
    print("LARGE RAG -> SAFETY GATE STRESS TEST")
    print("=" * 90)

    for case in CASES:

        new_memory = case["new"]

        query = MemoryQuery(
            subject=new_memory.subject,
            attribute=new_memory.attribute,
            value=new_memory.value,
            scope=new_memory.scope,
            context=new_memory.context,
            time=new_memory.time,
        )

        results = store.retrieve_related_memories(
            query=query,
            top_k=5,
            similarity_threshold=0.30,
        )

        print("\n" + "-" * 90)
        print(case["name"])
        print("NEW:", new_memory.value)

        if not results:
            print("NO RAG RESULTS")
            assert False, f"No RAG results for: {case['name']}"

        case_passed = False

        for old_memory, similarity in results:

            safety = evaluate_rule_safety(
                old_memory=old_memory,
                new_memory=new_memory,
                similarity=similarity,
            )

            print(
                f"\nOLD: {old_memory.value}"
                f"\nSimilarity: {similarity:.4f}"
                f"\nRelationship: {safety.relationship}"
                f"\nSafety: {safety.safe}"
                f"\nNext: {safety.next_stage}"
                f"\nAction: {safety.action_hint}"
            )

            # We consider the case successful if ANY strongly
            # relevant retrieved memory reaches the expected stage.
            if safety.next_stage == case["expected"]:
                case_passed = True

        total += 1

        if case_passed:
            passed += 1
            print("\nPASS")
        else:
            print(
                f"\nFAIL - expected at least one "
                f"retrieved candidate to reach: {case['expected']}"
            )

    print("\n")
    print("=" * 90)
    print(f"RESULT: {passed}/{total}")
    print("=" * 90)

    assert passed == total