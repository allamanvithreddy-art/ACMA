import json
from pathlib import Path

import pytest

from memory.schema import Memory
from memory.store import MemoryStore

from conflict.scope_rules import classify_relationship
from conflict.safety_gate import evaluate_rule_safety
from conflict.update_detector import detect_update_signals
from conflict.nli_engine import NLIEngine
from conflict.metadata import metadata_agreement
from conflict.evidence_score import calculate_evidence_score
from conflict.decision_policy import decide_action


# ============================================================
# Configuration
# ============================================================

TOP_K = 5
SIMILARITY_THRESHOLD = 0.30

# Test-only memory database.
#
# IMPORTANT:
# This is created automatically by the test.
# It does NOT modify the real sample memory database.
#
TEST_MEMORY_FILE = Path(
    "data/test_pipeline_memories.json"
)


# ============================================================
# Test memory dataset
# ============================================================

TEST_MEMORIES = [
    # --------------------------------------------------------
    # CASE 1
    # Java -> Python
    # --------------------------------------------------------
    {
        "memory_id": "test_java",
        "subject": "user",
        "attribute": "language",
        "value": "I use Java for programming",
        "scope": "general",
        "context": "general",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.5,
        "active": True,
        "version": 1,
    },

    # --------------------------------------------------------
    # CASE 2
    # Dark -> Light
    # --------------------------------------------------------
    {
        "memory_id": "test_dark_mode",
        "subject": "user",
        "attribute": "theme",
        "value": "I prefer dark mode",
        "scope": "general",
        "context": "general",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.5,
        "active": True,
        "version": 1,
    },

    # --------------------------------------------------------
    # CASE 3
    # SQLite -> PostgreSQL
    # --------------------------------------------------------
    {
        "memory_id": "test_sqlite",
        "subject": "user",
        "attribute": "database",
        "value": "The application uses SQLite",
        "scope": "general",
        "context": "application",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.5,
        "active": True,
        "version": 1,
    },

    # --------------------------------------------------------
    # CASE 4
    # Vegetarian constraint
    # --------------------------------------------------------
    {
        "memory_id": "test_vegetarian",
        "subject": "user",
        "attribute": "diet",
        "value": "I am vegetarian",
        "scope": "general",
        "context": "general",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.5,
        "active": True,
        "version": 1,
    },

    # --------------------------------------------------------
    # CASE 5
    # Python in ACMA
    # --------------------------------------------------------
    {
        "memory_id": "test_python_acma",
        "subject": "user",
        "attribute": "language",
        "value": "I use Python for ACMA",
        "scope": "general",
        "context": "ACMA",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.5,
        "active": True,
        "version": 1,
    },

    # --------------------------------------------------------
    # CASE 5
    # Python in DSA
    #
    # This is intentionally another context so RAG has
    # multiple semantically similar memories to distinguish.
    # --------------------------------------------------------
    {
        "memory_id": "test_python_dsa",
        "subject": "user",
        "attribute": "language",
        "value": "I use Python for DSA coursework",
        "scope": "general",
        "context": "DSA",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.5,
        "active": True,
        "version": 1,
    },

    # --------------------------------------------------------
    # CASE 6
    # Exact duplicate
    # --------------------------------------------------------
    {
        "memory_id": "test_python_acma_duplicate",
        "subject": "user",
        "attribute": "language",
        "value": "I use Python for ACMA",
        "scope": "general",
        "context": "ACMA",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.5,
        "active": True,
        "version": 1,
    },

    # --------------------------------------------------------
    # CASE 7
    # FastAPI -> Flask
    # --------------------------------------------------------
    {
        "memory_id": "test_fastapi",
        "subject": "user",
        "attribute": "backend",
        "value": "The ACMA backend uses FastAPI",
        "scope": "general",
        "context": "ACMA",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.5,
        "active": True,
        "version": 1,
    },

    # --------------------------------------------------------
    # CASE 8
    # Deadline
    # --------------------------------------------------------
    {
        "memory_id": "test_deadline",
        "subject": "user",
        "attribute": "deadline",
        "value": "The ACMA paper deadline is September 20",
        "scope": "general",
        "context": "ACMA paper",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.5,
        "active": True,
        "version": 1,
    },

    # --------------------------------------------------------
    # Distractor memories
    #
    # These make the RAG test more realistic.
    # --------------------------------------------------------

    {
        "memory_id": "test_java_dsa",
        "subject": "user",
        "attribute": "language",
        "value": "I sometimes study Java for DSA",
        "scope": "general",
        "context": "DSA",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.3,
        "active": True,
        "version": 1,
    },

    {
        "memory_id": "test_light_general",
        "subject": "user",
        "attribute": "lighting",
        "value": "I like bright rooms",
        "scope": "general",
        "context": "general",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.3,
        "active": True,
        "version": 1,
    },

    {
        "memory_id": "test_mysql",
        "subject": "user",
        "attribute": "database",
        "value": "I have used MySQL",
        "scope": "general",
        "context": "database",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.3,
        "active": True,
        "version": 1,
    },

    {
        "memory_id": "test_fastapi_other",
        "subject": "user",
        "attribute": "framework",
        "value": "FastAPI is useful for Python APIs",
        "scope": "general",
        "context": "learning",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.3,
        "active": True,
        "version": 1,
    },

    {
        "memory_id": "test_project_deadline",
        "subject": "user",
        "attribute": "deadline",
        "value": "The DSA project deadline is October 10",
        "scope": "general",
        "context": "DSA project",
        "time": None,
        "source": "user",
        "confidence": 1.0,
        "importance": 0.3,
        "active": True,
        "version": 1,
    },
]


# ============================================================
# Module-level NLI engine
#
# Loading bart-large-mnli is expensive.
# We therefore load it once for the whole test module.
# ============================================================

NLI_ENGINE = None


# ============================================================
# Test dataset setup
# ============================================================

@pytest.fixture(scope="module", autouse=True)
def prepare_test_memory_file():
    """
    Create an isolated memory file for the complete
    architecture test.

    This guarantees that the RAG stage has known memories
    for every scenario.
    """

    TEST_MEMORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        TEST_MEMORY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            TEST_MEMORIES,
            file,
            indent=4,
        )

    yield

    # --------------------------------------------------------
    # Remove the test file after the test module finishes.
    # --------------------------------------------------------

    if TEST_MEMORY_FILE.exists():
        TEST_MEMORY_FILE.unlink()


# ============================================================
# Helpers
# ============================================================

def create_memory(
    memory_id,
    attribute,
    value,
    scope="general",
    context="general",
    time=None,
):
    """
    Create a real ACMA Memory object.
    """

    return Memory(
        memory_id=memory_id,
        subject="user",
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        time=time,
        source="user",
        confidence=1.0,
        importance=0.5,
    )


# ============================================================
# Incoming memory creator
# ============================================================

def create_new_memory(
    value,
    attribute="general",
    scope="general",
    context="general",
    time=None,
):
    """
    Create the incoming memory that enters ACMA.
    """

    return create_memory(
        memory_id="incoming",
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        time=time,
    )


# ============================================================
# NLI engine loader
# ============================================================

def get_nli_engine():
    """
    Lazily load the real NLI engine.

    This prevents the model from being loaded for cases
    that never reach NLI.
    """

    global NLI_ENGINE

    if NLI_ENGINE is None:
        NLI_ENGINE = NLIEngine()

    return NLI_ENGINE


# ============================================================
# Complete pipeline
# ============================================================

def run_full_pipeline(
    new_memory,
):
    """
    Run the actual ACMA pipeline up to Decision Policy.

    Architecture:

        NEW MEMORY
             ↓
        NORMALIZATION
             ↓
             RAG
             ↓
        SAFETY GATE
             ↓
       ┌─────┴─────┐
       ↓           ↓
      SAFE        UNSAFE
       ↓           ↓
    direct        NLI
       │           │
       └─────┬─────┘
             ↓
    RELATIONSHIP / CONTEXT / UPDATE
             ↓
       EVIDENCE SCORE
             ↓
      DECISION POLICY
             ↓
    FINAL ACTION
    """

    # ========================================================
    # 1. RAG
    # ========================================================

    # IMPORTANT:
    # Use the actual MemoryStore implementation.
    #
    # The test dataset is passed explicitly so the test is
    # deterministic and isolated from the application's
    # normal sample memory file.

    store = MemoryStore(
        TEST_MEMORY_FILE
    )

    retrieved = store.retrieve_related_memories(
        new_memory,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    if not retrieved:

        return {
            "status": "RAG_FAILURE",
            "new_memory": new_memory,
            "retrieved": [],
            "candidates": [],
            "final": None,
        }

    # ========================================================
    # 2. Evaluate every RAG candidate
    # ========================================================

    candidate_results = []

    for rank, (
        old_memory,
        similarity,
    ) in enumerate(
        retrieved,
        start=1,
    ):

        # ----------------------------------------------------
        # Relationship classification
        # ----------------------------------------------------

        relationship_result = classify_relationship(
            old_memory,
            new_memory,
        )

        relationship = relationship_result[
            "relationship"
        ]

        # ----------------------------------------------------
        # Safety Gate
        #
        # Similarity comes directly from RAG.
        # ----------------------------------------------------

        safety_result = evaluate_rule_safety(
            old_memory=old_memory,
            new_memory=new_memory,
            similarity=similarity,
        )

        # ----------------------------------------------------
        # Update Detector
        #
        # Calculated from the real incoming memory.
        # ----------------------------------------------------

        update_info = detect_update_signals(
            new_memory.value
        )

        update_signal = float(
            update_info.get(
                "update_strength",
                0.0,
            )
        )

        # ----------------------------------------------------
        # NLI
        #
        # NLI is called ONLY when the Safety Gate marks the
        # candidate as unsafe.
        # ----------------------------------------------------

        nli_result = None

        if not safety_result["safe"]:

            nli_engine = get_nli_engine()

            nli_result = nli_engine.compare(
                old_memory,
                new_memory,
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_score = metadata_agreement(
            old_memory,
            new_memory,
        )

        # ----------------------------------------------------
        # NLI evidence
        # ----------------------------------------------------

        if nli_result is not None:

            nli_confidence = float(
                nli_result.get(
                    "confidence",
                    0.0,
                )
            )

            nli_margin = float(
                nli_result.get(
                    "margin",
                    0.0,
                )
            )

        else:

            nli_confidence = 0.0
            nli_margin = 0.0

        # ----------------------------------------------------
        # Evidence Score
        #
        # Every value comes from the real pipeline.
        # ----------------------------------------------------

        evidence_score = calculate_evidence_score(
            nli_confidence=nli_confidence,
            nli_margin=nli_margin,
            similarity=similarity,
            metadata_agreement=metadata_score,
            update_signal=update_signal,
            source_reliability=1.0,
        )

        # ----------------------------------------------------
        # Decision Policy
        # ----------------------------------------------------

        action = decide_action(
            relationship=relationship,
            nli_result=nli_result,
            metadata_agreement=metadata_score,
            similarity=similarity,
            update_signal=update_signal,
            evidence_score=evidence_score,
            old_memory=old_memory,
            new_memory=new_memory,
            action_hint=relationship_result.get(
                "action_hint"
            ),
            safety_result=safety_result,
            update_info=update_info,
        )

        # ----------------------------------------------------
        # Store candidate result
        # ----------------------------------------------------

        candidate_results.append(
            {
                "rank": rank,
                "old_memory": old_memory,
                "similarity": similarity,
                "relationship": relationship_result,
                "safety": safety_result,
                "update": update_info,
                "nli": nli_result,
                "metadata": metadata_score,
                "evidence": evidence_score,
                "action": action,
            }
        )

    # ========================================================
    # 3. Select final candidate
    # ========================================================

    def candidate_priority(result):

        relationship = result[
            "relationship"
        ]["relationship"]

        relationship_priority = {
            "duplicate": 5,
            "constraint_violation": 5,
            "possible_update": 5,
            "possible_conflict": 4,
            "possible_general_conflict": 4,
            "different_context": 3,
            "specific_event": 3,
            "different_attribute": 1,
            "different_subject": 0,
        }

        return (
            relationship_priority.get(
                relationship,
                0,
            ),
            result["evidence"],
            result["similarity"],
        )

    final_candidate = max(
        candidate_results,
        key=candidate_priority,
    )

    return {
        "status": "SUCCESS",
        "new_memory": new_memory,
        "retrieved": retrieved,
        "candidates": candidate_results,
        "final": final_candidate,
    }


# ============================================================
# Diagnostic printer
# ============================================================

def print_pipeline_result(
    name,
    result,
):
    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    new_memory = result[
        "new_memory"
    ]

    print()
    print("NEW MEMORY")
    print("-" * 100)
    print(
        new_memory.value
    )

    # --------------------------------------------------------
    # RAG failure
    # --------------------------------------------------------

    if result["status"] == "RAG_FAILURE":

        print()
        print("RAG")
        print("-" * 100)
        print(
            "NO RELATED MEMORY FOUND"
        )

        return

    # --------------------------------------------------------
    # RAG results
    # --------------------------------------------------------

    print()
    print("RAG RESULTS")
    print("-" * 100)

    for candidate in result[
        "candidates"
    ]:

        old_memory = candidate[
            "old_memory"
        ]

        print()
        print(
            f"RANK: {candidate['rank']}"
        )

        print(
            f"OLD MEMORY: "
            f"{old_memory.value}"
        )

        print(
            f"SIMILARITY: "
            f"{candidate['similarity']:.4f}"
        )

        print(
            "RELATIONSHIP: "
            f"{candidate['relationship']['relationship']}"
        )

        print(
            "SAFETY: "
            f"{candidate['safety']}"
        )

        print(
            "UPDATE: "
            f"{candidate['update']}"
        )

        if candidate["nli"] is None:

            print(
                "NLI: NOT CALLED"
            )

        else:

            print(
                "NLI: "
                f"{candidate['nli']}"
            )

        print(
            "METADATA: "
            f"{candidate['metadata']:.4f}"
        )

        print(
            "EVIDENCE: "
            f"{candidate['evidence']:.4f}"
        )

        print(
            "ACTION: "
            f"{candidate['action']}"
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    final = result[
        "final"
    ]

    print()
    print("=" * 100)
    print("FINAL DECISION")
    print("=" * 100)

    print(
        "Selected RAG rank:",
        final["rank"],
    )

    print(
        "Selected old memory:",
        final["old_memory"].value,
    )

    print(
        "Relationship:",
        final["relationship"]["relationship"],
    )

    print(
        "Similarity:",
        round(
            final["similarity"],
            4,
        ),
    )

    print(
        "Evidence:",
        round(
            final["evidence"],
            4,
        ),
    )

    print(
        "Final action:",
        final["action"],
    )


# ============================================================
# Test helper
# ============================================================

def assert_success(result):
    """
    Common assertion for all complete-pipeline tests.
    """

    assert result[
        "status"
    ] == "SUCCESS"

    assert result[
        "retrieved"
    ]

    assert result[
        "candidates"
    ]

    assert result[
        "final"
    ] is not None


# ============================================================
# CASE 1
#
# Java -> Python
#
# Expected:
# - RAG finds Java memory
# - Safety Gate detects risky update
# - NLI runs
# - Decision Policy produces a valid action
# ============================================================

def test_complete_pipeline_java_to_python():

    new_memory = create_new_memory(
        "I now use Python instead of Java",
        attribute="language",
    )

    result = run_full_pipeline(
        new_memory
    )

    print_pipeline_result(
        "CASE 1 — JAVA → PYTHON",
        result,
    )

    assert_success(result)

    final = result[
        "final"
    ]

    assert final[
        "nli"
    ] is not None

    assert final[
        "action"
    ] in {
        "Resolve",
        "Ask",
        "Preserve",
        "LLM_FALLBACK",
    }


# ============================================================
# CASE 2
#
# Dark -> Light
# ============================================================

def test_complete_pipeline_dark_to_light():

    new_memory = create_new_memory(
        "I now prefer light mode",
        attribute="theme",
    )

    result = run_full_pipeline(
        new_memory
    )

    print_pipeline_result(
        "CASE 2 — DARK → LIGHT",
        result,
    )

    assert_success(result)

    final = result[
        "final"
    ]

    assert final[
        "nli"
    ] is not None

    assert final[
        "action"
    ] in {
        "Resolve",
        "Ask",
        "Preserve",
        "LLM_FALLBACK",
    }


# ============================================================
# CASE 3
#
# SQLite -> PostgreSQL
# ============================================================

def test_complete_pipeline_sqlite_to_postgresql():

    new_memory = create_new_memory(
        "We migrated the application database to PostgreSQL",
        attribute="database",
        context="application",
    )

    result = run_full_pipeline(
        new_memory
    )

    print_pipeline_result(
        "CASE 3 — SQLITE → POSTGRESQL",
        result,
    )

    assert_success(result)

    final = result[
        "final"
    ]

    assert final[
        "nli"
    ] is not None

    assert final[
        "action"
    ] in {
        "Resolve",
        "Ask",
        "Preserve",
        "LLM_FALLBACK",
    }


# ============================================================
# CASE 4
#
# Vegetarian -> Chicken event
#
# Important:
#
# The event conflicts with a general vegetarian constraint.
#
# It should NOT automatically replace the vegetarian memory.
#
# Expected architecture behavior:
#
# relationship = constraint_violation
# NLI = called
# final action can be Ask / Resolve / etc depending on
# evidence and policy.
# ============================================================

def test_complete_pipeline_vegetarian_chicken():

    new_memory = create_new_memory(
        "I ate chicken at a wedding",
        attribute="diet",
        scope="event",
        context="wedding",
    )

    result = run_full_pipeline(
        new_memory
    )

    print_pipeline_result(
        "CASE 4 — VEGETARIAN → CHICKEN EVENT",
        result,
    )

    assert_success(result)

    final = result[
        "final"
    ]

    assert final[
        "relationship"
    ]["relationship"] == (
        "constraint_violation"
    )

    assert final[
        "nli"
    ] is not None

    assert final[
        "action"
    ] in {
        "Ask",
        "Preserve",
        "Resolve",
        "LLM_FALLBACK",
    }


# ============================================================
# CASE 5
#
# Python ACMA vs Python DSA
#
# Same language.
# Different context.
#
# The architecture should recognize the context distinction.
# ============================================================

def test_complete_pipeline_context_exception():

    new_memory = create_new_memory(
        "I use Python for DSA coursework",
        attribute="language",
        context="DSA",
    )

    result = run_full_pipeline(
        new_memory
    )

    print_pipeline_result(
        "CASE 5 — PYTHON ACMA vs PYTHON DSA",
        result,
    )

    assert_success(result)

    final = result[
        "final"
    ]

    assert final[
        "nli"
    ] is not None

    assert final[
        "action"
    ] in {
        "Preserve",
        "Ask",
        "Resolve",
        "LLM_FALLBACK",
    }


# ============================================================
# CASE 6
#
# Exact duplicate
#
# Expected:
# relationship = duplicate
# action = Ignore
# ============================================================

def test_complete_pipeline_exact_duplicate():

    new_memory = create_new_memory(
        "I use Python for ACMA",
        attribute="language",
        context="ACMA",
    )

    result = run_full_pipeline(
        new_memory
    )

    print_pipeline_result(
        "CASE 6 — EXACT DUPLICATE",
        result,
    )

    assert_success(result)

    final = result[
        "final"
    ]

    assert final[
        "relationship"
    ]["relationship"] == "duplicate"

    assert final[
        "action"
    ] == "Ignore"


# ============================================================
# CASE 7
#
# FastAPI -> Flask
# ============================================================

def test_complete_pipeline_fastapi_to_flask():

    new_memory = create_new_memory(
        "I switched the ACMA backend from FastAPI to Flask",
        attribute="backend",
        context="ACMA",
    )

    result = run_full_pipeline(
        new_memory
    )

    print_pipeline_result(
        "CASE 7 — FASTAPI → FLASK",
        result,
    )

    assert_success(result)

    final = result[
        "final"
    ]

    assert final[
        "nli"
    ] is not None

    assert final[
        "action"
    ] in {
        "Resolve",
        "Ask",
        "Preserve",
        "LLM_FALLBACK",
    }


# ============================================================
# CASE 8
#
# Deadline update
# ============================================================

def test_complete_pipeline_deadline_update():

    new_memory = create_new_memory(
        "The ACMA paper deadline moved to September 27",
        attribute="deadline",
        context="ACMA paper",
    )

    result = run_full_pipeline(
        new_memory
    )

    print_pipeline_result(
        "CASE 8 — DEADLINE UPDATE",
        result,
    )

    assert_success(result)

    final = result[
        "final"
    ]

    assert final[
        "nli"
    ] is not None

    assert final[
        "action"
    ] in {
        "Resolve",
        "Ask",
        "Preserve",
        "LLM_FALLBACK",
    }