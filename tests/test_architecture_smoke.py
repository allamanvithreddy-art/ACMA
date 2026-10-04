"""
ACMA Expanded Architecture Smoke Test
======================================

This is a large real-pipeline stress test.

The test DOES NOT manually provide:
    - retrieval scores
    - NLI labels
    - NLI confidence
    - NLI margin
    - NLI class scores
    - evidence scores
    - decision confidence

Those values are produced by the actual ACMA system.

The test DOES provide:
    - memory corpus
    - new-memory inputs
    - expected behavior for selected targeted cases
    - structural invariants for broad stress cases

Architecture exercised:

    New Memory
         ↓
    RAG Retrieval
         ↓
    Relationship / Metadata / Update
         ↓
    Safety Gate
         │
         ├── SAFE ───────────────┐
         │                       ↓
         └── UNSAFE → Real NLI → Evidence
                                  ↓
                              Decision
                                  ↓
                       Ignore / Preserve /
                       Resolve / Ask / Store
"""

import json
import random

import pytest

from memory.schema import Memory
from memory.store import MemoryStore
from conflict.pipeline import ACPipeline


# ============================================================================
# HELPERS
# ============================================================================

def make_memory(
    memory_id,
    subject,
    attribute,
    value,
    scope="general",
    context="general",
    time=None,
):
    return Memory(
        memory_id=memory_id,
        subject=subject,
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        time=time,
        source="user",
    )


def memory_to_dict(memory):
    return {
        "memory_id": memory.memory_id,
        "subject": memory.subject,
        "attribute": memory.attribute,
        "value": memory.value,
        "scope": memory.scope,
        "context": memory.context,
        "time": memory.time,
        "source": memory.source,
        "confidence": memory.confidence,
        "importance": memory.importance,
        "active": memory.active,
        "version": memory.version,
        "supersedes": memory.supersedes,
        "superseded_by": memory.superseded_by,
        "relationship_ids": memory.relationship_ids,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "last_decision": memory.last_decision,
        "last_decision_reason": memory.last_decision_reason,
        "last_decision_confidence": memory.last_decision_confidence,
    }


def run_rag_pipeline(
    pipeline,
    memory_store,
    new_memory,
):
    """
    Real RAG retrieval followed by the real ACMA pipeline.

    similarity_threshold=0.0 means:
        Do not artificially filter candidates during this stress test.

    The actual retrieval scores still come entirely from MemoryStore.
    """

    retrieved = memory_store.retrieve_related_memories(
        query=new_memory,
        top_k=5,
        similarity_threshold=0.0,
    )

    result = pipeline.evaluate_retrieved(
        new_memory=new_memory,
        retrieved=retrieved,
    )

    return retrieved, result


def print_result(
    name,
    retrieved,
    result,
):
    print("\n")
    print("=" * 110)
    print(name)
    print("=" * 110)

    print("\nRAG RESULTS:")

    for memory, score in retrieved:
        print(
            f"  {memory.memory_id:<30}"
            f" | score={score:<7}"
            f" | {memory.value}"
        )

    chosen = result.get("chosen")

    if chosen is not None:

        print("\nCHOSEN RESULT:")

        print(
            "  relationship =",
            chosen["relationship"]["relationship"],
        )

        print(
            "  safety       =",
            chosen["safety"],
        )

        print(
            "  nli          =",
            chosen["nli"],
        )

        print(
            "  evidence     =",
            chosen["evidence"],
        )

        print(
            "  decision     =",
            chosen["decision"],
        )

    print("\nFINAL ACTION     =", result["action"])
    print("FINAL CONFIDENCE =", result["confidence"])
    print("NLI CALLS       =", result["nli_calls"])


# ============================================================================
# LARGE AND DIVERSE MEMORY CORPUS
# ============================================================================

@pytest.fixture(scope="module")
def test_memories():

    memories = [

        # ------------------------------------------------------------------
        # PROGRAMMING LANGUAGES
        # ------------------------------------------------------------------

        make_memory(
            "m_java",
            "user",
            "programming_language",
            "I use Java",
        ),

        make_memory(
            "m_python",
            "user",
            "programming_language",
            "I use Python",
        ),

        make_memory(
            "m_cpp",
            "user",
            "programming_language",
            "I use C++",
        ),

        make_memory(
            "m_javascript",
            "user",
            "programming_language",
            "I use JavaScript",
        ),

        make_memory(
            "m_typescript",
            "user",
            "programming_language",
            "I use TypeScript",
        ),

        make_memory(
            "m_go",
            "user",
            "programming_language",
            "I use Go",
        ),

        make_memory(
            "m_rust",
            "user",
            "programming_language",
            "I use Rust",
        ),

        # ------------------------------------------------------------------
        # BACKEND
        # ------------------------------------------------------------------

        make_memory(
            "m_fastapi",
            "user",
            "backend",
            "I use FastAPI",
        ),

        make_memory(
            "m_django",
            "user",
            "backend",
            "I use Django",
        ),

        make_memory(
            "m_flask",
            "user",
            "backend",
            "I use Flask",
        ),

        make_memory(
            "m_node",
            "user",
            "backend",
            "I use Node.js",
        ),

        make_memory(
            "m_spring",
            "user",
            "backend",
            "I use Spring Boot",
        ),

        # ------------------------------------------------------------------
        # DATABASES
        # ------------------------------------------------------------------

        make_memory(
            "m_postgresql",
            "user",
            "database",
            "I use PostgreSQL",
        ),

        make_memory(
            "m_mysql",
            "user",
            "database",
            "I use MySQL",
        ),

        make_memory(
            "m_sqlite",
            "user",
            "database",
            "I use SQLite",
        ),

        make_memory(
            "m_mongodb",
            "user",
            "database",
            "I use MongoDB",
        ),

        make_memory(
            "m_oracle",
            "user",
            "database",
            "I use Oracle Database",
        ),

        # ------------------------------------------------------------------
        # AI / ML
        # ------------------------------------------------------------------

        make_memory(
            "m_llm",
            "user",
            "ai_model",
            "I use large language models",
        ),

        make_memory(
            "m_nli",
            "user",
            "ai_model",
            "I use natural language inference",
        ),

        make_memory(
            "m_embeddings",
            "user",
            "ai_method",
            "I use sentence embeddings",
        ),

        make_memory(
            "m_rag",
            "user",
            "ai_method",
            "I use retrieval augmented generation",
        ),

        make_memory(
            "m_pytorch",
            "user",
            "ml_framework",
            "I use PyTorch",
        ),

        make_memory(
            "m_tensorflow",
            "user",
            "ml_framework",
            "I use TensorFlow",
        ),

        make_memory(
            "m_opencv",
            "user",
            "computer_vision",
            "I use OpenCV",
        ),

        make_memory(
            "m_insightface",
            "user",
            "face_recognition",
            "I use InsightFace",
        ),

        # ------------------------------------------------------------------
        # ACMA PROJECT
        # ------------------------------------------------------------------

        make_memory(
            "m_acma_architecture",
            "project_acma",
            "architecture",
            "ACMA uses memory conflict resolution",
            context="ACMA",
        ),

        make_memory(
            "m_acma_nli",
            "project_acma",
            "reasoning",
            "ACMA uses NLI for risky memory conflicts",
            context="ACMA",
        ),

        make_memory(
            "m_acma_rag",
            "project_acma",
            "retrieval",
            "ACMA uses hybrid RAG retrieval",
            context="ACMA",
        ),

        make_memory(
            "m_acma_paper",
            "project_acma",
            "goal",
            "The ACMA project targets a conference paper",
            context="ACMA",
        ),

        make_memory(
            "m_acma_backend",
            "project_acma",
            "backend",
            "ACMA uses a Python backend",
            context="ACMA",
        ),

        make_memory(
            "m_acma_testing",
            "project_acma",
            "testing",
            "ACMA needs extensive pipeline testing",
            context="ACMA",
        ),

        # ------------------------------------------------------------------
        # ATTENDANCE / ERP
        # ------------------------------------------------------------------

        make_memory(
            "m_attendance_face",
            "project_attendance",
            "system",
            "The attendance project uses face recognition",
            context="Attendance",
        ),

        make_memory(
            "m_attendance_scrfd",
            "project_attendance",
            "detector",
            "The attendance project uses SCRFD",
            context="Attendance",
        ),

        make_memory(
            "m_attendance_embedding",
            "project_attendance",
            "embedding",
            "The attendance project uses face embeddings",
            context="Attendance",
        ),

        make_memory(
            "m_erp_database",
            "project_erp",
            "database",
            "The attendance ERP uses MySQL",
            context="ERP",
        ),

        # ------------------------------------------------------------------
        # EDUCATION
        # ------------------------------------------------------------------

        make_memory(
            "m_degree",
            "user",
            "education",
            "I am a CSE student",
        ),

        make_memory(
            "m_gate",
            "user",
            "exam",
            "I am preparing for GATE",
        ),

        make_memory(
            "m_dsa",
            "user",
            "learning",
            "I am learning data structures and algorithms",
        ),

        make_memory(
            "m_java_dsa",
            "user",
            "learning",
            "I practice DSA using Java",
            context="DSA",
        ),

        make_memory(
            "m_python_learning",
            "user",
            "learning",
            "I am learning Python",
            context="Learning",
        ),

        make_memory(
            "m_conference",
            "user",
            "academic_goal",
            "I want to prepare a conference paper",
        ),

        # ------------------------------------------------------------------
        # LOCATION
        # ------------------------------------------------------------------

        make_memory(
            "m_home_city",
            "user",
            "location",
            "I live in Hyderabad",
        ),

        make_memory(
            "m_college_city",
            "user",
            "college_location",
            "My college is in Hyderabad",
        ),

        make_memory(
            "m_office_city",
            "user",
            "office_location",
            "My office is in Bengaluru",
        ),

        make_memory(
            "m_timezone",
            "user",
            "timezone",
            "I use Indian Standard Time",
        ),

        # ------------------------------------------------------------------
        # FOOD
        # ------------------------------------------------------------------

        make_memory(
            "m_vegetarian",
            "user",
            "diet",
            "I am vegetarian",
        ),

        make_memory(
            "m_vegan",
            "user",
            "diet",
            "I am vegan",
        ),

        make_memory(
            "m_coffee",
            "user",
            "drink",
            "I like coffee",
        ),

        make_memory(
            "m_tea",
            "user",
            "drink",
            "I like tea",
        ),

        make_memory(
            "m_spicy",
            "user",
            "food_preference",
            "I prefer spicy food",
        ),

        make_memory(
            "m_breakfast",
            "user",
            "meal",
            "I usually eat breakfast early",
        ),

        # ------------------------------------------------------------------
        # DEVICES / OS
        # ------------------------------------------------------------------

        make_memory(
            "m_windows",
            "user",
            "operating_system",
            "I use Windows",
        ),

        make_memory(
            "m_linux",
            "user",
            "operating_system",
            "I use Linux",
        ),

        make_memory(
            "m_android",
            "user",
            "mobile_os",
            "I use Android",
        ),

        make_memory(
            "m_chrome",
            "user",
            "browser",
            "I use Chrome",
        ),

        make_memory(
            "m_vscode",
            "user",
            "editor",
            "I use VS Code",
        ),

        # ------------------------------------------------------------------
        # PREFERENCES
        # ------------------------------------------------------------------

        make_memory(
            "m_dark_mode",
            "user",
            "theme",
            "I prefer dark mode",
        ),

        make_memory(
            "m_notifications",
            "user",
            "notifications",
            "I keep notifications enabled",
        ),

        make_memory(
            "m_email",
            "user",
            "communication",
            "I prefer email",
        ),

        make_memory(
            "m_sms",
            "user",
            "communication",
            "I use SMS",
        ),

        make_memory(
            "m_weekend",
            "user",
            "schedule",
            "I prefer studying on weekends",
        ),

        # ------------------------------------------------------------------
        # FINANCE
        # ------------------------------------------------------------------

        make_memory(
            "m_currency",
            "user",
            "currency",
            "I use Indian Rupees",
        ),

        make_memory(
            "m_savings",
            "user",
            "finance",
            "I am saving money for education",
        ),

        make_memory(
            "m_budget",
            "user",
            "budget",
            "I prefer to stay within a monthly budget",
        ),

        make_memory(
            "m_bank",
            "user",
            "banking",
            "I use online banking",
        ),

        # ------------------------------------------------------------------
        # TRAVEL
        # ------------------------------------------------------------------

        make_memory(
            "m_travel",
            "user",
            "travel",
            "I travel mostly within India",
        ),

        make_memory(
            "m_flight",
            "user",
            "transport",
            "I usually travel by flight",
        ),

        make_memory(
            "m_train",
            "user",
            "transport",
            "I sometimes travel by train",
        ),

        make_memory(
            "m_hotel",
            "user",
            "accommodation",
            "I prefer hotels when travelling",
        ),

        # ------------------------------------------------------------------
        # HEALTH / ROUTINE
        # ------------------------------------------------------------------

        make_memory(
            "m_sleep",
            "user",
            "routine",
            "I usually sleep around midnight",
        ),

        make_memory(
            "m_exercise",
            "user",
            "fitness",
            "I exercise regularly",
        ),

        make_memory(
            "m_walk",
            "user",
            "fitness",
            "I go for evening walks",
        ),

        make_memory(
            "m_water",
            "user",
            "routine",
            "I try to drink enough water",
        ),

        # ------------------------------------------------------------------
        # HOBBIES
        # ------------------------------------------------------------------

        make_memory(
            "m_music",
            "user",
            "hobby",
            "I listen to music",
        ),

        make_memory(
            "m_movies",
            "user",
            "hobby",
            "I watch movies",
        ),

        make_memory(
            "m_games",
            "user",
            "hobby",
            "I play video games",
        ),

        make_memory(
            "m_reading",
            "user",
            "hobby",
            "I enjoy reading",
        ),

        # ------------------------------------------------------------------
        # DEADLINES / SCHEDULE
        # ------------------------------------------------------------------

        make_memory(
            "m_deadline",
            "user",
            "deadline",
            "My project deadline is next month",
        ),

        make_memory(
            "m_exam",
            "user",
            "deadline",
            "My exam is next month",
        ),

        make_memory(
            "m_meeting",
            "user",
            "schedule",
            "I have a project meeting tomorrow",
        ),

        make_memory(
            "m_submission",
            "user",
            "deadline",
            "My assignment is due this week",
        ),
    ]

    return memories


# ============================================================================
# TEMPORARY REAL MEMORY STORE
# ============================================================================

@pytest.fixture(scope="module")
def memory_store(
    tmp_path_factory,
    test_memories,
):
    directory = tmp_path_factory.mktemp(
        "acma_large_smoke"
    )

    memory_file = directory / "memories.json"

    with open(
        memory_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            [
                memory_to_dict(memory)
                for memory in test_memories
            ],
            file,
            indent=2,
        )

    return MemoryStore(
        file_path=str(memory_file)
    )


# ============================================================================
# REAL PIPELINE
# ============================================================================

@pytest.fixture(scope="module")
def pipeline():
    return ACPipeline()


# ============================================================================
# TARGETED HIGH-VALUE CASES
# ============================================================================
#
# These are the cases where we explicitly know the expected architecture
# behavior.
#
# Model outputs themselves are NEVER supplied manually.
# ============================================================================

TARGETED_CASES = [

    (
        "exact_duplicate_java",
        make_memory(
            "new_target_001",
            "user",
            "programming_language",
            "I use Java",
        ),
        "duplicate",
        "Ignore",
    ),

    (
        "exact_duplicate_python",
        make_memory(
            "new_target_002",
            "user",
            "programming_language",
            "I use Python",
        ),
        "duplicate",
        "Ignore",
    ),

    (
        "java_to_python",
        make_memory(
            "new_target_003",
            "user",
            "programming_language",
            "I now prefer Python instead of Java",
        ),
        "possible_update",
        "Resolve",
    ),

    (
        "dark_to_light",
        make_memory(
            "new_target_004",
            "user",
            "theme",
            "I switched to light mode",
        ),
        "possible_update",
        "Resolve",
    ),

    (
        "sqlite_to_postgresql",
        make_memory(
            "new_target_005",
            "user",
            "database",
            "I switched from SQLite to PostgreSQL",
        ),
        "possible_update",
        "Resolve",
    ),

    (
        "mysql_to_postgresql",
        make_memory(
            "new_target_006",
            "user",
            "database",
            "I now prefer PostgreSQL instead of MySQL",
        ),
        "possible_update",
        "Resolve",
    ),

    (
        "fastapi_to_flask",
        make_memory(
            "new_target_007",
            "user",
            "backend",
            "I switched from FastAPI to Flask",
        ),
        "possible_update",
        "Resolve",
    ),

    (
        "location_update",
        make_memory(
            "new_target_008",
            "user",
            "location",
            "I now live in Bengaluru",
        ),
        "possible_update",
        "Resolve",
    ),

    (
        "vegetarian_chicken_event",
        make_memory(
            "new_target_009",
            "user",
            "diet_event",
            "I ate chicken at a wedding yesterday",
            scope="specific_event",
            context="wedding",
            time="yesterday",
        ),
        "constraint_violation",
        "Ask",
    ),

    (
        "vegan_milk_event",
        make_memory(
            "new_target_010",
            "user",
            "diet_event",
            "I drank milk at a cafe yesterday",
            scope="specific_event",
            context="cafe",
            time="yesterday",
        ),
        "constraint_violation",
        "Ask",
    ),

    (
        "explicit_vegetarian_update",
        make_memory(
            "new_target_011",
            "user",
            "diet",
            "I am no longer vegetarian",
        ),
        "possible_update",
        "Resolve",
    ),

    (
        "different_attribute",
        make_memory(
            "new_target_012",
            "user",
            "favorite_color",
            "Blue",
        ),
        "different_attribute",
        "Preserve",
    ),

    (
        "different_subject",
        make_memory(
            "new_target_013",
            "project_x",
            "language",
            "The project uses Java",
        ),
        "different_subject",
        "Ignore",
    ),

    (
        "python_dsa_context",
        make_memory(
            "new_target_014",
            "user",
            "learning",
            "I practice DSA using Java",
            context="DSA",
        ),
        "duplicate",
        "Ignore",
    ),

    (
        "windows_to_linux",
        make_memory(
            "new_target_015",
            "user",
            "operating_system",
            "I switched from Windows to Linux",
        ),
        "possible_update",
        "Resolve",
    ),

    (
        "email_to_sms",
        make_memory(
            "new_target_016",
            "user",
            "communication",
            "I now prefer SMS instead of email",
        ),
        "possible_update",
        "Resolve",
    ),
]


@pytest.mark.parametrize(
    "name,new_memory,expected_relationship,expected_action",
    TARGETED_CASES,
)
def test_targeted_real_pipeline_cases(
    pipeline,
    memory_store,
    name,
    new_memory,
    expected_relationship,
    expected_action,
):
    retrieved, result = run_rag_pipeline(
        pipeline,
        memory_store,
        new_memory,
    )

    print_result(
        name,
        retrieved,
        result,
    )

    chosen = result["chosen"]

    assert chosen is not None

    relationship = (
        chosen["relationship"]["relationship"]
    )

    assert relationship == expected_relationship

    assert result["action"] == expected_action

    # SAFE -> NLI must not be called.
    if chosen["safety"].safe:
        assert chosen["nli"] is None

    # UNSAFE -> real NLI must be called.
    else:
        assert chosen["nli"] is not None


# ============================================================================
# LARGE DIVERSE INPUT POOL
# ============================================================================
#
# These cases intentionally cover different domains and different temporal
# / contextual patterns.
#
# No model outputs are specified.
# ============================================================================

BROAD_CASES = [

    # ------------------------------------------------------------------
    # Programming
    # ------------------------------------------------------------------

    (
        "Java Python",
        "user",
        "programming_language",
        "I now use Python instead of Java",
        "general",
        "general",
    ),

    (
        "Python Java",
        "user",
        "programming_language",
        "I switched from Python to Java",
        "general",
        "general",
    ),

    (
        "Java C++",
        "user",
        "programming_language",
        "I now prefer C++ instead of Java",
        "general",
        "general",
    ),

    (
        "Python Rust",
        "user",
        "programming_language",
        "I switched from Python to Rust",
        "general",
        "general",
    ),

    (
        "JavaScript TypeScript",
        "user",
        "programming_language",
        "I now prefer TypeScript instead of JavaScript",
        "general",
        "general",
    ),

    (
        "Go Rust",
        "user",
        "programming_language",
        "I switched from Go to Rust",
        "general",
        "general",
    ),

    (
        "Python continued",
        "user",
        "programming_language",
        "I still use Python",
        "general",
        "general",
    ),

    (
        "Java event",
        "user",
        "programming_language",
        "I used Java during a workshop yesterday",
        "specific_event",
        "workshop",
    ),

    (
        "Python event",
        "user",
        "programming_language",
        "I used Python during a workshop yesterday",
        "specific_event",
        "workshop",
    ),

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    (
        "FastAPI Flask",
        "user",
        "backend",
        "I switched from FastAPI to Flask",
        "general",
        "general",
    ),

    (
        "Flask Django",
        "user",
        "backend",
        "I now use Django instead of Flask",
        "general",
        "general",
    ),

    (
        "Django FastAPI",
        "user",
        "backend",
        "I moved from Django to FastAPI",
        "general",
        "general",
    ),

    (
        "Node FastAPI",
        "user",
        "backend",
        "I now prefer FastAPI instead of Node.js",
        "general",
        "general",
    ),

    (
        "Spring event",
        "user",
        "backend",
        "I tested Spring Boot during a college project",
        "specific_event",
        "college",
    ),

    # ------------------------------------------------------------------
    # Databases
    # ------------------------------------------------------------------

    (
        "SQLite PostgreSQL",
        "user",
        "database",
        "I switched from SQLite to PostgreSQL",
        "general",
        "general",
    ),

    (
        "MySQL PostgreSQL",
        "user",
        "database",
        "I now prefer PostgreSQL instead of MySQL",
        "general",
        "general",
    ),

    (
        "MongoDB MySQL",
        "user",
        "database",
        "I switched from MongoDB to MySQL",
        "general",
        "general",
    ),

    (
        "Oracle PostgreSQL",
        "user",
        "database",
        "I moved from Oracle Database to PostgreSQL",
        "general",
        "general",
    ),

    (
        "MongoDB event",
        "user",
        "database",
        "I used MongoDB for a college demonstration",
        "specific_event",
        "college_demo",
    ),

    # ------------------------------------------------------------------
    # AI / ML
    # ------------------------------------------------------------------

    (
        "NLI update",
        "user",
        "ai_model",
        "I now use an NLI model for conflict detection",
        "general",
        "general",
    ),

    (
        "LLM update",
        "user",
        "ai_model",
        "I now use a local LLM for fallback reasoning",
        "general",
        "general",
    ),

    (
        "Embedding update",
        "user",
        "ai_method",
        "I switched to sentence embeddings",
        "general",
        "general",
    ),

    (
        "RAG update",
        "user",
        "ai_method",
        "I now use hybrid RAG retrieval",
        "general",
        "general",
    ),

    (
        "PyTorch event",
        "user",
        "ml_framework",
        "I used PyTorch during an experiment",
        "specific_event",
        "experiment",
    ),

    (
        "TensorFlow event",
        "user",
        "ml_framework",
        "I tested TensorFlow during a project",
        "specific_event",
        "project",
    ),

    (
        "OpenCV event",
        "user",
        "computer_vision",
        "I used OpenCV for image processing yesterday",
        "specific_event",
        "experiment",
    ),

    # ------------------------------------------------------------------
    # ACMA
    # ------------------------------------------------------------------

    (
        "ACMA safety gate",
        "project_acma",
        "architecture",
        "ACMA now uses a safety gate before NLI",
        "general",
        "ACMA",
    ),

    (
        "ACMA retrieval",
        "project_acma",
        "retrieval",
        "ACMA now uses hybrid retrieval",
        "general",
        "ACMA",
    ),

    (
        "ACMA paper",
        "project_acma",
        "goal",
        "I discussed the ACMA paper with my mentor yesterday",
        "specific_event",
        "mentor_meeting",
    ),

    (
        "ACMA testing",
        "project_acma",
        "testing",
        "I tested ACMA conflict resolution yesterday",
        "specific_event",
        "testing",
    ),

    (
        "ACMA NLI",
        "project_acma",
        "reasoning",
        "I tested NLI on ACMA yesterday",
        "specific_event",
        "testing",
    ),

    # ------------------------------------------------------------------
    # Education
    # ------------------------------------------------------------------

    (
        "Education update",
        "user",
        "education",
        "I am now studying computer science",
        "general",
        "general",
    ),

    (
        "GATE update",
        "user",
        "exam",
        "I am now preparing for GATE",
        "general",
        "general",
    ),

    (
        "DSA update",
        "user",
        "learning",
        "I now practice DSA every day",
        "general",
        "general",
    ),

    (
        "Java DSA event",
        "user",
        "learning",
        "I solved a Java DSA problem yesterday",
        "specific_event",
        "DSA",
    ),

    (
        "Python learning event",
        "user",
        "learning",
        "I practiced Python today",
        "specific_event",
        "Learning",
    ),

    (
        "Conference event",
        "user",
        "academic_goal",
        "I discussed my conference paper yesterday",
        "specific_event",
        "academic",
    ),

    # ------------------------------------------------------------------
    # Location
    # ------------------------------------------------------------------

    (
        "Hyderabad Bengaluru",
        "user",
        "location",
        "I now live in Bengaluru instead of Hyderabad",
        "general",
        "general",
    ),

    (
        "Hyderabad visit",
        "user",
        "location",
        "I visited Hyderabad yesterday",
        "specific_event",
        "travel",
    ),

    (
        "Bengaluru office",
        "user",
        "office_location",
        "I now work from Bengaluru",
        "general",
        "work",
    ),

    (
        "College visit",
        "user",
        "college_location",
        "I visited my college today",
        "specific_event",
        "college",
    ),

    # ------------------------------------------------------------------
    # Food / dietary
    # ------------------------------------------------------------------

    (
        "Vegetarian chicken",
        "user",
        "diet_event",
        "I ate chicken at a wedding yesterday",
        "specific_event",
        "wedding",
    ),

    (
        "Vegetarian fish",
        "user",
        "diet_event",
        "I ate fish during a family dinner",
        "specific_event",
        "family_dinner",
    ),

    (
        "Vegetarian meat",
        "user",
        "diet_event",
        "I ate meat at a party last night",
        "specific_event",
        "party",
    ),

    (
        "Vegan milk",
        "user",
        "diet_event",
        "I drank milk at a cafe yesterday",
        "specific_event",
        "cafe",
    ),

    (
        "Vegetarian update",
        "user",
        "diet",
        "I am no longer vegetarian",
        "general",
        "general",
    ),

    (
        "Vegan update",
        "user",
        "diet",
        "I am no longer vegan",
        "general",
        "general",
    ),

    (
        "Coffee tea",
        "user",
        "drink",
        "I now prefer tea instead of coffee",
        "general",
        "general",
    ),

    (
        "Tea event",
        "user",
        "drink",
        "I drank tea at a meeting",
        "specific_event",
        "meeting",
    ),

    (
        "Spicy event",
        "user",
        "food_preference",
        "I ate spicy food at dinner",
        "specific_event",
        "dinner",
    ),

    # ------------------------------------------------------------------
    # Operating systems / tools
    # ------------------------------------------------------------------

    (
        "Windows Linux",
        "user",
        "operating_system",
        "I switched from Windows to Linux",
        "general",
        "general",
    ),

    (
        "Linux Windows",
        "user",
        "operating_system",
        "I moved from Linux to Windows",
        "general",
        "general",
    ),

    (
        "Android event",
        "user",
        "mobile_os",
        "I used an Android phone during a trip",
        "specific_event",
        "travel",
    ),

    (
        "Chrome update",
        "user",
        "browser",
        "I now use Chrome as my main browser",
        "general",
        "general",
    ),

    (
        "VS Code update",
        "user",
        "editor",
        "I now use VS Code for development",
        "general",
        "general",
    ),

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    (
        "Dark light",
        "user",
        "theme",
        "I switched from dark mode to light mode",
        "general",
        "general",
    ),

    (
        "Light dark",
        "user",
        "theme",
        "I now prefer dark mode instead of light mode",
        "general",
        "general",
    ),

    (
        "Email SMS",
        "user",
        "communication",
        "I now prefer SMS instead of email",
        "general",
        "general",
    ),

    (
        "Notifications event",
        "user",
        "notifications",
        "I disabled notifications during an exam",
        "specific_event",
        "exam",
    ),

    (
        "Weekend event",
        "user",
        "schedule",
        "I studied on Sunday yesterday",
        "specific_event",
        "study",
    ),

    # ------------------------------------------------------------------
    # Finance
    # ------------------------------------------------------------------

    (
        "Currency",
        "user",
        "currency",
        "I use Indian Rupees",
        "general",
        "general",
    ),

    (
        "Savings event",
        "user",
        "finance",
        "I saved money for tuition this month",
        "specific_event",
        "finance",
    ),

    (
        "Budget update",
        "user",
        "budget",
        "I now follow a stricter monthly budget",
        "general",
        "general",
    ),

    (
        "Banking event",
        "user",
        "banking",
        "I used online banking today",
        "specific_event",
        "banking",
    ),

    # ------------------------------------------------------------------
    # Travel
    # ------------------------------------------------------------------

    (
        "Flight event",
        "user",
        "transport",
        "I travelled by flight yesterday",
        "specific_event",
        "travel",
    ),

    (
        "Train event",
        "user",
        "transport",
        "I travelled by train yesterday",
        "specific_event",
        "travel",
    ),

    (
        "Travel update",
        "user",
        "travel",
        "I now travel mostly outside India",
        "general",
        "general",
    ),

    (
        "Hotel event",
        "user",
        "accommodation",
        "I stayed in a hotel last weekend",
        "specific_event",
        "travel",
    ),

    # ------------------------------------------------------------------
    # Routine / fitness
    # ------------------------------------------------------------------

    (
        "Sleep update",
        "user",
        "routine",
        "I now sleep around 11 PM",
        "general",
        "general",
    ),

    (
        "Sleep event",
        "user",
        "routine",
        "I slept late yesterday",
        "specific_event",
        "yesterday",
    ),

    (
        "Exercise update",
        "user",
        "fitness",
        "I now exercise every morning",
        "general",
        "general",
    ),

    (
        "Walk event",
        "user",
        "fitness",
        "I went for an evening walk yesterday",
        "specific_event",
        "yesterday",
    ),

    (
        "Water event",
        "user",
        "routine",
        "I drank extra water after exercising",
        "specific_event",
        "exercise",
    ),

    # ------------------------------------------------------------------
    # Hobbies
    # ------------------------------------------------------------------

    (
        "Music update",
        "user",
        "hobby",
        "I now listen to music while studying",
        "general",
        "study",
    ),

    (
        "Movie event",
        "user",
        "hobby",
        "I watched a movie yesterday",
        "specific_event",
        "leisure",
    ),

    (
        "Game event",
        "user",
        "hobby",
        "I played a video game last night",
        "specific_event",
        "leisure",
    ),

    (
        "Reading event",
        "user",
        "hobby",
        "I read a technical book yesterday",
        "specific_event",
        "study",
    ),

    # ------------------------------------------------------------------
    # Deadlines / schedule
    # ------------------------------------------------------------------

    (
        "Project deadline",
        "user",
        "deadline",
        "My project deadline is next week",
        "general",
        "general",
    ),

    (
        "Exam deadline",
        "user",
        "deadline",
        "My exam is next week",
        "general",
        "general",
    ),

    (
        "Meeting event",
        "user",
        "schedule",
        "I had a project meeting yesterday",
        "specific_event",
        "meeting",
    ),

    (
        "Assignment event",
        "user",
        "deadline",
        "I submitted my assignment yesterday",
        "specific_event",
        "college",
    ),
]


# ============================================================================
# CREATE LARGE MIXED TEST SET
# ============================================================================

def build_stress_cases():

    cases = []

    # ------------------------------------------------------------------
    # 1. Original cases
    # ------------------------------------------------------------------

    for index, case in enumerate(
        BROAD_CASES,
        start=1,
    ):

        (
            label,
            subject,
            attribute,
            value,
            scope,
            context,
        ) = case

        cases.append(
            (
                f"base-{index:03d}-{label}",
                make_memory(
                    f"stress_base_{index:03d}",
                    subject,
                    attribute,
                    value,
                    scope=scope,
                    context=context,
                ),
            )
        )

    # ------------------------------------------------------------------
    # 2. Wording variations
    #
    # These are still realistic user inputs, not model outputs.
    # ------------------------------------------------------------------

    prefixes = [
        "Currently, ",
        "Today, ",
        "I want to note that ",
        "At the moment, ",
    ]

    for index, case in enumerate(
        BROAD_CASES,
        start=1,
    ):

        (
            label,
            subject,
            attribute,
            value,
            scope,
            context,
        ) = case

        for variant_index, prefix in enumerate(
            prefixes,
            start=1,
        ):

            # Keep first-person sentences natural.
            if value.startswith("I "):

                varied_value = (
                    prefix
                    + value[0].lower()
                    + value[1:]
                )

            else:

                varied_value = (
                    prefix
                    + value
                )

            cases.append(
                (
                    f"variant-{index:03d}-{variant_index}-{label}",
                    make_memory(
                        (
                            f"stress_variant_"
                            f"{index:03d}_"
                            f"{variant_index}"
                        ),
                        subject,
                        attribute,
                        varied_value,
                        scope=scope,
                        context=context,
                    ),
                )
            )

    # ------------------------------------------------------------------
    # 3. Exact duplicates
    # ------------------------------------------------------------------

    duplicate_memories = [

        make_memory(
            "duplicate_java",
            "user",
            "programming_language",
            "I use Java",
        ),

        make_memory(
            "duplicate_python",
            "user",
            "programming_language",
            "I use Python",
        ),

        make_memory(
            "duplicate_dark",
            "user",
            "theme",
            "I prefer dark mode",
        ),

        make_memory(
            "duplicate_sqlite",
            "user",
            "database",
            "I use SQLite",
        ),

        make_memory(
            "duplicate_vegetarian",
            "user",
            "diet",
            "I am vegetarian",
        ),

        make_memory(
            "duplicate_windows",
            "user",
            "operating_system",
            "I use Windows",
        ),

        make_memory(
            "duplicate_coffee",
            "user",
            "drink",
            "I like coffee",
        ),

        make_memory(
            "duplicate_music",
            "user",
            "hobby",
            "I listen to music",
        ),

        make_memory(
            "duplicate_hyderabad",
            "user",
            "location",
            "I live in Hyderabad",
        ),

        make_memory(
            "duplicate_rag",
            "user",
            "ai_method",
            "I use retrieval augmented generation",
        ),
    ]

    for memory in duplicate_memories:

        cases.append(
            (
                f"duplicate-{memory.memory_id}",
                memory,
            )
        )

    # ------------------------------------------------------------------
    # 4. Shuffle deterministically.
    #
    # This prevents all programming cases from running together, followed
    # by all food cases, etc.
    # ------------------------------------------------------------------

    rng = random.Random(20260925)

    rng.shuffle(cases)

    return cases


STRESS_CASES = build_stress_cases()


# ============================================================================
# LARGE REAL PIPELINE TEST
# ============================================================================

@pytest.mark.parametrize(
    "case_name,new_memory",
    STRESS_CASES,
)
def test_large_diverse_real_pipeline(
    pipeline,
    memory_store,
    case_name,
    new_memory,
):

    retrieved, result = run_rag_pipeline(
        pipeline,
        memory_store,
        new_memory,
    )

    # --------------------------------------------------------------
    # RAG must return candidates.
    # --------------------------------------------------------------

    assert retrieved, (
        f"{case_name}: RAG returned no candidates"
    )

    # --------------------------------------------------------------
    # Pipeline must select a candidate.
    # --------------------------------------------------------------

    assert result["chosen"] is not None, (
        f"{case_name}: pipeline selected no candidate"
    )

    chosen = result["chosen"]

    # --------------------------------------------------------------
    # Basic candidate integrity.
    # --------------------------------------------------------------

    assert chosen["old_memory"] is not None

    assert chosen["new_memory"] is new_memory

    # --------------------------------------------------------------
    # Retrieval score must be generated by RAG.
    # --------------------------------------------------------------

    retrieval_score = chosen[
        "retrieval_score"
    ]

    assert isinstance(
        retrieval_score,
        float,
    )

    assert 0.0 <= retrieval_score <= 1.0

    # --------------------------------------------------------------
    # Relationship layer.
    # --------------------------------------------------------------

    relationship = chosen[
        "relationship"
    ]

    assert isinstance(
        relationship,
        dict,
    )

    assert relationship.get(
        "relationship"
    ) is not None

    # --------------------------------------------------------------
    # Safety Gate.
    # --------------------------------------------------------------

    safety = chosen[
        "safety"
    ]

    assert isinstance(
        safety.safe,
        bool,
    )

    # --------------------------------------------------------------
    # SAFE:
    #
    # NLI must NOT run.
    # --------------------------------------------------------------

    if safety.safe:

        assert chosen["nli"] is None

    # --------------------------------------------------------------
    # UNSAFE:
    #
    # Real NLI must run.
    # --------------------------------------------------------------

    else:

        nli = chosen["nli"]

        assert nli is not None

        assert nli["label"] in {
            "entailment",
            "contradiction",
            "neutral",
        }

        assert 0.0 <= (
            nli["confidence"]
        ) <= 1.0

        assert 0.0 <= (
            nli["margin"]
        ) <= 1.0

        assert isinstance(
            nli["scores"],
            dict,
        )

    # --------------------------------------------------------------
    # Evidence.
    #
    # The score must come from the real evidence layer.
    # --------------------------------------------------------------

    evidence = chosen[
        "evidence"
    ]

    assert isinstance(
        evidence,
        dict,
    )

    assert 0.0 <= (
        evidence["evidence_score"]
    ) <= 1.0

    # --------------------------------------------------------------
    # Decision.
    # --------------------------------------------------------------

    assert result["action"] in {
        "Ignore",
        "Preserve",
        "Resolve",
        "Ask",
        "Store",
    }

    assert 0.0 <= (
        result["confidence"]
    ) <= 1.0

    # --------------------------------------------------------------
    # Compact output.
    #
    # We intentionally don't print the complete NLI/evidence object for
    # hundreds of cases because that would make the test output enormous.
    # --------------------------------------------------------------

    print(
        f"\n[{case_name}] "
        f"relationship="
        f"{relationship['relationship']} "
        f"action={result['action']} "
        f"retrieval={retrieval_score:.4f} "
        f"nli="
        f"{'YES' if chosen['nli'] is not None else 'NO'}"
    )


# ============================================================================
# PIPELINE STATISTICS
# ============================================================================

def test_large_stress_statistics(
    pipeline,
):

    statistics = pipeline.statistics()

    print("\n")
    print("=" * 110)
    print("LARGE STRESS TEST STATISTICS")
    print("=" * 110)

    print(
        "Total candidates =",
        statistics["total_candidates"],
    )

    print(
        "Total NLI calls  =",
        statistics["total_nli_calls"],
    )

    print(
        "NLI rate         =",
        statistics["nli_rate"],
    )

    assert (
        statistics["total_candidates"]
        > 200
    )

    assert (
        statistics["total_nli_calls"]
        <= statistics["total_candidates"]
    )

    assert (
        0.0
        <= statistics["nli_rate"]
        <= 1.0
    )


# ============================================================================
# MEMORY CORPUS DIVERSITY TEST
# ============================================================================

def test_memory_corpus_is_diverse(
    test_memories,
):

    attributes = {
        memory.attribute
        for memory in test_memories
    }

    contexts = {
        memory.context
        for memory in test_memories
    }

    subjects = {
        memory.subject
        for memory in test_memories
    }

    # We want a genuinely broad corpus.
    assert len(test_memories) >= 70

    assert len(attributes) >= 20

    assert len(contexts) >= 8

    assert len(subjects) >= 4