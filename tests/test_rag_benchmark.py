import json
from pathlib import Path

from memory.schema import Memory, MemoryQuery
from memory.store import MemoryStore


# ============================================================
# CONFIGURATION
# ============================================================

TOTAL_CASES = 600
TOP_K = 5
DIAGNOSTIC_K = 10
SIMILARITY_THRESHOLD = 0.30

CORPUS_PATH = Path("data/rag_benchmark_memories.json")


# ============================================================
# HELPERS
# ============================================================

def memory_key(memory):
    """
    Canonical identity of a memory.

    We intentionally ignore memory_id so that duplicate copies
    of the same logical memory do not artificially occupy
    multiple retrieval positions.
    """
    return (
        memory.subject.strip().lower(),
        memory.attribute.strip().lower(),
        memory.value.strip().lower(),
        memory.scope.strip().lower(),
        memory.context.strip().lower(),
        (memory.time or "").strip().lower(),
    )


def build_base_memories():
    """
    Creates a realistic set of memories that represent the
    long-term memory corpus of an AI assistant.
    """

    memories = [
        Memory(
            memory_id="base_001",
            subject="user",
            attribute="language",
            value="Python is used for ACMA development",
            scope="general",
            context="ACMA project",
            confidence=0.9,
            importance=0.8,
        ),
        Memory(
            memory_id="base_002",
            subject="user",
            attribute="language",
            value="Java is used for DSA coursework",
            scope="general",
            context="DSA",
            confidence=0.9,
            importance=0.7,
        ),
        Memory(
            memory_id="base_003",
            subject="user",
            attribute="backend",
            value="ACMA backend uses FastAPI",
            scope="general",
            context="ACMA project",
            confidence=0.9,
            importance=0.8,
        ),
        Memory(
            memory_id="base_004",
            subject="user",
            attribute="food",
            value="User is vegetarian",
            scope="general",
            context="diet",
            confidence=0.9,
            importance=0.8,
        ),
        Memory(
            memory_id="base_005",
            subject="user",
            attribute="notifications",
            value="User prefers email notifications",
            scope="general",
            context="communication",
            confidence=0.8,
            importance=0.6,
        ),
        Memory(
            memory_id="base_006",
            subject="application",
            attribute="theme",
            value="Application normally uses dark mode",
            scope="general",
            context="application",
            confidence=0.8,
            importance=0.5,
        ),
        Memory(
            memory_id="base_007",
            subject="user",
            attribute="operating_system",
            value="User normally develops on Windows",
            scope="general",
            context="development",
            confidence=0.9,
            importance=0.7,
        ),
        Memory(
            memory_id="base_008",
            subject="user",
            attribute="database",
            value="User uses PostgreSQL for backend projects",
            scope="general",
            context="backend development",
            confidence=0.8,
            importance=0.7,
        ),
        Memory(
            memory_id="base_009",
            subject="user",
            attribute="database",
            value="User uses SQLite for small experiments",
            scope="general",
            context="experiments",
            confidence=0.8,
            importance=0.5,
        ),
        Memory(
            memory_id="base_010",
            subject="user",
            attribute="editor",
            value="User prefers VS Code for development",
            scope="general",
            context="development",
            confidence=0.8,
            importance=0.6,
        ),
        Memory(
            memory_id="base_011",
            subject="user",
            attribute="framework",
            value="User uses React for frontend applications",
            scope="general",
            context="web development",
            confidence=0.8,
            importance=0.7,
        ),
        Memory(
            memory_id="base_012",
            subject="user",
            attribute="framework",
            value="User uses FastAPI for Python APIs",
            scope="general",
            context="web development",
            confidence=0.8,
            importance=0.7,
        ),
        Memory(
            memory_id="base_013",
            subject="user",
            attribute="project",
            value="ACMA is an adaptive memory architecture for autonomous AI agents",
            scope="general",
            context="ACMA project",
            confidence=0.9,
            importance=0.9,
        ),
        Memory(
            memory_id="base_014",
            subject="user",
            attribute="memory",
            value="ACMA uses retrieval before conflict analysis",
            scope="general",
            context="ACMA architecture",
            confidence=0.9,
            importance=0.9,
        ),
        Memory(
            memory_id="base_015",
            subject="user",
            attribute="memory",
            value="ACMA uses NLI for semantic relationship analysis",
            scope="general",
            context="ACMA architecture",
            confidence=0.9,
            importance=0.9,
        ),
        Memory(
            memory_id="base_016",
            subject="user",
            attribute="memory",
            value="ACMA uses rule safety before expensive reasoning",
            scope="general",
            context="ACMA architecture",
            confidence=0.9,
            importance=0.8,
        ),
        Memory(
            memory_id="base_017",
            subject="user",
            attribute="goal",
            value="User wants ACMA to work as a real-time application",
            scope="general",
            context="ACMA project",
            confidence=0.9,
            importance=0.9,
        ),
        Memory(
            memory_id="base_018",
            subject="user",
            attribute="goal",
            value="User wants ACMA to be suitable for a conference paper",
            scope="general",
            context="ACMA project",
            confidence=0.9,
            importance=0.9,
        ),
        Memory(
            memory_id="base_019",
            subject="user",
            attribute="learning",
            value="User is learning Java and DSA",
            scope="general",
            context="DSA",
            confidence=0.9,
            importance=0.8,
        ),
        Memory(
            memory_id="base_020",
            subject="user",
            attribute="learning",
            value="User prefers step-by-step explanations when learning programming",
            scope="general",
            context="learning",
            confidence=0.9,
            importance=0.7,
        ),
    ]

    return memories


def generate_benchmark_cases(base_memories):
    """
    Generate 600 retrieval cases.

    Each case contains:
        - new memory/query
        - expected related old memory
        - category

    The same logical memories may be reused across cases,
    but the corpus itself is canonicalized so duplicates
    do not crowd retrieval results.
    """

    cases = []

    templates = [
        # ----------------------------------------------------
        # Python
        # ----------------------------------------------------
        (
            "Python",
            "I am currently working with Python for my ACMA development.",
            "base_001",
        ),
        (
            "Python",
            "The ACMA project is being developed using Python.",
            "base_001",
        ),
        (
            "Python",
            "Python is the main language I use for ACMA.",
            "base_001",
        ),
        (
            "Python",
            "For ACMA development, I use Python.",
            "base_001",
        ),
        (
            "Python",
            "My ACMA backend work is done in Python.",
            "base_001",
        ),

        # ----------------------------------------------------
        # Java / DSA
        # ----------------------------------------------------
        (
            "Java DSA",
            "I use Java while solving my DSA coursework.",
            "base_002",
        ),
        (
            "Java DSA",
            "Java is my language for DSA practice.",
            "base_002",
        ),
        (
            "Java DSA",
            "My DSA assignments are implemented in Java.",
            "base_002",
        ),
        (
            "Java DSA",
            "I normally use Java for DSA problems.",
            "base_002",
        ),
        (
            "Java DSA",
            "Java is used for my data structures coursework.",
            "base_002",
        ),

        # ----------------------------------------------------
        # FastAPI backend
        # ----------------------------------------------------
        (
            "FastAPI",
            "The ACMA backend is implemented with FastAPI.",
            "base_003",
        ),
        (
            "FastAPI",
            "I use FastAPI as the backend framework for ACMA.",
            "base_003",
        ),
        (
            "FastAPI",
            "ACMA's backend uses the FastAPI framework.",
            "base_003",
        ),
        (
            "FastAPI",
            "FastAPI is used on the server side of ACMA.",
            "base_003",
        ),
        (
            "FastAPI",
            "My ACMA API backend is based on FastAPI.",
            "base_003",
        ),

        # ----------------------------------------------------
        # Vegetarian
        # ----------------------------------------------------
        (
            "vegetarian",
            "I am vegetarian.",
            "base_004",
        ),
        (
            "vegetarian",
            "I follow a vegetarian diet.",
            "base_004",
        ),
        (
            "vegetarian",
            "My general diet is vegetarian.",
            "base_004",
        ),
        (
            "vegetarian",
            "I prefer vegetarian food.",
            "base_004",
        ),
        (
            "vegetarian",
            "I normally eat vegetarian meals.",
            "base_004",
        ),

        # ----------------------------------------------------
        # Email
        # ----------------------------------------------------
        (
            "email",
            "I prefer receiving notifications through email.",
            "base_005",
        ),
        (
            "email",
            "Email is my preferred notification method.",
            "base_005",
        ),
        (
            "email",
            "Send my notifications by email.",
            "base_005",
        ),
        (
            "email",
            "I normally want notifications through email.",
            "base_005",
        ),
        (
            "email",
            "Email is the communication method I prefer for alerts.",
            "base_005",
        ),

        # ----------------------------------------------------
        # Dark mode
        # ----------------------------------------------------
        (
            "dark mode",
            "The application normally uses dark mode.",
            "base_006",
        ),
        (
            "dark mode",
            "I usually keep the application in dark theme.",
            "base_006",
        ),
        (
            "dark mode",
            "The app's normal theme is dark.",
            "base_006",
        ),
        (
            "dark mode",
            "Dark mode is the default application theme.",
            "base_006",
        ),
        (
            "dark mode",
            "The application generally runs with a dark interface.",
            "base_006",
        ),

        # ----------------------------------------------------
        # Windows
        # ----------------------------------------------------
        (
            "Windows",
            "I normally develop on Windows.",
            "base_007",
        ),
        (
            "Windows",
            "Windows is my usual development operating system.",
            "base_007",
        ),
        (
            "Windows",
            "I use Windows for development work.",
            "base_007",
        ),
        (
            "Windows",
            "My normal development environment is Windows.",
            "base_007",
        ),
        (
            "Windows",
            "Windows is the operating system I generally work on.",
            "base_007",
        ),

        # ----------------------------------------------------
        # PostgreSQL
        # ----------------------------------------------------
        (
            "PostgreSQL",
            "I use PostgreSQL for backend projects.",
            "base_008",
        ),
        (
            "PostgreSQL",
            "PostgreSQL is my database for backend development.",
            "base_008",
        ),
        (
            "PostgreSQL",
            "My backend applications commonly use PostgreSQL.",
            "base_008",
        ),
        (
            "PostgreSQL",
            "I normally use PostgreSQL when building backend systems.",
            "base_008",
        ),
        (
            "PostgreSQL",
            "PostgreSQL is used in my backend projects.",
            "base_008",
        ),

        # ----------------------------------------------------
        # SQLite
        # ----------------------------------------------------
        (
            "SQLite",
            "I use SQLite for small experiments.",
            "base_009",
        ),
        (
            "SQLite",
            "Small experimental projects use SQLite.",
            "base_009",
        ),
        (
            "SQLite",
            "SQLite is useful for my small experiments.",
            "base_009",
        ),
        (
            "SQLite",
            "I normally choose SQLite for small experiments.",
            "base_009",
        ),
        (
            "SQLite",
            "My small test projects often use SQLite.",
            "base_009",
        ),

        # ----------------------------------------------------
        # VS Code
        # ----------------------------------------------------
        (
            "VS Code",
            "I prefer VS Code for development.",
            "base_010",
        ),
        (
            "VS Code",
            "Visual Studio Code is my preferred development editor.",
            "base_010",
        ),
        (
            "VS Code",
            "I normally code using VS Code.",
            "base_010",
        ),
        (
            "VS Code",
            "My preferred editor is VS Code.",
            "base_010",
        ),
        (
            "VS Code",
            "I use Visual Studio Code for programming.",
            "base_010",
        ),

        # ----------------------------------------------------
        # React
        # ----------------------------------------------------
        (
            "React",
            "I use React for frontend applications.",
            "base_011",
        ),
        (
            "React",
            "React is my frontend framework.",
            "base_011",
        ),
        (
            "React",
            "I build web frontends using React.",
            "base_011",
        ),
        (
            "React",
            "My frontend applications use React.",
            "base_011",
        ),
        (
            "React",
            "React is used for my web interfaces.",
            "base_011",
        ),

        # ----------------------------------------------------
        # FastAPI Python APIs
        # ----------------------------------------------------
        (
            "FastAPI APIs",
            "I use FastAPI when building Python APIs.",
            "base_012",
        ),
        (
            "FastAPI APIs",
            "FastAPI is my framework for Python API development.",
            "base_012",
        ),
        (
            "FastAPI APIs",
            "I build Python APIs with FastAPI.",
            "base_012",
        ),
        (
            "FastAPI APIs",
            "FastAPI is used for my Python web APIs.",
            "base_012",
        ),
        (
            "FastAPI APIs",
            "My Python APIs are built using FastAPI.",
            "base_012",
        ),

        # ----------------------------------------------------
        # ACMA project
        # ----------------------------------------------------
        (
            "ACMA",
            "ACMA is an adaptive memory architecture for autonomous AI agents.",
            "base_013",
        ),
        (
            "ACMA",
            "My project ACMA focuses on adaptive memory for AI agents.",
            "base_013",
        ),
        (
            "ACMA",
            "ACMA is designed as a memory architecture for autonomous agents.",
            "base_013",
        ),
        (
            "ACMA",
            "The ACMA project provides adaptive memory for AI agents.",
            "base_013",
        ),
        (
            "ACMA",
            "ACMA is my adaptive cognitive memory architecture project.",
            "base_013",
        ),

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------
        (
            "retrieval",
            "ACMA retrieves relevant memories before conflict analysis.",
            "base_014",
        ),
        (
            "retrieval",
            "Memory retrieval happens before conflict analysis in ACMA.",
            "base_014",
        ),
        (
            "retrieval",
            "ACMA uses retrieval as an earlier stage before conflict resolution.",
            "base_014",
        ),
        (
            "retrieval",
            "Relevant memories are retrieved before analyzing conflicts.",
            "base_014",
        ),
        (
            "retrieval",
            "The ACMA pipeline performs retrieval before conflict analysis.",
            "base_014",
        ),

        # ----------------------------------------------------
        # NLI
        # ----------------------------------------------------
        (
            "NLI",
            "ACMA uses NLI for semantic relationship analysis.",
            "base_015",
        ),
        (
            "NLI",
            "Natural language inference is used to analyze memory relationships.",
            "base_015",
        ),
        (
            "NLI",
            "The ACMA pipeline uses an NLI model for semantic comparison.",
            "base_015",
        ),
        (
            "NLI",
            "NLI determines semantic relationships between memories in ACMA.",
            "base_015",
        ),
        (
            "NLI",
            "ACMA applies natural language inference to memory statements.",
            "base_015",
        ),

        # ----------------------------------------------------
        # Rule safety
        # ----------------------------------------------------
        (
            "rule safety",
            "ACMA uses a rule safety layer before expensive reasoning.",
            "base_016",
        ),
        (
            "rule safety",
            "The safety gate avoids unnecessary expensive reasoning.",
            "base_016",
        ),
        (
            "rule safety",
            "ACMA checks rule safety before using more expensive reasoning.",
            "base_016",
        ),
        (
            "rule safety",
            "The ACMA architecture uses a safety gate before costly inference.",
            "base_016",
        ),
        (
            "rule safety",
            "Rules are checked before expensive reasoning in ACMA.",
            "base_016",
        ),

        # ----------------------------------------------------
        # Real-time
        # ----------------------------------------------------
        (
            "real-time",
            "I want ACMA to work as a real-time application.",
            "base_017",
        ),
        (
            "real-time",
            "The ACMA project should operate in real time.",
            "base_017",
        ),
        (
            "real-time",
            "I want the memory architecture to support real-time applications.",
            "base_017",
        ),
        (
            "real-time",
            "ACMA needs to be suitable for real-time use.",
            "base_017",
        ),
        (
            "real-time",
            "My goal is to use ACMA in a real-time application.",
            "base_017",
        ),

        # ----------------------------------------------------
        # Conference paper
        # ----------------------------------------------------
        (
            "conference paper",
            "I want ACMA to be suitable for a conference paper.",
            "base_018",
        ),
        (
            "conference paper",
            "The ACMA project is intended to support a conference paper.",
            "base_018",
        ),
        (
            "conference paper",
            "I want to publish the ACMA work as a conference paper.",
            "base_018",
        ),
        (
            "conference paper",
            "ACMA should have a research contribution suitable for a conference paper.",
            "base_018",
        ),
        (
            "conference paper",
            "The project is being developed with conference-paper evaluation in mind.",
            "base_018",
        ),

        # ----------------------------------------------------
        # Java learning
        # ----------------------------------------------------
        (
            "Java learning",
            "I am learning Java and DSA.",
            "base_019",
        ),
        (
            "Java learning",
            "Java and data structures are part of my learning.",
            "base_019",
        ),
        (
            "Java learning",
            "I am currently studying Java for DSA.",
            "base_019",
        ),
        (
            "Java learning",
            "My programming studies include Java and DSA.",
            "base_019",
        ),
        (
            "Java learning",
            "I am practicing Java while learning data structures.",
            "base_019",
        ),

        # ----------------------------------------------------
        # Step-by-step
        # ----------------------------------------------------
        (
            "step by step",
            "I prefer step-by-step explanations when learning programming.",
            "base_020",
        ),
        (
            "step by step",
            "When learning code, I prefer explanations broken into steps.",
            "base_020",
        ),
        (
            "step by step",
            "I like programming explanations to be step by step.",
            "base_020",
        ),
        (
            "step by step",
            "For programming learning, detailed sequential explanations help me.",
            "base_020",
        ),
        (
            "step by step",
            "I prefer learning programming through step-by-step guidance.",
            "base_020",
        ),
    ]

    # 20 groups × 5 variants = 100 templates.
    # Repeat them to reach exactly 600 cases.
    case_id = 1

    while len(cases) < TOTAL_CASES:
        for category, text, expected_id in templates:
            if len(cases) >= TOTAL_CASES:
                break

            cases.append(
                {
                    "case_id": case_id,
                    "category": category,
                    "new_text": text,
                    "expected_memory_id": expected_id,
                }
            )

            case_id += 1

    return cases


def write_corpus(memories):
    """
    Writes the canonical benchmark corpus to disk.
    """

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CORPUS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            [memory.to_dict() for memory in memories],
            file,
            indent=2,
        )


# ============================================================
# RAG BENCHMARK
# ============================================================

def run_rag_benchmark():

    print("\n" + "=" * 80)
    print("ACMA RAG-ONLY BENCHMARK")
    print("=" * 80)

    print("\nImportant:")
    print("This benchmark tests ONLY memory retrieval.")
    print("NLI, Safety Gate, Evidence Score and Decision Policy")
    print("are intentionally NOT executed.")
    print("=" * 80)

    # --------------------------------------------------------
    # Build corpus
    # --------------------------------------------------------

    base_memories = build_base_memories()

    # Canonicalize the corpus.
    # This prevents duplicate logical memories from occupying
    # multiple retrieval positions.
    unique_memories = {}

    for memory in base_memories:
        key = memory_key(memory)

        if key not in unique_memories:
            unique_memories[key] = memory

    memories = list(unique_memories.values())

    write_corpus(memories)

    print(f"\nCorpus memories: {len(memories)}")
    print(f"Benchmark cases: {TOTAL_CASES}")
    print(f"Top-K operational: {TOP_K}")
    print(f"Top-K diagnostic: {DIAGNOSTIC_K}")

    # --------------------------------------------------------
    # Build benchmark
    # --------------------------------------------------------

    cases = generate_benchmark_cases(memories)

    # Map IDs to memory objects.
    memory_by_id = {
        memory.memory_id: memory
        for memory in memories
    }

    # --------------------------------------------------------
    # Memory Store
    # --------------------------------------------------------

    store = MemoryStore(
        file_path=str(CORPUS_PATH)
    )

    print("\nBuilding embedding cache...")
    store._build_embedding_cache()

    print("Embedding cache ready.")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0

    found_cases = 0
    missing_cases = 0

    ranks = []
    expected_scores = []

    failures = []
    representative_cases = []

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    for index, case in enumerate(cases, start=1):

        expected_id = case["expected_memory_id"]

        expected_memory = memory_by_id.get(expected_id)

        if expected_memory is None:
            print(
                f"\nERROR: expected memory {expected_id} "
                f"does not exist."
            )
            continue

        query = MemoryQuery(
            subject=expected_memory.subject,
            attribute=expected_memory.attribute,
            value=case["new_text"],
            scope=expected_memory.scope,
            context=expected_memory.context,
            time=expected_memory.time,
        )

        # Retrieve diagnostic top 10.
        results = store.retrieve_related_memories(
            query=query,
            top_k=DIAGNOSTIC_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

        result_keys = [
            memory_key(memory)
            for memory, score in results
        ]

        expected_key = memory_key(expected_memory)

        rank = None
        expected_score = None

        for position, (memory, score) in enumerate(results, start=1):

            if memory_key(memory) == expected_key:
                rank = position
                expected_score = score
                break

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        if rank is not None:

            found_cases += 1
            ranks.append(rank)

            if expected_score is not None:
                expected_scores.append(expected_score)

            if rank <= 1:
                recall_1 += 1

            if rank <= 5:
                recall_5 += 1

            if rank <= 10:
                recall_10 += 1

        else:

            missing_cases += 1

            failures.append(
                {
                    "case": case,
                    "results": results,
                }
            )

        # ----------------------------------------------------
        # Representative examples
        # ----------------------------------------------------

        if len(representative_cases) < 15:

            representative_cases.append(
                {
                    "case": case,
                    "results": results,
                    "rank": rank,
                    "expected_score": expected_score,
                }
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if index % 25 == 0 or index == TOTAL_CASES:

            current_recall_5 = (
                recall_5 / index * 100
            )

            print(
                f"Progress: {index:>3}/{TOTAL_CASES} | "
                f"Recall@5: {current_recall_5:>6.2f}% | "
                f"Found@10: {found_cases:>3}"
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL RAG RESULTS")
    print("=" * 80)

    print(f"\nTotal cases:       {TOTAL_CASES}")
    print(f"Found in Top-10:   {found_cases}")
    print(f"Missing Top-10:    {missing_cases}")

    print("\nRETRIEVAL RECALL")
    print("-" * 80)

    print(
        f"Recall@1:  "
        f"{recall_1 / TOTAL_CASES * 100:.2f}%"
    )

    print(
        f"Recall@5:  "
        f"{recall_5 / TOTAL_CASES * 100:.2f}%"
    )

    print(
        f"Recall@10: "
        f"{recall_10 / TOTAL_CASES * 100:.2f}%"
    )

    if ranks:

        average_rank = sum(ranks) / len(ranks)

        print(
            f"\nAverage rank when found: "
            f"{average_rank:.2f}"
        )

    if expected_scores:

        average_score = (
            sum(expected_scores)
            / len(expected_scores)
        )

        print(
            f"Average expected-memory score: "
            f"{average_score:.4f}"
        )

    # ========================================================
    # REPRESENTATIVE RETRIEVALS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("REPRESENTATIVE RAG RETRIEVALS")
    print("=" * 80)

    for item in representative_cases:

        case = item["case"]
        results = item["results"]
        rank = item["rank"]

        print("\n" + "-" * 80)

        print(
            f"Case {case['case_id']} | "
            f"Category: {case['category']}"
        )

        print(
            f"Query: {case['new_text']}"
        )

        print(
            f"Expected memory ID: "
            f"{case['expected_memory_id']}"
        )

        if rank is None:
            print("Expected memory rank: NOT FOUND")
        else:
            print(
                f"Expected memory rank: {rank}"
            )

        print("\nTop retrieved memories:")

        for position, (memory, score) in enumerate(
            results[:5],
            start=1
        ):

            marker = ""

            if (
                memory.memory_id
                == case["expected_memory_id"]
            ):
                marker = "  <-- EXPECTED"

            print(
                f"{position}. "
                f"[score={score:.4f}] "
                f"{memory.memory_id} | "
                f"{memory.attribute} | "
                f"{memory.value}"
                f"{marker}"
            )

    # ========================================================
    # FAILURE ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("RAG FAILURE ANALYSIS")
    print("=" * 80)

    if not failures:

        print("\nNo Top-10 retrieval failures.")

    else:

        print(
            f"\nShowing first "
            f"{min(20, len(failures))} failures:"
        )

        for item in failures[:20]:

            case = item["case"]
            results = item["results"]

            print("\n" + "-" * 80)

            print(
                f"Case {case['case_id']} | "
                f"{case['category']}"
            )

            print(
                f"Query: {case['new_text']}"
            )

            print(
                f"Expected: {case['expected_memory_id']}"
            )

            print("Top retrieved:")

            for position, (memory, score) in enumerate(
                results[:5],
                start=1
            ):

                print(
                    f"  {position}. "
                    f"{memory.memory_id} | "
                    f"{score:.4f} | "
                    f"{memory.value}"
                )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("\n")
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    recall5 = recall_5 / TOTAL_CASES

    if recall5 >= 0.90:

        print(
            "\nRAG retrieval is finding the expected memory "
            "in most cases."
        )

    elif recall5 >= 0.70:

        print(
            "\nRAG retrieval is working reasonably well, "
            "but there are still meaningful retrieval misses."
        )

    elif recall5 >= 0.50:

        print(
            "\nRAG retrieval is only partially reliable."
        )

    else:

        print(
            "\nRAG retrieval is currently weak."
        )

    print(
        "\nThis benchmark does NOT measure final ACMA "
        "decision accuracy."
    )

    print(
        "It only answers one question:"
    )

    print(
        "Can ACMA retrieve the memory that should be "
        "investigated?"
    )

    print("=" * 80)


# ============================================================
# PYTEST ENTRY POINT
# ============================================================

def test_rag_benchmark():

    run_rag_benchmark()