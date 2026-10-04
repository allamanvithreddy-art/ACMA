import json
from pathlib import Path

from memory.schema import Memory
from memory.store import MemoryStore

from conflict.safety_gate import evaluate_rule_safety
from conflict.nli_checker import compare_statements


DATA_FILE = Path(
    "data/safety_gate_memories.json"
)


def load_memories():

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return [
        Memory(**item)
        for item in data
    ]



def create_new_memory(text):

    return Memory(
        memory_id="new",
        subject="user",
        attribute="general",
        value=text,
        scope="general",
        context="general",
        source="user",
        confidence=1.0,
        importance=0.5
    )



def run_pipeline(new_text):


    print("\n")
    print("="*80)
    print("NEW MEMORY")
    print(new_text)


    # -------------------------------------------------
    # 1. Create incoming memory
    # -------------------------------------------------

    new_memory = create_new_memory(
        new_text
    )


    # -------------------------------------------------
    # 2. RAG retrieval
    # -------------------------------------------------

    store = MemoryStore()

    retrieved = store.retrieve_related_memories(
    new_memory,
    top_k=3
)


    print("\nRETRIEVED MEMORIES")


    for memory, similarity in retrieved:

        print("-"*50)

        print(
            "OLD:",
            memory.value
        )

        print(
            "Similarity:",
            round(similarity,4)
        )


        # -------------------------------------------------
        # 3. Safety Gate receives similarity from RAG
        # -------------------------------------------------

        safety = evaluate_rule_safety(
    memory,
    new_memory
)


        print("\nSAFETY GATE")

        print(
            "Relationship:",
            safety.relationship
        )

        print(
            "Safe:",
            safety.safe
        )

        print(
            "Next Stage:",
            safety.next_stage
        )


        # -------------------------------------------------
        # 4. Unsafe cases go to NLI
        # -------------------------------------------------

        if not safety.safe:


            nli = compare_statements(
                memory.value,
                new_memory.value
            )


            print("\nNLI RESULT")

            print(
                nli
            )



def test_full_pipeline():

    cases = [

        "I switched ACMA backend from FastAPI to Flask",

        "I no longer use Python for ACMA",

        "I ate chicken at a wedding",

        "The ACMA paper deadline moved to September 27",

        "The application uses dark mode normally but tomorrow use light mode",

        "ACMA uses FastAPI backend"

    ]


    for case in cases:

        run_pipeline(case)