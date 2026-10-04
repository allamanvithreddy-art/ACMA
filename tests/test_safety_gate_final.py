from memory.schema import Memory
from conflict.safety_gate import evaluate_rule_safety



def create_memory(
    id,
    attribute,
    value,
    context="general"
):

    return Memory(

        memory_id=id,

        subject="user",

        attribute=attribute,

        value=value,

        scope="general",

        context=context,

        source="user",

        confidence=1.0,

        importance=0.5
    )



def test_cases():


    cases=[


    (
        "Duplicate",

        create_memory(
            "1",
            "language",
            "Python"
        ),

        create_memory(
            "2",
            "language",
            "Python"
        )

    ),


    (
        "Update",

        create_memory(
            "1",
            "backend",
            "FastAPI"
        ),

        create_memory(
            "2",
            "backend",
            "Switched to Flask"
        )

    ),


    (
        "Different fact",

        create_memory(
            "1",
            "backend",
            "FastAPI"
        ),

        create_memory(
            "2",
            "database",
            "PostgreSQL"
        )

    ),


    (
        "Event",

        create_memory(
            "1",
            "food",
            "vegetarian"
        ),

        create_memory(
            "2",
            "food",
            "ate chicken at wedding",
            "wedding"
        )

    )

    ]


    for name,old,new in cases:

        result = evaluate_rule_safety(
            old,
            new
        )


        print("\n====================")
        print(name)
        print("====================")

        print(result)


