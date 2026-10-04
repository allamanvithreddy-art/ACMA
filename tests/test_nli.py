from memory.schema import Memory
from conflict.nli_engine import NLIEngine



def create_memory(value):

    return Memory(
        memory_id="1",
        subject="user",
        attribute="test",
        value=value,
        scope="general",
        context="general",
        source="user",
        confidence=1.0,
        importance=0.5
    )



def test_nli_cases():

    engine = NLIEngine()


    cases=[

        (
            "I use Python for ACMA",
            "I use Python for ACMA project"
        ),


        (
            "I use Java for ACMA",
            "I switched ACMA development to Python"
        ),


        (
            "I am vegetarian",
            "I ate chicken at a wedding"
        )

    ]


    for old,new in cases:

        old_memory=create_memory(old)
        new_memory=create_memory(new)


        result=engine.compare(
            old_memory,
            new_memory
        )


        print("\n----------------")
        print("OLD:",old)
        print("NEW:",new)

        print(result)