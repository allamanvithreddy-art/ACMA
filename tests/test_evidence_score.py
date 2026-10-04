import json

from memory.store import MemoryStore

from memory.schema import MemoryQuery

from conflict.safety_gate import evaluate_rule_safety

from conflict.nli_checker import compare_statements

from conflict.evidence_score import calculate_evidence_score

from conflict.decision_policy import decide_action



def create_query(text):

    return MemoryQuery(

        subject="unknown",

        attribute="unknown",

        value=text,

        scope="general",

        context=None

    )



def get_nli_confidence(result):

    scores = result.get(
        "scores",
        {}
    )

    if not scores:
        return 0.0

    return max(
        scores.values()
    )



def test_full_acma_pipeline():


    cases=[


        {
            "old":
            "I prefer Java",

            "new":
            "I switched ACMA development to Python",

            "expected":
            "Resolve"
        },


        {
            "old":
            "I am vegetarian",

            "new":
            "I ate chicken at a wedding",

            "expected":
            "Preserve"
        },


        {
            "old":
            "I use Python",

            "new":
            "I use Python",

            "expected":
            "Ignore"
        },


        {
            "old":
            "I live in Hyderabad",

            "new":
            "I now live in Bengaluru",

            "expected":
            "Resolve"
        }

    ]


    store=MemoryStore()


    correct=0



    for case in cases:


        old=create_query(
            case["old"]
        )


        new=create_query(
            case["new"]
        )


        retrieved=store.retrieve_related_memories(
            new,
            top_k=3
        )


        final_action=None



        for memory,similarity in retrieved:


            safety=evaluate_rule_safety(

                memory,

                new,

                similarity

            )


            if safety.safe:


                final_action=safety.action

                break



            nli=compare_statements(

                memory.value,

                new.value

            )


            evidence=calculate_evidence_score(

                nli_confidence=get_nli_confidence(nli),

                nli_margin=nli.get(
                    "margin",
                    0.0
                ),

                similarity=similarity,

                metadata_agreement=0.5,

                update_signal=0.5

            )



            final_action=decide_action(

                relationship=safety.relationship,

                evidence_score=evidence,

                nli_label=nli["label"],

                nli_confidence=get_nli_confidence(nli)

            )



        print("\n----------------")

        print("OLD:",case["old"])

        print("NEW:",case["new"])

        print("EXPECTED:",case["expected"])

        print("PREDICTED:",final_action)



        if final_action==case["expected"]:

            correct+=1



    accuracy=(correct/len(cases))*100


    print("\n================")

    print(
        "Accuracy:",
        accuracy
    )


    assert accuracy >= 50