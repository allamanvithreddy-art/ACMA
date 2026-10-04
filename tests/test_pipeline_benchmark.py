import json

from memory.store import MemoryStore

from memory.schema import MemoryQuery

from conflict.safety_gate import evaluate_rule_safety

from conflict.nli_engine import NLIEngine



def create_query(text):

    return MemoryQuery(
        subject="user",
        attribute="unknown",
        value=text,
        scope="general",
        context="general"
    )



def test_pipeline_accuracy():

    with open(
        "data/full_pipeline_benchmark.json",
        "r",
        encoding="utf-8"
    ) as f:

        cases=json.load(f)



    store=MemoryStore()

    nli=NLIEngine()


    correct=0

    wrong=0



    print("\n==============================")
    print("FULL PIPELINE DEBUG BENCHMARK")
    print("==============================")



    for case in cases:


        query=create_query(
            case["new"]
        )


        retrieved=store.retrieve_related_memories(
            query,
            top_k=3
        )


        final_action="Preserve"



        for old, similarity in retrieved:



            safety=evaluate_rule_safety(
                old,
                query,
                similarity
            )



            if safety.safe:


                final_action=safety.action_hint



            else:


                nli_result=nli.compare(
                    old,
                    query
                )


                label=nli_result["label"]

                confidence=nli_result["confidence"]



                if (
                    label=="contradiction"
                    and
                    confidence >= 0.70
                ):

                    final_action="Resolve"



                elif (
                    label=="entailment"
                    and
                    confidence >= 0.70
                ):

                    final_action="Ignore duplicate"



                else:

                    final_action="Ask"



            break



        expected=case["expected"]



        if final_action == expected:

            correct+=1



        else:

            wrong+=1


            print("\n================================")
            print("WRONG CASE")
            print("================================")

            print(
                "NEW:",
                case["new"]
            )

            print(
                "EXPECTED:",
                expected
            )

            print(
                "PREDICTED:",
                final_action
            )


            print("\nRETRIEVED:")

            for old,sim in retrieved:

                print(
                    old.value,
                    "SIM:",
                    sim
                )



    total=len(cases)



    print("\n================")
    print("RESULT")
    print("================")


    print(
        "Total:",
        total
    )


    print(
        "Correct:",
        correct
    )


    print(
        "Wrong:",
        wrong
    )


    print(
        "Accuracy:",
        round(
            correct/total*100,
            2
        )
    )