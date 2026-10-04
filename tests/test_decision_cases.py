from conflict.decision_policy import decide_action



def test_decision_policy():


    cases=[


        {

            "name":"Duplicate memory",

            "relationship":"duplicate",

            "evidence_score":0.9,

            "nli_label":"entailment",

            "nli_confidence":0.95,

            "expected":"Ignore"

        },


        {


            "name":"Different event",

            "relationship":"different_attribute",

            "evidence_score":0.8,

            "nli_label":"neutral",

            "nli_confidence":0.90,

            "expected":"Preserve"

        },


        {


            "name":"Strong contradiction",

            "relationship":"possible_update",

            "evidence_score":0.90,

            "nli_label":"contradiction",

            "nli_confidence":0.95,

            "expected":"Resolve"

        },


        {


            "name":"Weak contradiction",

            "relationship":"possible_update",

            "evidence_score":0.35,

            "nli_label":"contradiction",

            "nli_confidence":0.40,

            "expected":"Preserve"

        },


        {


            "name":"Uncertain case",

            "relationship":"possible_update",

            "evidence_score":0.60,

            "nli_label":"neutral",

            "nli_confidence":0.60,

            "expected":"Ask"

        }



    ]


    print("\n==============================")
    print("DECISION POLICY TEST")
    print("==============================")


    passed=0


    for case in cases:


        result=decide_action(

            relationship=case["relationship"],

            evidence_score=case["evidence_score"],

            nli_label=case["nli_label"],

            nli_confidence=case["nli_confidence"]

        )


        print("\nCase:",case["name"])

        print(
            "Expected:",
            case["expected"]
        )

        print(
            "Predicted:",
            result
        )


        if result==case["expected"]:

            print("PASS")
            passed+=1

        else:

            print("FAIL")



    print("\n==============================")

    print(
        "Accuracy:",
        passed,
        "/",
        len(cases)
    )


    assert passed==len(cases)


