import json

from memory.schema import Memory
from conflict.safety_gate import evaluate_rule_safety



def convert(data):

    return Memory(
        memory_id="test",
        subject=data["subject"],
        attribute=data["attribute"],
        value=data["value"],
        scope=data["scope"],
        context=data["context"],
        source="user",
        confidence=1.0,
        importance=0.5
    )



def test_benchmark():

    with open(
        "data/safety_gate_benchmark.json"
    ) as f:

        cases=json.load(f)


    correct=0


    for case in cases:


        old=convert(case["old"])
        new=convert(case["new"])



        result=evaluate_rule_safety(
            old,
            new,
            similarity=0.9
        )


        prediction = (
            "SAFE"
            if result.safe
            else
            "UNSAFE"
        )


        if prediction == case["expected"]:
            correct += 1
        else:
            print("\n--- MISCLASSIFIED CASE ---")
            print("Expected:", case["expected"])
            print("Predicted:", prediction)
            print("Old memory:", case["old"])
            print("New memory:", case["new"])
            print("Safety result:", result)

    accuracy = correct / len(cases) * 100



    accuracy=correct/len(cases)*100


    print("\n===================")
    print("Safety Gate Benchmark")
    print("===================")

    print(
        "Total:",
        len(cases)
    )

    print(
        "Correct:",
        correct
    )


    print(
        "Accuracy:",
        accuracy,
        "%"
    )


    assert accuracy > 80