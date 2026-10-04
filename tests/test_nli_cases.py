from memory.schema import Memory
from conflict.nli_engine import NLIEngine


def make_memory(
    memory_id,
    value,
    subject="user",
    attribute="test",
    scope="general",
    context="general",
):
    return Memory(
        memory_id=memory_id,
        subject=subject,
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        confidence=0.9,
        importance=0.7,
    )


def print_result(case_name, old_value, new_value, result):

    print()
    print("=" * 80)
    print(case_name)
    print("=" * 80)

    print("OLD:")
    print(old_value)

    print("\nNEW:")
    print(new_value)

    print("\nNLI RESULT:")
    print(f"Label:       {result['label']}")
    print(f"Confidence:  {result['confidence']:.4f}")
    print(f"Margin:      {result['margin']:.4f}")
    print(f"Scores:      {result['scores']}")


def test_nli_cases():

    engine = NLIEngine()

    cases = [

        # ====================================================
        # CLEAR CONTRADICTION
        # ====================================================

        (
            "CONTRADICTION - Java to Python",
            "I use Java for DSA.",
            "I use Python for DSA.",
            "contradiction",
        ),

        (
            "CONTRADICTION - vegetarian",
            "I am vegetarian.",
            "I prefer non-vegetarian food.",
            "contradiction",
        ),

        (
            "CONTRADICTION - Windows to Linux",
            "I use Windows.",
            "I use Linux.",
            "contradiction",
        ),

        (
            "CONTRADICTION - dark to light",
            "The application uses dark mode.",
            "The application uses light mode.",
            "contradiction",
        ),

        (
            "CONTRADICTION - email to SMS",
            "I prefer email notifications.",
            "I prefer SMS notifications.",
            "contradiction",
        ),

        # ====================================================
        # CLEAR ENTAILMENT
        # ====================================================

        (
            "ENTAILMENT - Python paraphrase",
            "Python is used for ACMA development.",
            "The ACMA project is developed using Python.",
            "entailment",
        ),

        (
            "ENTAILMENT - vegetarian paraphrase",
            "I am vegetarian.",
            "I follow a vegetarian diet.",
            "entailment",
        ),

        (
            "ENTAILMENT - FastAPI paraphrase",
            "The ACMA backend uses FastAPI.",
            "ACMA's backend is implemented with FastAPI.",
            "entailment",
        ),

        # ====================================================
        # NEUTRAL / CONTEXTUAL
        # ====================================================

        (
            "NEUTRAL - vegetarian event",
            "I am vegetarian.",
            "I ate chicken at a wedding.",
            "neutral",
        ),

        (
            "NEUTRAL - Python vs FastAPI",
            "Python is used for ACMA development.",
            "ACMA backend uses FastAPI.",
            "neutral",
        ),

        (
            "NEUTRAL - different attribute",
            "I use Java for DSA.",
            "I prefer vegetarian food.",
            "neutral",
        ),

        (
            "NEUTRAL - different project",
            "Python is used for ACMA development.",
            "Python is used for DSA coursework.",
            "neutral",
        ),

        # ====================================================
        # EXPLICIT UPDATES
        # ====================================================

        (
            "UPDATE - Java to C++",
            "The user uses Java for DSA.",
            "I switched from Java to C++ for DSA.",
            "contradiction",
        ),

        (
            "UPDATE - SQLite to PostgreSQL",
            "The application uses SQLite.",
            "We migrated the application database to PostgreSQL.",
            "contradiction",
        ),

        (
            "UPDATE - FastAPI to Django",
            "The application backend uses FastAPI.",
            "The backend now uses Django.",
            "contradiction",
        ),
    ]

    passed = 0

    print()
    print("#" * 80)
    print("ACMA PURE NLI EVALUATION")
    print("#" * 80)

    print(
        "\nThis test evaluates NLI itself."
        "\nDecision Policy and LLM fallback are NOT involved."
    )

    for (
        case_name,
        old_value,
        new_value,
        expected_label,
    ) in cases:

        old_memory = make_memory(
            "old",
            old_value,
        )

        new_memory = make_memory(
            "new",
            new_value,
        )

        # IMPORTANT:
        # Pass Memory objects.
        #
        # NLIEngine will compare .value only.
        result = engine.compare(
            old_memory,
            new_memory,
        )

        print_result(
            case_name,
            old_value,
            new_value,
            result,
        )

        actual_label = result["label"]

        if actual_label == expected_label:
            passed += 1

        print(
            f"\nExpected: {expected_label}"
        )

        print(
            f"Actual:   {actual_label}"
        )

        print(
            "STATUS:   "
            + (
                "PASS"
                if actual_label == expected_label
                else "FAIL"
            )
        )

    total = len(cases)

    print()
    print("#" * 80)
    print("NLI SUMMARY")
    print("#" * 80)

    print(f"Total cases: {total}")
    print(f"Correct:     {passed}")
    print(f"Incorrect:   {total - passed}")
    print(
        f"Accuracy:    {passed / total * 100:.2f}%"
    )

    assert passed > 0