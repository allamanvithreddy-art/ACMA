from conflict.nli_checker import compare_statements


test_cases = [
    (
        "I am vegetarian.",
        "I do not eat meat."
    ),
    (
        "I am vegetarian.",
        "I ate chicken at a wedding."
    ),
    (
        "I use Java.",
        "I use Python."
    ),
    (
        "I use Java.",
        "I use Java."
    )
]


for old_text, new_text in test_cases:
    print("Old:", old_text)
    print("New:", new_text)

    result = compare_statements(old_text, new_text)

    print("Result:", result)
    print("-" * 60)