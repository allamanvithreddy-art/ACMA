from conflict.update_detector import detect_update_signals


test_statements = [
    "I now prefer Python.",
    "I no longer use Java.",
    "I moved to Bengaluru.",
    "I ate chicken at a wedding.",
    "I previously lived in Hyderabad.",
    "I like machine learning."
]


for statement in test_statements:
    result = detect_update_signals(statement)

    print("Statement:", statement)
    print("Result:", result)
    print("-" * 60)