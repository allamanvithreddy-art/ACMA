from memory.schema import Memory
from conflict.safety_gate import evaluate_rule_safety


def create_memory(
    memory_id,
    attribute,
    value,
    scope="general",
    context="general",
    time=None,
    source="user",
    confidence=1.0,
    importance=0.5,
):
    return Memory(
        memory_id=memory_id,
        subject="user",
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        time=time,
        source=source,
        confidence=confidence,
        importance=importance,
    )


def print_result(name, result):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("safe:", result.safe)
    print("next_stage:", result.next_stage)
    print("action_hint:", result.action_hint)
    print("relationship:", result.relationship)
    print("safety_confidence:", result.safety_confidence)
    print("reason:", result.reason)
    print("risk_flags:", result.risk_flags)
    print("signals:", result.signals)


def test_exact_duplicate_is_safe():
    old = create_memory(
        "m1",
        "programming_preference",
        "Python"
    )

    new = create_memory(
        "m2",
        "programming_preference",
        "Python"
    )

    result = evaluate_rule_safety(
        old,
        new,
        similarity=0.99
    )

    print_result("Exact duplicate", result)

    assert result.safe is True
    assert result.next_stage == "rule_action"
    assert result.action_hint == "Ignore"


def test_different_subject_is_safe():
    old = create_memory(
        "m1",
        "programming_preference",
        "Python"
    )

    new = create_memory(
        "m2",
        "programming_preference",
        "Python"
    )

    new.subject = "system"

    result = evaluate_rule_safety(
        old,
        new,
        similarity=0.60
    )

    print_result("Different subject", result)

    assert result.safe is True
    assert result.action_hint == "Unrelated"


def test_different_attribute_is_safe():
    old = create_memory(
        "m1",
        "food_preference",
        "vegetarian"
    )

    new = create_memory(
        "m2",
        "programming_preference",
        "Python"
    )

    result = evaluate_rule_safety(
        old,
        new,
        similarity=0.70
    )

    print_result("Different attribute", result)

    assert result.safe is True
    assert result.action_hint == "Preserve"


def test_possible_general_conflict_goes_to_nli():
    old = create_memory(
        "m1",
        "programming_preference",
        "Java"
    )

    new = create_memory(
        "m2",
        "programming_preference",
        "Python"
    )

    result = evaluate_rule_safety(
        old,
        new,
        similarity=0.90
    )

    print_result("Possible general conflict", result)

    assert result.safe is False
    assert result.next_stage == "nli"


def test_explicit_update_goes_to_nli():
    old = create_memory(
        "m1",
        "programming_preference",
        "Java"
    )

    new = create_memory(
        "m2",
        "programming_preference",
        "I now prefer Python"
    )

    result = evaluate_rule_safety(
        old,
        new,
        similarity=0.90
    )

    print_result("Explicit update", result)

    assert result.safe is False
    assert result.next_stage == "nli"


def test_specific_event_is_not_safely_resolved():
    old = create_memory(
        "m1",
        "food_preference",
        "vegetarian",
        scope="general"
    )

    new = create_memory(
        "m2",
        "food_event",
        "ate chicken",
        scope="specific_event",
        context="wedding"
    )

    result = evaluate_rule_safety(
        old,
        new,
        similarity=0.70
    )

    print_result("General preference vs specific event", result)

    assert result.safe is False
    assert result.next_stage == "nli"


def test_negation_goes_to_nli():
    old = create_memory(
        "m1",
        "programming_preference",
        "Python"
    )

    new = create_memory(
        "m2",
        "programming_preference",
        "I do not prefer Python"
    )

    result = evaluate_rule_safety(
        old,
        new,
        similarity=0.90
    )

    print_result("Negation", result)

    assert result.safe is False
    assert result.next_stage == "nli"


def test_temporal_change_goes_to_nli():
    old = create_memory(
        "m1",
        "programming_preference",
        "Java",
        time="2025"
    )

    new = create_memory(
        "m2",
        "programming_preference",
        "Python",
        time="2026"
    )

    result = evaluate_rule_safety(
        old,
        new,
        similarity=0.90
    )

    print_result("Temporal change", result)

    assert result.safe is False
    assert result.next_stage == "nli"