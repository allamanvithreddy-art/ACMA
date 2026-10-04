from memory.schema import Memory
from conflict.metadata import compare_metadata
from conflict.update_detector import analyze_update
from conflict.safety_gate import evaluate_rule_safety
from conflict.decision_policy import decide_action


def make_memory(
    memory_id: str,
    *,
    subject: str,
    attribute: str,
    value: str,
    metadata: dict | None = None,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        text=value,
        subject=subject,
        attribute=attribute,
        value=value,
        scope="",
        context="",
        metadata=metadata or {},
    )


def test_metadata_comparison_does_not_invent_semantics():
    old = make_memory(
        "old",
        subject="entity-1",
        attribute="property-1",
        value="value-A",
    )
    new = make_memory(
        "new",
        subject="entity-1",
        attribute="property-1",
        value="value-B",
    )

    result = compare_metadata(old, new)

    assert result["same_subject"]
    assert result["same_attribute"]
    assert not result["same_value"]


def test_unstructured_different_values_are_not_automatically_resolved():
    old = make_memory(
        "old",
        subject="entity-1",
        attribute="property-1",
        value="value-A",
    )
    new = make_memory(
        "new",
        subject="entity-1",
        attribute="property-1",
        value="value-B",
    )

    result = analyze_update(old, new)

    assert result["targeted_to_old_memory"] is False


def test_update_must_explicitly_target_old_memory():
    old = make_memory(
        "old",
        subject="entity-1",
        attribute="property-1",
        value="value-A",
    )
    new = make_memory(
        "new",
        subject="entity-1",
        attribute="property-1",
        value="value-B",
        metadata={
            "is_update": True,
            "update_of": "old",
            "is_replacement": True,
        },
    )

    result = analyze_update(old, new)

    assert result["targeted_to_old_memory"] is True
    assert result["explicit_replacement"] is True


def test_unresolved_candidate_is_routed_to_nli():
    old = make_memory(
        "old",
        subject="entity-1",
        attribute="property-1",
        value="value-A",
    )
    new = make_memory(
        "new",
        subject="entity-1",
        attribute="property-1",
        value="value-B",
    )

    result = evaluate_rule_safety(old, new)

    assert result.next_stage == "nli"
    assert result.action_hint is None


def test_contradiction_without_replacement_requires_ask():
    safety = {
        "next_stage": "nli",
        "relationship": "unresolved",
        "signals": {},
    }
    evidence = {
        "relationship": "unresolved",
        "structural": {},
        "update": {
            "targeted_to_old_memory": False,
            "explicit_replacement": None,
        },
        "nli": {
            "label": "contradiction",
            "confidence": 0.91,
            "margin": 0.80,
            "model": "test-model",
        },
    }

    result = decide_action(safety=safety, evidence=evidence)

    assert result["action"] == "Ask"


def test_targeted_explicit_replacement_with_contradiction_resolves():
    safety = {
        "next_stage": "nli",
        "relationship": "unresolved",
        "signals": {},
    }
    evidence = {
        "relationship": "unresolved",
        "structural": {},
        "update": {
            "targeted_to_old_memory": True,
            "explicit_replacement": True,
        },
        "nli": {
            "label": "contradiction",
            "confidence": 0.91,
            "margin": 0.80,
            "model": "test-model",
        },
    }

    result = decide_action(safety=safety, evidence=evidence)

    assert result["action"] == "Resolve"