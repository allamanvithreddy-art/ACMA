from memory.schema import Memory
from conflict.pipeline import ACPipeline
from conflict.safety_gate import evaluate_rule_safety
from conflict.decision_policy import decide_action


def mem(memory_id, value, scope, context="", metadata=None):
    return Memory(
        memory_id=memory_id,
        user_id="u1",
        text=value,
        subject="user",
        attribute="diet",
        value=value,
        scope=scope,
        context=context,
        metadata=metadata or {},
    )


def test_event_contradiction_requires_ask_without_update():
    old = mem("old", "I am vegetarian", "general", "general")
    new = mem("new", "I ate chicken at a wedding", "specific_event", "wedding")
    safety = evaluate_rule_safety(old, new, similarity=0.8)
    assert safety.next_stage == "nli"
    evidence = {
        "relationship": safety.relationship,
        "context": {
            "new_is_event": True,
        },
        "update": {
            "targeted_to_old_memory": False,
            "replacement_supported": False,
            "same_subject_and_attribute": True,
        },
        "nli": {
            "label": "contradiction",
            "confidence": 0.92,
            "margin": 0.60,
        },
    }
    decision = decide_action(safety=safety, evidence=evidence)
    assert decision.action == "Ask"


def test_explicit_update_resolves():
    old = mem("old", "I use Java", "general", "backend")
    new = Memory(
        memory_id="new",
        user_id="u1",
        text="I now use Python instead",
        subject="user",
        attribute="backend_language",
        value="I now use Python instead",
        scope="general",
        context="backend",
        metadata={
            "is_update": True,
            "is_replacement": True,
            "update_of": "old",
        },
    )
    safety = evaluate_rule_safety(old, new, similarity=0.85)
    assert safety.safe is True
    assert safety.action_hint == "Resolve"
    decision = decide_action(safety=safety, evidence={"relationship": "update", "update": safety.signals["update"], "nli": None})
    assert decision.action == "Resolve"


def test_exact_duplicate_is_ignored():
    old = mem("old", "I use Python", "general", "general")
    new = mem("new", "I use Python", "general", "general")
    safety = evaluate_rule_safety(old, new, similarity=1.0)
    assert safety.action_hint == "Ignore"
    decision = decide_action(safety=safety, evidence={"relationship": "duplicate", "update": {}, "nli": None})
    assert decision.action == "Ignore"


def test_independent_attribute_is_preserved():
    old = Memory(memory_id="old", user_id="u1", subject="user", attribute="language", value="I use Python", scope="general", context="general")
    new = Memory(memory_id="new", user_id="u1", subject="user", attribute="editor", value="I use VS Code", scope="general", context="general")
    safety = evaluate_rule_safety(old, new, similarity=0.4)
    assert safety.action_hint == "Preserve"
    decision = decide_action(safety=safety, evidence={"relationship": "independent", "update": {}, "nli": None})
    assert decision.action == "Preserve"
