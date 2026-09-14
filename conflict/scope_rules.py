from conflict.update_detector import detect_update_signals


def classify_relationship(old_memory, new_memory):
    """
    Classify the relationship between two memories.

    This is an initial rule-based classifier.
    NLI and LLM fallback will be added later.
    """

    # Case 1: Different subjects
    if old_memory.subject != new_memory.subject:
        return {
            "relationship": "different_subject",
            "action_hint": "Preserve",
            "reason": "The memories belong to different subjects."
        }

    # Case 2: Different attributes
    if old_memory.attribute != new_memory.attribute:
        return {
            "relationship": "different_attribute",
            "action_hint": "Preserve",
            "reason": "The memories describe different attributes."
        }

    # Case 3: Same value
    if old_memory.value.lower().strip() == new_memory.value.lower().strip():
        return {
            "relationship": "duplicate",
            "action_hint": "Preserve",
            "reason": "The new memory contains the same value as the old memory."
        }

    # Case 4: Detect update signals
    update_info = detect_update_signals(new_memory.value)

    if update_info["explicit_update"]:
        return {
            "relationship": "explicit_update",
            "action_hint": "Resolve",
            "reason": (
                "The new memory contains language indicating "
                "that the previous memory may have changed."
            ),
            "signals": update_info
        }

    # Case 5: Detect temporal signals
    if update_info["temporal_change"]:
        return {
            "relationship": "temporal_change",
            "action_hint": "Evaluate",
            "reason": (
                "The new memory contains temporal language. "
                "Further evidence is required."
            ),
            "signals": update_info
        }

    # Case 6: General memory versus specific event
    if (
        old_memory.scope == "general"
        and new_memory.scope == "specific_event"
    ):
        return {
            "relationship": "specific_event",
            "action_hint": "Preserve",
            "reason": (
                "The new memory describes a specific event "
                "and does not necessarily replace the general memory."
            )
        }

    # Case 7: Specific event versus general memory
    if (
        old_memory.scope == "specific_event"
        and new_memory.scope == "general"
    ):
        return {
            "relationship": "possible_general_update",
            "action_hint": "Evaluate",
            "reason": (
                "The new general statement may update an earlier "
                "event-specific memory."
            )
        }

    # Case 8: Different contexts
    if old_memory.context != new_memory.context:
        return {
            "relationship": "different_context",
            "action_hint": "Ask",
            "reason": (
                "The memories have different contexts. "
                "The system needs more evidence before replacing anything."
            )
        }

    # Case 9: Both memories are general
    if (
        old_memory.scope == "general"
        and new_memory.scope == "general"
    ):
        return {
            "relationship": "possible_general_conflict",
            "action_hint": "Evaluate",
            "reason": (
                "Both memories are general statements. "
                "NLI and evidence evaluation are required."
            )
        }

    # Case 10: Fallback
    return {
        "relationship": "ambiguous",
        "action_hint": "Ask",
        "reason": "The relationship could not be confidently classified."
    }