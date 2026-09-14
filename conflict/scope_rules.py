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

    # Case 4: General memory versus specific event
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

    # Case 5: Specific event versus general memory
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

    # Case 6: Different contexts
    if old_memory.context != new_memory.context:
        return {
            "relationship": "different_context",
            "action_hint": "Ask",
            "reason": (
                "The memories have different contexts. "
                "The system needs more evidence before replacing anything."
            )
        }

    # Case 7: Both memories are general
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

    # Case 8: Fallback
    return {
        "relationship": "ambiguous",
        "action_hint": "Ask",
        "reason": "The relationship could not be confidently classified."
    }