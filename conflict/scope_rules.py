def analyze_scope(old_memory, new_memory):
    """
    Analyze the scope and context of two related memories.
    """

    if old_memory.scope == "general" and new_memory.scope == "specific_event":
        return {
            "category": "possible_exception",
            "action_hint": "Preserve",
            "reason": (
                "The new memory describes a specific event, "
                "not a general preference change."
            )
        }

    if (
        old_memory.scope == "general"
        and new_memory.scope == "general"
        and old_memory.context != new_memory.context
    ):
        return {
            "category": "different_context",
            "action_hint": "Ask",
            "reason": (
                "The memories have the same general scope "
                "but refer to different contexts."
            )
        }

    if old_memory.scope == "general" and new_memory.scope == "general":
        return {
            "category": "general_update",
            "action_hint": "Evaluate",
            "reason": (
                "Both memories are general statements. "
                "Further semantic comparison is required."
            )
        }

    return {
        "category": "unknown",
        "action_hint": "Ask",
        "reason": "The memory scopes could not be confidently interpreted."
    }