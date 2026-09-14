def compare_metadata(old_memory, new_memory):
    """
    Compare two Memory objects using their metadata.
    """

    result = {
        "same_subject": old_memory.subject == new_memory.subject,
        "same_attribute": old_memory.attribute == new_memory.attribute,
        "same_context": old_memory.context == new_memory.context,
        "old_scope": old_memory.scope,
        "new_scope": new_memory.scope,
        "old_value": old_memory.value,
        "new_value": new_memory.value,
    }

    result["related"] = (
        result["same_subject"]
        and result["same_attribute"]
    )

    return result