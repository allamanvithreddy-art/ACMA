from memory.schema import Memory


memory = Memory(
    memory_id="mem_001",
    subject="user",
    attribute="food_preference",
    value="vegetarian",
    scope="general",
    context=None,
    source="user",
    confidence=0.95
)

print(memory)
print()
print(memory.to_dict())