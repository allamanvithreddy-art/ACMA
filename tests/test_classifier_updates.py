from memory.schema import Memory
from memory.store import MemoryStore
from conflict.scope_rules import classify_relationship


store = MemoryStore()
memories = store.get_active_memories()

old_memory = memories[2]


test_statements = [
    "I now prefer Python.",
    "I no longer prefer Java.",
    "I currently prefer Python.",
    "I prefer Python.",
    "Python is also fine."
]


for statement in test_statements:
    new_memory = Memory(
        memory_id="test_memory",
        subject="user",
        attribute="programming_preference",
        value=statement,
        scope="general",
        context=None,
        source="user",
        confidence=0.90
    )

    result = classify_relationship(old_memory, new_memory)

    print("Old memory:", old_memory.value)
    print("New memory:", new_memory.value)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("-" * 60)