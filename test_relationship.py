from memory.store import MemoryStore
from conflict.scope_rules import classify_relationship


store = MemoryStore()
memories = store.get_active_memories()


def test_case(number, old_index, new_index):
    old_memory = memories[old_index]
    new_memory = memories[new_index]

    result = classify_relationship(old_memory, new_memory)

    print(f"Test {number}")
    print("-" * 50)
    print("Old memory:", old_memory.value)
    print("New memory:", new_memory.value)

    for key, value in result.items():
        print(f"{key}: {value}")

    print()


test_case(
    number=1,
    old_index=0,
    new_index=1
)

test_case(
    number=2,
    old_index=2,
    new_index=3
)

test_case(
    number=3,
    old_index=2,
    new_index=2
)