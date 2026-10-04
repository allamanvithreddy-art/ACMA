from memory.store import MemoryStore
from conflict.metadata import compare_metadata


store = MemoryStore()
memories = store.get_active_memories()

old_memory = memories[2]
new_memory = memories[3]

result = compare_metadata(old_memory, new_memory)

print("Old memory:", old_memory.value)
print("New memory:", new_memory.value)
print()
print("Metadata comparison:")

for key, value in result.items():
    print(f"{key}: {value}")