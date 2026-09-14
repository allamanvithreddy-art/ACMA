from memory.store import MemoryStore
from conflict.scope_rules import analyze_scope


store = MemoryStore()
memories = store.get_active_memories()

print("Test 1: General preference versus specific event")
print("-" * 50)

old_memory = memories[0]
new_memory = memories[1]

result = analyze_scope(old_memory, new_memory)

for key, value in result.items():
    print(f"{key}: {value}")


print()
print("Test 2: General preference versus general preference")
print("-" * 50)

old_memory = memories[2]
new_memory = memories[3]

result = analyze_scope(old_memory, new_memory)

for key, value in result.items():
    print(f"{key}: {value}")