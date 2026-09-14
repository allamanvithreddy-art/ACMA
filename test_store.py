from memory.store import MemoryStore


store = MemoryStore()

memories = store.get_active_memories()

for memory in memories:
    print(
        memory.memory_id,
        "|",
        memory.attribute,
        "|",
        memory.value,
        "|",
        memory.scope
    )