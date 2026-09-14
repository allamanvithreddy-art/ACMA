import json
from pathlib import Path

from memory.schema import Memory


class MemoryStore:
    def __init__(self, file_path="data/sample_memories.json"):
        self.file_path = Path(file_path)

    def load_memories(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return [
            Memory(**memory_data)
            for memory_data in data
        ]

    def get_active_memories(self):
        memories = self.load_memories()
        return [
            memory for memory in memories
            if memory.active
        ]