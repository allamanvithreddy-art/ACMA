from __future__ import annotations

from importlib import import_module

from memory.schema import Memory, MemoryQuery

__all__ = [
    "Memory",
    "MemoryQuery",
    "MemoryStore",
    "MemoryEvolution",
    "MemoryUnderstanding",
    "WorkingMemory",
    "WorkingMemoryItem",
]


def __getattr__(name: str):
    modules = {
        "MemoryStore": "memory.store",
        "MemoryEvolution": "memory.evolution",
        "MemoryUnderstanding": "memory.understanding",
        "WorkingMemory": "memory.working_memory",
        "WorkingMemoryItem": "memory.working_memory",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
