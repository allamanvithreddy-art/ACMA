from __future__ import annotations

from memory.schema import Memory
from memory.store import MemoryStore


def calculate_similarity(
    old_memory: Memory,
    new_memory: Memory,
) -> float:
    """
    Calculate semantic similarity using ACMA's shared embedding model.

    The model lifecycle is owned by MemoryStore so the application does not
    create independent SentenceTransformer instances in multiple modules.
    """

    store = MemoryStore()

    old_text = store.memory_to_text(old_memory)
    new_text = store.memory_to_text(new_memory)

    if not old_text.strip() or not new_text.strip():
        return 0.0

    model = store._load_embedding_model()

    embeddings = model.encode(
        [old_text, new_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return float(embeddings[0] @ embeddings[1])
