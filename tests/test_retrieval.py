from memory.store import MemoryStore
from memory.schema import MemoryQuery


def test_retrieve_related_memories():
    store = MemoryStore()

    # A retrieval query is not a stored memory.
    # Therefore it does not need confidence, importance,
    # version, decision state, etc.
    query = MemoryQuery(
        subject="user",
        attribute="programming_preference",
        value="Python",
        scope="general",
        context="general"
    )

    results = store.retrieve_related_memories(
        query=query,
        top_k=3
    )

    print("\nRetrieved memories:")
    print("-" * 60)

    for memory, score in results:
        print(
            f"{score:.4f} | "
            f"{memory.subject} | "
            f"{memory.attribute} | "
            f"{memory.value}"
        )

    # Basic correctness checks
    assert isinstance(results, list)
    assert len(results) <= 3

    scores = []

    for memory, score in results:
        assert memory is not None
        assert isinstance(score, float)

        scores.append(score)

    # Results must be sorted by similarity, highest first.
    assert scores == sorted(scores, reverse=True)