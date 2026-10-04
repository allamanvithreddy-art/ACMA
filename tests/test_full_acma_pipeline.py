"""
ACMA Full Pipeline Test

Pipeline:

Memory Query
    ↓
Semantic Retrieval
    ↓
Safety Gate
    ↓
NLI
    ↓
Metadata Agreement
    ↓
Update Signal
    ↓
Evidence Score
    ↓
Decision Policy

No similarity, NLI confidence, metadata agreement,
or update signal is hardcoded.
"""

from memory.schema import Memory
from memory.store import MemoryStore

from conflict.safety_gate import evaluate_rule_safety

from conflict.update_detector import (
    detect_update_signals
)

from conflict.evidence_score import (
    calculate_nli_confidence,
    calculate_nli_margin,
    calculate_metadata_agreement,
    calculate_update_signal,
    calculate_evidence_score
)

from conflict.nli_checker import (
    compare_statements
)

from conflict.decision_policy import (
    decide_action
)


# ============================================================
# MEMORY CREATION
# ============================================================

def create_memory(
    memory_id,
    value,
    attribute="general",
    scope="general",
    context="general",
    source="user",
    confidence=1.0,
    importance=0.5
):
    """
    Create a test memory.
    """

    return Memory(
        memory_id=memory_id,
        subject="user",
        attribute=attribute,
        value=value,
        scope=scope,
        context=context,
        source=source,
        confidence=confidence,
        importance=importance
    )


# ============================================================
# NLI RESULT HELPERS
# ============================================================

def run_nli(
    old_memory,
    new_memory
):
    """
    Run the existing ACMA NLI implementation.
    """

    return compare_statements(
        old_memory.value,
        new_memory.value
    )


# ============================================================
# SINGLE MEMORY COMPARISON
# ============================================================

def evaluate_pair(
    old_memory,
    new_memory,
    similarity
):
    """
    Run one old/new memory pair through:

        Safety Gate
        NLI
        Evidence Score
        Decision Policy
    """

    print("\n" + "=" * 80)

    print("OLD MEMORY:")
    print(old_memory.value)

    print("\nNEW MEMORY:")
    print(new_memory.value)

    print("\nRAG SIMILARITY:")
    print(round(similarity, 4))

    # --------------------------------------------------------
    # 1. SAFETY GATE
    # --------------------------------------------------------

    safety = evaluate_rule_safety(
        old_memory,
        new_memory,
        similarity
    )

    print("\nSAFETY GATE")
    print("Safe:", safety.safe)
    print("Next stage:", safety.next_stage)
    print("Relationship:", safety.relationship)
    print("Action hint:", safety.action_hint)
    print("Reason:", safety.reason)

    # --------------------------------------------------------
    # SAFE CASE
    # --------------------------------------------------------

    if safety.safe:

        return {
            "action": safety.action_hint,
            "safety": safety,
            "nli": None,
            "evidence_score": None,
            "llm_required": False
        }

    # --------------------------------------------------------
    # 2. NLI
    # --------------------------------------------------------

    nli = run_nli(
        old_memory,
        new_memory
    )

    print("\nNLI")
    print(nli)

    nli_confidence = calculate_nli_confidence(
        nli
    )

    nli_margin = calculate_nli_margin(
        nli
    )

    # --------------------------------------------------------
    # 3. METADATA AGREEMENT
    # --------------------------------------------------------

    metadata_agreement = calculate_metadata_agreement(
        old_memory,
        new_memory
    )

    print("\nMETADATA AGREEMENT")
    print(metadata_agreement)

    # --------------------------------------------------------
    # 4. UPDATE SIGNAL
    # --------------------------------------------------------

    update_result = detect_update_signals(
        new_memory.value
    )

    update_signal = calculate_update_signal(
        update_result
    )

    print("\nUPDATE SIGNAL")
    print(update_result)
    print("Numeric update signal:", update_signal)

    # --------------------------------------------------------
    # 5. SOURCE RELIABILITY
    # --------------------------------------------------------

    source_reliability = min(
        float(old_memory.confidence),
        float(new_memory.confidence)
    )

    # --------------------------------------------------------
    # 6. EVIDENCE SCORE
    # --------------------------------------------------------

    evidence_score = calculate_evidence_score(
        nli_confidence=nli_confidence,
        nli_margin=nli_margin,
        similarity=similarity,
        metadata_agreement=metadata_agreement,
        update_signal=update_signal,
        source_reliability=source_reliability
    )

    print("\nEVIDENCE SCORE")
    print(evidence_score)

    # --------------------------------------------------------
    # 7. DECISION POLICY
    # --------------------------------------------------------

    nli_label = nli.get(
        "label",
        nli.get(
            "relationship",
            "unknown"
        )
    )

    action = decide_action(
        relationship=safety.relationship,
        evidence_score=evidence_score,
        nli_label=nli_label,
        nli_confidence=nli_confidence,
        nli_margin=nli_margin,
        update_signal=update_signal
    )

    print("\nDECISION")
    print(action)

    return {
        "action": action,
        "safety": safety,
        "nli": nli,
        "nli_confidence": nli_confidence,
        "nli_margin": nli_margin,
        "metadata_agreement": metadata_agreement,
        "update_signal": update_signal,
        "evidence_score": evidence_score,
        "llm_required": action == "LLM_FALLBACK"
    }


# ============================================================
# DIRECT PAIR TESTS
# ============================================================

def test_full_acma_pipeline():

    cases = [

        {
            "name": "Exact duplicate",

            "old": create_memory(
                "old_1",
                "I use Python for ACMA",
                attribute="programming_preference"
            ),

            "new": create_memory(
                "new_1",
                "I use Python for ACMA",
                attribute="programming_preference"
            ),

            "expected": "Ignore"
        },

        {
            "name": "Different attribute",

            "old": create_memory(
                "old_2",
                "I prefer vegetarian food",
                attribute="food_preference"
            ),

            "new": create_memory(
                "new_2",
                "I use Python for development",
                attribute="programming_preference"
            ),

            "expected": "Preserve"
        },

        {
            "name": "Explicit programming update",

            "old": create_memory(
                "old_3",
                "I prefer Java",
                attribute="programming_preference"
            ),

            "new": create_memory(
                "new_3",
                "I now prefer Python",
                attribute="programming_preference"
            ),

            "expected": None
        },

        {
            "name": "Specific event exception",

            "old": create_memory(
                "old_4",
                "I am vegetarian",
                attribute="food_preference",
                scope="general",
                context="general"
            ),

            "new": create_memory(
                "new_4",
                "I ate chicken at a wedding",
                attribute="food_event",
                scope="specific_event",
                context="wedding"
            ),

            "expected": None
        },

        {
            "name": "Location update",

            "old": create_memory(
                "old_5",
                "I live in Hyderabad",
                attribute="location"
            ),

            "new": create_memory(
                "new_5",
                "I now live in Bengaluru",
                attribute="location"
            ),

            "expected": None
        }
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    # These direct pair tests don't need RAG because they are
    # testing the conflict pipeline itself.
    #
    # The similarity is calculated dynamically using the same
    # embedding model as MemoryStore.
    # --------------------------------------------------------

    store = MemoryStore()

    correct = 0

    evaluated = 0

    for case in cases:

        old_memory = case["old"]

        new_memory = case["new"]

        # ----------------------------------------------------
        # Calculate actual semantic similarity
        # ----------------------------------------------------

        retrieved = store.retrieve_related_memories(
            new_memory,
            top_k=10,
            similarity_threshold=0.0
        )

        similarity = None

        for memory, score in retrieved:

            if memory.value == old_memory.value:

                similarity = score

                break

        # ----------------------------------------------------
        # If old memory is not in the persistent store,
        # calculate its similarity directly.
        # ----------------------------------------------------

        if similarity is None:

            model = store._load_embedding_model()

            old_text = store.memory_to_text(
                old_memory
            )

            new_text = store.memory_to_text(
                new_memory
            )

            embeddings = model.encode(
                [
                    old_text,
                    new_text
                ],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            similarity = float(
                embeddings[0] @ embeddings[1]
            )

        result = evaluate_pair(
            old_memory,
            new_memory,
            similarity
        )

        predicted = result["action"]

        expected = case["expected"]

        print("\nCASE:", case["name"])
        print("EXPECTED:", expected)
        print("PREDICTED:", predicted)

        # ----------------------------------------------------
        # Some cases are intentionally ambiguous because
        # the final LLM stage has not been integrated yet.
        #
        # We therefore only enforce deterministic expectations
        # for the cases where the current pipeline should know
        # the answer.
        # ----------------------------------------------------

        if expected is not None:

            evaluated += 1

            if predicted == expected:

                correct += 1

                print("PASS")

            else:

                print("FAIL")

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    if evaluated > 0:

        accuracy = (
            correct
            /
            evaluated
        ) * 100

    else:

        accuracy = 0.0

    print("\n" + "=" * 80)
    print("DETERMINISTIC PIPELINE ACCURACY")
    print(
        f"{correct}/{evaluated}"
    )
    print(
        f"{accuracy:.2f}%"
    )

    assert correct == evaluated