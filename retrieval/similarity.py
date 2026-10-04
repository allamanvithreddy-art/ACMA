from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load once
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def calculate_similarity(old_memory, new_memory):

    old_text = (
        f"{old_memory.subject} "
        f"{old_memory.attribute} "
        f"{old_memory.value} "
        f"{old_memory.context}"
    )

    new_text = (
        f"{new_memory.subject} "
        f"{new_memory.attribute} "
        f"{new_memory.value} "
        f"{new_memory.context}"
    )


    embeddings = model.encode(
        [
            old_text,
            new_text
        ]
    )


    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]


    return float(score)