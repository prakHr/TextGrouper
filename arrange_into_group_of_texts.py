from semantic_clusterer import SemanticClusterer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pprint import pprint

import numpy as np
import re


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# NLI model used for meaningfulness checking
NLI_MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"

# Semantic similarity threshold
SIMILARITY_THRESHOLD = 0.60

# Meaningfulness threshold
MEANINGFULNESS_THRESHOLD = 0.50

EPSILON = 0.001


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Embedding model loaded.")


print("\nLoading NLI model...")

from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model=NLI_MODEL_NAME
)

print("NLI model loaded.")


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def normalize_text(text):
    """
    Normalize text for duplicate detection.
    """

    text = text.lower().strip()

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

def remove_duplicates(list_of_texts):

    seen = set()

    result = []

    for text in list_of_texts:

        normalized = normalize_text(text)

        if normalized not in seen:

            seen.add(normalized)

            result.append(text)

        else:

            print(
                f"DUPLICATE REMOVED -> {text}"
            )

    return result


# ============================================================
# BASIC GARBAGE DETECTION
# ============================================================

def looks_like_garbage(text):

    normalized = normalize_text(text)

    # Empty
    if not normalized:
        return True

    # Very short
    words = normalized.split()

    if len(words) == 1:

        word = words[0]

        # Single random-looking word
        if len(word) < 3:
            return True

    # Repeated same character
    if len(set(normalized.replace(" ", ""))) <= 2:
        return True

    # Extremely high digit ratio
    characters = normalized.replace(" ", "")

    if characters:

        digit_ratio = sum(
            c.isdigit()
            for c in characters
        ) / len(characters)

        if digit_ratio > 0.7:
            return True

    return False


# ============================================================
# SEMANTIC OUTLIER DETECTION
# ============================================================

def remove_semantic_outliers(
    list_of_texts,
    model,
    threshold=SIMILARITY_THRESHOLD
):

    if len(list_of_texts) <= 1:
        return list_of_texts

    embeddings = model.encode(
        list_of_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarity_matrix = cosine_similarity(
        embeddings
    )

    result = []

    print("\nSemantic similarity:")

    for i, text in enumerate(list_of_texts):

        similarities = np.delete(
            similarity_matrix[i],
            i
        )

        max_similarity = np.max(
            similarities
        )

        print(
            f"{max_similarity:.4f} -> {text}"
        )

        if max_similarity >= threshold:

            result.append(text)

        else:

            print(
                f"SEMANTIC OUTLIER REMOVED -> {text}"
            )

    return result


# ============================================================
# MEANINGFULNESS CHECK
# ============================================================

def check_meaningfulness(
    list_of_texts,
    classifier,
    threshold=MEANINGFULNESS_THRESHOLD-EPSILON
):

    if not list_of_texts:
        return []

    result = []

    candidate_labels = [
        "a meaningful natural language sentence",
        "random meaningless text"
    ]

    print("\nMeaningfulness check:")

    for text in list_of_texts:

        # ----------------------------------------------------
        # First perform cheap heuristic check
        # ----------------------------------------------------

        if looks_like_garbage(text):

            print(
                f"GARBAGE REMOVED -> {text}"
            )

            continue

        # ----------------------------------------------------
        # Zero-shot classification
        # ----------------------------------------------------

        output = classifier(
            text,
            candidate_labels,
            multi_label=False
        )

        labels = output["labels"]
        scores = output["scores"]

        meaningful_score = 0.0

        for label, score in zip(
            labels,
            scores
        ):

            if label == candidate_labels[0]:

                meaningful_score = score

        print(
            f"{meaningful_score:.4f} -> {text}"
        )

        if meaningful_score >= threshold:

            result.append(text)

        else:

            print(
                f"MEANINGLESS REMOVED -> {text}"
            )

    return result


# ============================================================
# COMPLETE SENTENCE CLEANER
# ============================================================

def clean_sentences(
    list_of_texts,
    embedding_model,
    nli_classifier
):

    print("\n")
    print("=" * 70)
    print("STARTING SENTENCE CLEANING")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1
    # Remove exact duplicates
    # --------------------------------------------------------

    print("\nSTEP 1: DUPLICATE REMOVAL")

    texts = remove_duplicates(
        list_of_texts
    )

    print(
        f"Remaining: {len(texts)}"
    )

    # --------------------------------------------------------
    # STEP 2
    # Remove obvious garbage
    # --------------------------------------------------------

    print("\nSTEP 2: BASIC GARBAGE FILTER")

    texts = [
        text
        for text in texts
        if not looks_like_garbage(text)
    ]

    print(
        f"Remaining: {len(texts)}"
    )

    # --------------------------------------------------------
    # STEP 3
    # Semantic outlier detection
    # --------------------------------------------------------

    print("\nSTEP 3: SEMANTIC OUTLIER FILTER")

    texts = remove_semantic_outliers(
        texts,
        embedding_model,
        SIMILARITY_THRESHOLD
    )

    print(
        f"Remaining: {len(texts)}"
    )

    # --------------------------------------------------------
    # STEP 4
    # Meaningfulness detection
    # --------------------------------------------------------

    print("\nSTEP 4: MEANINGFULNESS CHECK")

    texts = check_meaningfulness(
        texts,
        nli_classifier,
        MEANINGFULNESS_THRESHOLD
    )

    print(
        f"Remaining: {len(texts)}"
    )

    return texts


# ============================================================
# CLUSTERING
# ============================================================

def arrange_into_clusters_from_list_of_texts(
    list_of_texts
):

    # --------------------------------------------------------
    # STEP 1: Clean the entire dataset
    # --------------------------------------------------------

    cleaned_texts = clean_sentences(
        list_of_texts,
        embedding_model,
        classifier
    )

    print("\n")
    print("=" * 70)
    print("CLEANED DATA")
    print("=" * 70)

    pprint(cleaned_texts)

    if not cleaned_texts:
        return []

    # --------------------------------------------------------
    # STEP 2: Cluster cleaned data
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CLUSTERING")
    print("=" * 70)

    clusterer = SemanticClusterer()

    clusters = clusterer.cluster(
        cleaned_texts,
        return_format="detailed"
    )

    # --------------------------------------------------------
    # STEP 3: Create final result
    # --------------------------------------------------------

    results = []

    for cluster in clusters:

        results.append(
            {
                "list_of_texts": cluster["items"],
                "model_name": EMBEDDING_MODEL_NAME,
                "THRESHOLD": SIMILARITY_THRESHOLD,
                "EPSILON": EPSILON,
                "topic_label": cluster.get(
                    "topic_label"
                ),
                "confidence": cluster.get(
                    "confidence"
                )
            }
        )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    list_of_texts = [

        "Random Sentence",

        "How do I reset my password?",
        "I forgot my login password",

        "What are your business hours?",
        "When do you open on weekdays?",
        
        "Random Sentence",
        "Random Sentence",
        "My package hasn't arrived",
        "Where is my order?",

        "Random Sentence",
        "Random Sentence",

    ]

    results = (
        arrange_into_clusters_from_list_of_texts(
            list_of_texts
        )
    )

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    pprint(results)
