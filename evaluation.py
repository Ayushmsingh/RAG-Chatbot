import os
import csv
import time

import matplotlib.pyplot as plt

from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import embedding_models
from src.vectorstore import create_vectorstore
from src.bm_25retriever import create_bm25_retriever
from src.hybrid_retriever import hybrid_retrieve


# ==========================================================
# CONFIGURATION
# ==========================================================

PDF_PATH = "data/sample_data.pdf"

TOP_K = 5
FETCH_K = 20
LAMBDA_MULT = 0.7

BM25_K = 5

HYBRID_K = 5
FAISS_WEIGHT = 0.5
BM25_WEIGHT = 0.5
RRF_K = 60

RESULTS_DIR = "evaluation_results"


# ==========================================================
# EVALUATION DATASET
# ==========================================================

evaluation_data = [
    {
        "question": "What is a Transformer?",
        "expected_terms": ["transformer"],
    },
    {
        "question": "What is self-attention?",
        "expected_terms": ["self-attention"],
    },
    {
        "question": "What is positional encoding?",
        "expected_terms": ["positional encoding"],
    },
    {
        "question": "What is the role of the encoder in a Transformer?",
        "expected_terms": ["encoder"],
    },
    {
        "question": "What is the role of the decoder in a Transformer?",
        "expected_terms": ["decoder"],
    },
]


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def normalize_text(text):
    """
    Normalize text for keyword-based relevance checking.
    """

    return " ".join(
        text.lower().split()
    )


def document_contains_terms(doc, expected_terms):
    """
    Check whether a document contains all expected terms.
    """

    text = normalize_text(
        doc.page_content
    )

    return all(
        normalize_text(term) in text
        for term in expected_terms
    )


def contains_answer(documents, expected_terms):
    """
    Returns True if at least one retrieved document
    contains all expected terms.
    """

    for doc in documents:

        if document_contains_terms(
            doc,
            expected_terms
        ):
            return True

    return False


def reciprocal_rank(documents, expected_terms):
    """
    Calculate Reciprocal Rank.

    Rank 1 -> 1.0
    Rank 2 -> 0.5
    Rank 3 -> 0.333
    etc.

    Returns 0 if no relevant document is found.
    """

    for rank, doc in enumerate(
        documents,
        start=1
    ):

        if document_contains_terms(
            doc,
            expected_terms
        ):

            return 1.0 / rank

    return 0.0


# ==========================================================
# CREATE RESULTS STRUCTURE
# ==========================================================

results = {
    "FAISS": {
        "hits": 0,
        "mrr": 0.0,
        "latency": 0.0,
    },

    "BM25": {
        "hits": 0,
        "mrr": 0.0,
        "latency": 0.0,
    },

    "Hybrid": {
        "hits": 0,
        "mrr": 0.0,
        "latency": 0.0,
    },
}


# ==========================================================
# LOAD PDF
# ==========================================================

print("\n" + "=" * 70)
print("LOADING PDF")
print("=" * 70)

docs = load_pdf(
    PDF_PATH
)

if not docs:

    raise ValueError(
        "No readable content found in the PDF."
    )

print(
    f"Pages loaded: {len(docs)}"
)


# ==========================================================
# CREATE CHUNKS
# ==========================================================

print("\nCreating chunks...")

chunks = split_documents(
    docs
)

if not chunks:

    raise ValueError(
        "No chunks were created from the PDF."
    )

print(
    f"Chunks created: {len(chunks)}"
)


# ==========================================================
# LOAD EMBEDDING MODEL
# ==========================================================

print("\nLoading embedding model...")

embedding_model = embedding_models()

print(
    "Embedding model loaded successfully."
)


# ==========================================================
# CREATE FAISS RETRIEVER
# ==========================================================

print("\nCreating FAISS retriever...")

vectorstore = create_vectorstore(
    chunks,
    embedding_model
)

faiss_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": TOP_K,
        "fetch_k": FETCH_K,
        "lambda_mult": LAMBDA_MULT,
    },
)

print(
    "FAISS retriever initialized."
)


# ==========================================================
# CREATE BM25 RETRIEVER
# ==========================================================

print("\nCreating BM25 retriever...")

bm25_retriever = create_bm25_retriever(
    chunks,
    k=BM25_K,
)

print(
    "BM25 retriever initialized."
)


# ==========================================================
# START EVALUATION
# ==========================================================

print("\n" + "=" * 70)
print("STARTING RETRIEVAL EVALUATION")
print("=" * 70)


for index, item in enumerate(
    evaluation_data,
    start=1
):

    question = item["question"]

    expected_terms = item[
        "expected_terms"
    ]

    print(
        f"\nQuestion {index}: {question}"
    )


    # ======================================================
    # FAISS
    # ======================================================

    start_time = time.perf_counter()

    faiss_docs = faiss_retriever.invoke(
        question
    )

    faiss_latency = (
        time.perf_counter()
        - start_time
    )

    faiss_hit = contains_answer(
        faiss_docs,
        expected_terms
    )

    faiss_rr = reciprocal_rank(
        faiss_docs,
        expected_terms
    )

    results["FAISS"]["hits"] += int(
        faiss_hit
    )

    results["FAISS"]["mrr"] += (
        faiss_rr
    )

    results["FAISS"]["latency"] += (
        faiss_latency
    )

    print(
        f"FAISS  | "
        f"Hit: {faiss_hit} | "
        f"RR: {faiss_rr:.3f} | "
        f"Time: {faiss_latency:.4f}s"
    )


    # ======================================================
    # BM25
    # ======================================================

    start_time = time.perf_counter()

    bm25_docs = bm25_retriever.invoke(
        question
    )

    bm25_latency = (
        time.perf_counter()
        - start_time
    )

    bm25_hit = contains_answer(
        bm25_docs,
        expected_terms
    )

    bm25_rr = reciprocal_rank(
        bm25_docs,
        expected_terms
    )

    results["BM25"]["hits"] += int(
        bm25_hit
    )

    results["BM25"]["mrr"] += (
        bm25_rr
    )

    results["BM25"]["latency"] += (
        bm25_latency
    )

    print(
        f"BM25   | "
        f"Hit: {bm25_hit} | "
        f"RR: {bm25_rr:.3f} | "
        f"Time: {bm25_latency:.4f}s"
    )


    # ======================================================
    # HYBRID
    # ======================================================

    start_time = time.perf_counter()

    hybrid_docs = hybrid_retrieve(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        query=question,
        k=HYBRID_K,
        faiss_weight=FAISS_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        rrf_k=RRF_K,
    )

    hybrid_latency = (
        time.perf_counter()
        - start_time
    )

    hybrid_hit = contains_answer(
        hybrid_docs,
        expected_terms
    )

    hybrid_rr = reciprocal_rank(
        hybrid_docs,
        expected_terms
    )

    results["Hybrid"]["hits"] += int(
        hybrid_hit
    )

    results["Hybrid"]["mrr"] += (
        hybrid_rr
    )

    results["Hybrid"]["latency"] += (
        hybrid_latency
    )

    print(
        f"Hybrid | "
        f"Hit: {hybrid_hit} | "
        f"RR: {hybrid_rr:.3f} | "
        f"Time: {hybrid_latency:.4f}s"
    )


# ==========================================================
# CALCULATE FINAL METRICS
# ==========================================================

total_questions = len(
    evaluation_data
)

if total_questions == 0:

    raise ValueError(
        "Evaluation dataset is empty."
    )


final_results = {
    "FAISS": {
        "Recall@5": 0.0,
        "MRR@5": 0.0,
        "Latency": 0.0,
    },

    "BM25": {
        "Recall@5": 0.0,
        "MRR@5": 0.0,
        "Latency": 0.0,
    },

    "Hybrid": {
        "Recall@5": 0.0,
        "MRR@5": 0.0,
        "Latency": 0.0,
    },
}


for method in results:

    final_results[method][
        "Recall@5"
    ] = (
        results[method]["hits"]
        / total_questions
    )

    final_results[method][
        "MRR@5"
    ] = (
        results[method]["mrr"]
        / total_questions
    )

    final_results[method][
        "Latency"
    ] = (
        results[method]["latency"]
        / total_questions
    )


# ==========================================================
# PRINT FINAL RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)


for method in [
    "FAISS",
    "BM25",
    "Hybrid",
]:

    print(
        f"\n{method}"
    )

    print(
        "-" * 30
    )

    print(
        f"Recall@5     : "
        f"{final_results[method]['Recall@5']:.3f}"
    )

    print(
        f"MRR@5        : "
        f"{final_results[method]['MRR@5']:.3f}"
    )

    print(
        f"Avg Latency  : "
        f"{final_results[method]['Latency']:.4f} sec"
    )


# ==========================================================
# CREATE RESULTS DIRECTORY
# ==========================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ==========================================================
# PREPARE GRAPH DATA
# ==========================================================

methods = [
    "FAISS",
    "BM25",
    "Hybrid",
]

recall_values = [
    final_results[method]["Recall@5"]
    for method in methods
]

mrr_values = [
    final_results[method]["MRR@5"]
    for method in methods
]

latency_values = [
    final_results[method]["Latency"]
    for method in methods
]


# ==========================================================
# GRAPH 1
# RECALL@5 + MRR@5
# ==========================================================

x_positions = range(
    len(methods)
)

width = 0.35


plt.figure(
    figsize=(9, 6)
)


recall_bars = plt.bar(
    [
        x - width / 2
        for x in x_positions
    ],
    recall_values,
    width,
    label="Recall@5",
)


mrr_bars = plt.bar(
    [
        x + width / 2
        for x in x_positions
    ],
    mrr_values,
    width,
    label="MRR@5",
)


plt.title(
    "FAISS vs BM25 vs Hybrid Retrieval"
)

plt.xlabel(
    "Retrieval Method"
)

plt.ylabel(
    "Score"
)

plt.xticks(
    list(x_positions),
    methods
)

plt.ylim(
    0,
    1.1
)

plt.legend()


# Add Recall labels

for bar, value in zip(
    recall_bars,
    recall_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 0.02,

        f"{value:.2f}",

        ha="center",
    )


# Add MRR labels

for bar, value in zip(
    mrr_bars,
    mrr_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 0.02,

        f"{value:.2f}",

        ha="center",
    )


plt.tight_layout()


comparison_path = os.path.join(
    RESULTS_DIR,
    "retrieval_comparison.png",
)


plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight",
)


plt.show()

plt.close()


# ==========================================================
# GRAPH 2
# LATENCY
# ==========================================================

plt.figure(
    figsize=(9, 6)
)


latency_bars = plt.bar(
    methods,
    latency_values,
)


plt.title(
    "Average Retrieval Latency"
)

plt.xlabel(
    "Retrieval Method"
)

plt.ylabel(
    "Latency (seconds)"
)


max_latency = max(
    latency_values
)


if max_latency > 0:

    plt.ylim(
        0,
        max_latency * 1.25
    )


for bar, value in zip(
    latency_bars,
    latency_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value
        + max_latency * 0.02,

        f"{value:.4f}s",

        ha="center",
    )


plt.tight_layout()


latency_path = os.path.join(
    RESULTS_DIR,
    "latency_comparison.png",
)


plt.savefig(
    latency_path,
    dpi=300,
    bbox_inches="tight",
)


plt.show()

plt.close()


# ==========================================================
# GRAPH 3
# RECALL@5 ONLY
# ==========================================================

plt.figure(
    figsize=(8, 5)
)


recall_bars = plt.bar(
    methods,
    recall_values,
)


plt.title(
    "Recall@5 Comparison"
)

plt.xlabel(
    "Retrieval Method"
)

plt.ylabel(
    "Recall@5"
)

plt.ylim(
    0,
    1.1
)


for bar, value in zip(
    recall_bars,
    recall_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 0.02,

        f"{value:.2f}",

        ha="center",
    )


plt.tight_layout()


recall_path = os.path.join(
    RESULTS_DIR,
    "recall_at_5.png",
)


plt.savefig(
    recall_path,
    dpi=300,
    bbox_inches="tight",
)


plt.show()

plt.close()


# ==========================================================
# GRAPH 4
# MRR@5 ONLY
# ==========================================================

plt.figure(
    figsize=(8, 5)
)


mrr_bars = plt.bar(
    methods,
    mrr_values,
)


plt.title(
    "MRR@5 Comparison"
)

plt.xlabel(
    "Retrieval Method"
)

plt.ylabel(
    "MRR@5"
)

plt.ylim(
    0,
    1.1
)


for bar, value in zip(
    mrr_bars,
    mrr_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 0.02,

        f"{value:.2f}",

        ha="center",
    )


plt.tight_layout()


mrr_path = os.path.join(
    RESULTS_DIR,
    "mrr_at_5.png",
)


plt.savefig(
    mrr_path,
    dpi=300,
    bbox_inches="tight",
)


plt.show()

plt.close()


# ==========================================================
# SAVE RESULTS TO CSV
# ==========================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "results.csv",
)


with open(
    csv_path,
    "w",
    newline="",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "Method",
            "Recall@5",
            "MRR@5",
            "Average Latency (s)",
        ]
    )


    for method in methods:

        writer.writerow(
            [
                method,

                final_results[method][
                    "Recall@5"
                ],

                final_results[method][
                    "MRR@5"
                ],

                final_results[method][
                    "Latency"
                ],
            ]
        )


# ==========================================================
# FINAL MESSAGE
# ==========================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print(
    f"1. {comparison_path}"
)

print(
    f"2. {latency_path}"
)

print(
    f"3. {recall_path}"
)

print(
    f"4. {mrr_path}"
)

print(
    f"5. {csv_path}"
)

print("\nAll three retrieval methods were evaluated:")
print("FAISS")
print("BM25")
print("Hybrid")