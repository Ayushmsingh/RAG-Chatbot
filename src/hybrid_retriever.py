from typing import List

from langchain_core.documents import Document


def hybrid_retrieve(
    faiss_retriever,
    bm25_retriever,
    query: str,
    k: int = 5,
    faiss_weight: float = 0.5,
    bm25_weight: float = 0.5,
    rrf_k: int = 60,
) -> List[Document]:
    """
    Retrieve documents using both FAISS and BM25
    and combine their rankings using Reciprocal Rank Fusion (RRF).

    FAISS:
        Semantic / dense retrieval

    BM25:
        Keyword / sparse retrieval

    RRF:
        Combines rankings without requiring the two
        retrievers to have comparable score scales.
    """

    # Retrieve candidates from both systems
    faiss_docs = faiss_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)

    scores = {}
    documents = {}

    # -------------------- FAISS Results --------------------

    for rank, doc in enumerate(faiss_docs, start=1):

        doc_id = _get_document_id(doc)

        if doc_id not in documents:
            documents[doc_id] = doc

        scores[doc_id] = scores.get(doc_id, 0.0) + (
            faiss_weight / (rrf_k + rank)
        )

    # -------------------- BM25 Results --------------------

    for rank, doc in enumerate(bm25_docs, start=1):

        doc_id = _get_document_id(doc)

        if doc_id not in documents:
            documents[doc_id] = doc

        scores[doc_id] = scores.get(doc_id, 0.0) + (
            bm25_weight / (rrf_k + rank)
        )

    # -------------------- Rank Results --------------------

    ranked_ids = sorted(
        scores.keys(),
        key=lambda doc_id: scores[doc_id],
        reverse=True
    )

    return [
        documents[doc_id]
        for doc_id in ranked_ids[:k]
    ]


def _get_document_id(doc: Document) -> str:
    """
    Generate a stable identifier for a document chunk.
    """

    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    content = doc.page_content.strip()

    return f"{source}|{page}|{content}"