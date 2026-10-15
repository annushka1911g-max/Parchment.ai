import math
import re
from typing import List, Optional, Tuple, Dict
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text into clean lowercase alphanumeric words.
    """
    if not text:
        return []
    return [word.lower() for word in re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text)]


class HybridRetriever:
    """
    Production-grade Hybrid Retriever combining:
    - Dense Vector Search (FAISS)
    - Sparse Keyword Search (BM25Okapi)
    via Reciprocal Rank Fusion (RRF) with tunable alpha weighting.
    """

    def __init__(self, documents: List[Document], vector_store=None):
        self.documents = documents
        self.vector_store = vector_store
        self.tokenized_corpus = [
            tokenize_text(doc.page_content)
            for doc in documents
        ]
        self.bm25 = (
            BM25Okapi(self.tokenized_corpus)
            if self.tokenized_corpus and len(self.tokenized_corpus) > 0
            else None
        )

    def search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.65,
        source_filter: Optional[str] = None,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Executes hybrid search across vector store and BM25 index.

        Args:
            query: User's question or search query.
            k: Number of top documents to return.
            alpha: Weight for dense search (0.0 = BM25 only, 1.0 = FAISS only, 0.65 = balanced hybrid).
            source_filter: Optional document name filter (e.g. 'report.pdf').
            rrf_k: Constant for Reciprocal Rank Fusion (default 60).
        """
        if not self.documents:
            return []

        fetch_k = min(len(self.documents), max(k * 3, 10))

        # ----------------------------------------------------
        # 1. Dense Semantic Search (FAISS)
        # ----------------------------------------------------
        dense_ranked: List[Tuple[Document, int]] = []
        if self.vector_store is not None and alpha > 0.0:
            try:
                # similarity_search_with_score returns (doc, distance)
                dense_results = self.vector_store.similarity_search_with_score(query, k=fetch_k)
                for rank, (doc, _dist) in enumerate(dense_results, start=1):
                    dense_ranked.append((doc, rank))
            except Exception:
                # Fallback to standard similarity search
                dense_results = self.vector_store.similarity_search(query, k=fetch_k)
                for rank, doc in enumerate(dense_results, start=1):
                    dense_ranked.append((doc, rank))

        # ----------------------------------------------------
        # 2. Sparse Keyword Search (BM25)
        # ----------------------------------------------------
        bm25_ranked: List[Tuple[Document, int]] = []
        if self.bm25 is not None and alpha < 1.0:
            tokenized_query = tokenize_text(query)
            if tokenized_query:
                scores = self.bm25.get_scores(tokenized_query)
                # Sort indices by score descending
                scored_indices = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True
                )
                # Filter out zero scores if possible, keep up to fetch_k
                for rank, idx in enumerate(scored_indices[:fetch_k], start=1):
                    if scores[idx] > 0.0:
                        bm25_ranked.append((self.documents[idx], rank))

        # ----------------------------------------------------
        # 3. Reciprocal Rank Fusion (RRF)
        # ----------------------------------------------------
        # Key function to uniquely identify chunks
        def get_chunk_key(doc: Document) -> Tuple:
            meta = doc.metadata or {}
            return (
                meta.get("source", ""),
                meta.get("page", 0),
                meta.get("chunk", 0),
                doc.page_content[:80]
            )

        doc_map: Dict[Tuple, Document] = {}
        rrf_scores: Dict[Tuple, float] = {}
        sources_found: Dict[Tuple, str] = {}

        # Add Dense contributions
        weight_dense = alpha if bm25_ranked else 1.0
        for doc, rank in dense_ranked:
            key = get_chunk_key(doc)
            doc_map[key] = doc
            contrib = weight_dense * (1.0 / (rrf_k + rank))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + contrib
            sources_found[key] = "Dense"

        # Add BM25 contributions
        weight_bm25 = (1.0 - alpha) if dense_ranked else 1.0
        for doc, rank in bm25_ranked:
            key = get_chunk_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
                sources_found[key] = "Keyword"
            else:
                sources_found[key] = "Hybrid"
            contrib = weight_bm25 * (1.0 / (rrf_k + rank))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + contrib

        if not rrf_scores:
            # Fallback if both sparse and dense returned nothing
            return self.documents[:k]

        # Calculate max theoretical RRF score for confidence normalization
        max_theoretical_rrf = (weight_dense / (rrf_k + 1)) + (weight_bm25 / (rrf_k + 1))
        if max_theoretical_rrf == 0:
            max_theoretical_rrf = 1.0

        # Filter by source if requested
        ranked_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        filtered_docs: List[Document] = []

        for key in ranked_keys:
            doc = doc_map[key]
            doc_source = doc.metadata.get("source", "")
            if source_filter and source_filter != "All Documents" and doc_source != source_filter:
                continue

            raw_score = rrf_scores[key]
            # Normalize confidence percentage between 55% and 99%
            norm_ratio = raw_score / max_theoretical_rrf
            confidence = min(99.0, max(52.0, round(norm_ratio * 100, 1)))

            # Clone/enrich metadata
            enriched_meta = dict(doc.metadata)
            enriched_meta["rrf_score"] = round(raw_score, 4)
            enriched_meta["confidence_score"] = confidence
            enriched_meta["retrieval_mode"] = sources_found.get(key, "Hybrid")

            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=enriched_meta
            )
            filtered_docs.append(enriched_doc)
            if len(filtered_docs) >= k:
                break

        return filtered_docs