
import re

from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def tokenize(self, text):
        # Keep technical terms such as NEG-CLIP together.
        return re.findall(
            r"[a-z0-9]+(?:[-_][a-z0-9]+)*",
            text.lower()
        )

    def keyword_search(self, query, top_k=10, sources=None):
        # Get indexed chunks from ChromaDB.
        where_filter = None

        if sources:
            where_filter = {
                "source": {"$in": sources}
            }

        results = self.vector_store.collection.get(
            where=where_filter,
            include=["documents", "metadatas"]
        )

        documents = results["documents"]
        metadatas = results["metadatas"]
        ids = results["ids"]

        if not documents:
            return []

        # Tokenize every indexed chunk.
        tokenized_corpus = [
            self.tokenize(document)
            for document in documents
        ]

        # Build the BM25 index.
        bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        # Calculate BM25 relevance scores.
        scores = bm25.get_scores(query_tokens)

        # Sort chunks by their BM25 scores.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [
            {
                "id": ids[i],
                "text": documents[i],
                "source": metadatas[i]["source"],
                "page": metadatas[i]["page"],
                "bm25_score": float(scores[i])
            }
            for i in ranked_indices[:top_k]
            if scores[i] > 0
        ]

    def search(self, query, top_k=5, sources=None):
        # Retrieve candidates using semantic similarity.
        semantic_results = self.vector_store.search(
            query,
            top_k=20,
            sources=sources
        )

        # Retrieve candidates using BM25.
        keyword_results = self.keyword_search(
            query,
            top_k=20,
            sources=sources
        )

        # Reciprocal Rank Fusion.
        rrf_scores = {}
        combined_results = {}

        k = 60

        for rank, result in enumerate(
            semantic_results, start=1
        ):
            # Match chunks across the two retrieval methods.
            chunk_id = (
                result["source"],
                result["page"],
                result["text"]
            )

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0)
                + 1 / (k + rank)
            )

            combined_results[chunk_id] = result

        for rank, result in enumerate(
            keyword_results, start=1
        ):
            chunk_id = (
                result["source"],
                result["page"],
                result["text"]
            )

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0)
                + 1 / (k + rank)
            )

            combined_results[chunk_id] = result

        # Sort candidates by their fused scores.
        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )

        return [
            {
                **combined_results[chunk_id],
                "rrf_score": rrf_scores[chunk_id]
            }
            for chunk_id in ranked_ids[:top_k]
        ]
