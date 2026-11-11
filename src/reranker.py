
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        # Load the pretrained cross-encoder model.
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, chunks, top_k=5):
        if not chunks:
            return []

        # Create query-document pairs.
        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]

        # Predict relevance scores for each pair.
        scores = self.model.predict(pairs)

        # Attach relevance scores to chunks.
        scored_chunks = []

        for chunk, score in zip(chunks, scores):
            scored_chunks.append({
                **chunk,
                "rerank_score": float(score)
            })

        # Sort by cross-encoder relevance score.
        scored_chunks.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return scored_chunks[:top_k]
