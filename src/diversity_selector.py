
import numpy as np


class DiversitySelector:
    def __init__(self, embedding_model):
        """
        Reuse the embedding model from VectorStore.
        """
        self.model = embedding_model

    def select(self, chunks, top_k=5, lambda_param=0.8):
        """
        Select relevant but diverse chunks using MMR.

        lambda_param:
            1.0 -> only relevance
            0.0 -> only diversity after the first selection
        """

        if not chunks or top_k <= 0:
            return []

        if not 0 <= lambda_param <= 1:
            raise ValueError(
                "lambda_param must be between 0 and 1"
            )

        # Compute normalized embeddings for all candidate chunks.
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        # Cosine similarity between all candidate chunks.
        similarity_matrix = embeddings @ embeddings.T

        # Convert Cross-Encoder scores into a 0-1 range.
        scores = np.array(
            [chunk["rerank_score"] for chunk in chunks],
            dtype=float
        )

        min_score = scores.min()
        max_score = scores.max()

        if max_score > min_score:
            relevance_scores = (
                (scores - min_score) / (max_score - min_score)
            )
        else:
            relevance_scores = np.ones(len(chunks))

        selected_indices = []
        remaining_indices = list(range(len(chunks)))

        # Start with the most relevant chunk.
        first_index = int(np.argmax(relevance_scores))

        selected_indices.append(first_index)
        remaining_indices.remove(first_index)

        while (
            remaining_indices
            and len(selected_indices) < top_k
        ):

            best_index = None
            best_score = float("-inf")

            for index in remaining_indices:

                # Highest similarity to any selected chunk.
                redundancy = max(
                    similarity_matrix[index][selected_index]
                    for selected_index in selected_indices
                )

                # MMR balances relevance and diversity.
                mmr_score = (
                    lambda_param * relevance_scores[index]
                    - (1 - lambda_param) * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        return [chunks[index] for index in selected_indices]
