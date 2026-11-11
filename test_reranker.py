
from src.vector_store import VectorStore
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker


vector_store = VectorStore()
hybrid = HybridRetriever(vector_store)
reranker = Reranker()

question = "How does NEG-CLIP use hard negatives?"

# Retrieve 20 candidates using Hybrid Search.
candidates = hybrid.search(
    question,
    top_k=20,
    sources=["paper1.pdf"]
)

print("\n=== HYBRID SEARCH TOP 5 ===")

for i, chunk in enumerate(candidates[:5], 1):
    print(
        f"{i}. Page {chunk['page']} "
        f"| RRF: {chunk['rrf_score']:.4f}"
    )
    print(chunk["text"][:250], "\n")


# Rerank all 20 candidates.
reranked = reranker.rerank(
    question,
    candidates,
    top_k=5
)

print("\n=== RERANKED TOP 5 ===")

for i, chunk in enumerate(reranked, 1):
    print(
        f"{i}. Page {chunk['page']} "
        f"| Score: {chunk['rerank_score']:.4f}"
    )
    print(chunk["text"][:250], "\n")
