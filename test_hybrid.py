
from src.vector_store import VectorStore
from src.hybrid_retriever import HybridRetriever


vector_store = VectorStore()
hybrid = HybridRetriever(vector_store)

question = "How does NEG-CLIP use hard negatives?"

print("\nQUESTION:", question)

print("\n=== SEMANTIC SEARCH ===")

semantic_results = vector_store.search(
    question,
    top_k=5,
    sources=["paper1.pdf"]
)

for i, result in enumerate(semantic_results, 1):
    print(
        f"{i}. Page {result['page']} "
        f"| Distance: {result['distance']:.4f}"
    )
    print(result["text"][:200], "\n")


print("\n=== HYBRID SEARCH ===")

hybrid_results = hybrid.search(
    question,
    top_k=5,
    sources=["paper1.pdf"]
)

for i, result in enumerate(hybrid_results, 1):
    print(
        f"{i}. Page {result['page']} "
        f"| RRF score: {result['rrf_score']:.4f}"
    )
    print(result["text"][:200], "\n")
