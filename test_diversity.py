
from src.vector_store import VectorStore
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.diversity_selector import DiversitySelector


query = (
    "How does NEG-CLIP generate hard negative captions, "
    "and why are they useful?"
)

vector_store = VectorStore()
retriever = HybridRetriever(vector_store)
reranker = Reranker()

selector = DiversitySelector(vector_store.model)

# Retrieve 20 candidates using hybrid search.
candidates = retriever.search(
    query,
    top_k=20,
    sources=["paper1.pdf"]
)

# Rerank all 20 candidates.
reranked = reranker.rerank(
    query,
    candidates,
    top_k=20
)

# Select 5 relevant and diverse chunks.
diverse_chunks = selector.select(
    reranked,
    top_k=5,
    lambda_param=0.8
)


print("\n=== BEFORE MMR ===")

for i, chunk in enumerate(reranked[:5], start=1):
    print(f"\n{i}. {chunk['source']} - Page {chunk['page']}")
    print("Rerank score:", round(chunk["rerank_score"], 3))
    print("Text:", chunk["text"][:250])


print("\n=== AFTER MMR ===")

for i, chunk in enumerate(diverse_chunks, start=1):
    print(f"\n{i}. {chunk['source']} - Page {chunk['page']}")
    print("Rerank score:", round(chunk["rerank_score"], 3))
    print("Text:", chunk["text"][:250])
