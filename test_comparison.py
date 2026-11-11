
from src.rag_pipeline import RAGPipeline


def main():
    rag = RAGPipeline()

    
    question = (
    "Compare ARO and SPARCL in terms of their "
    "approaches to improving compositional understanding."
)

    sources = ["paper1.pdf", "paper2.pdf"]

    chunks = rag.retrieve_for_comparison(
        question,
        sources=sources,
        per_document=3
    )

    print("\n=== COMPARISON RETRIEVAL ===")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n[{i}] {chunk['source']} - Page {chunk['page']}")
        print("Rerank score:", round(chunk["rerank_score"], 3))
        print("Text:", chunk["text"][:400])

    print("\n=== DOCUMENT COVERAGE ===")

    for source in sources:
        count = sum(
            chunk["source"] == source
            for chunk in chunks
        )
        print(f"{source}: {count} chunks")


if __name__ == "__main__":
    main()
