
from src.rag_pipeline import RAGPipeline


def main():
    rag = RAGPipeline()

    question = (
        "Compare ARO and SPARCL in terms of their "
        "approaches to improving compositional understanding."
    )

    sources = ["paper1.pdf", "paper2.pdf"]

    result = rag.compare_papers(
        question=question,
        sources=sources
    )

    print("\n=== GENERATED COMPARISON ===\n")
    print(result["answer"])

    print("\n=== SOURCES ===\n")

    for i, chunk in enumerate(result["sources"], start=1):
        print(
            f"[{i}] {chunk['source']} - "
            f"Page {chunk['page']}"
        )


if __name__ == "__main__":
    main()
