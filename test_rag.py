
from src.rag_pipeline import RAGPipeline


def main():
    # Initialize the RAG pipeline.
    rag = RAGPipeline()

    # Ask a question about the indexed paper.
    question = (
        "Why do vision-language models struggle "
        "with word order?"
    )

    print("\nQuestion:", question)
    print("\nGenerating answer...\n")

    # Retrieve context and generate the answer.
    result = rag.answer_question(question, top_k=3)

    print("=== AI ANSWER ===")
    print(result["answer"])

    print("\n=== RETRIEVED SOURCES ===")

    for i, source in enumerate(result["sources"], start=1):
        print(f"\n[{i}] {source['source']}, page {source['page']}")
        if "rerank_score" in source:
            print("Rerank score:", round(source["rerank_score"], 4))
        print("Excerpt:", source["text"][:300])
        if "distance" in source:
            print("Semantic distance:", round(source["distance"], 4))

        if "rrf_score" in source:
            print("RRF score:", round(source["rrf_score"], 4))

        if "rerank_score" in source:
            print("Rerank score:", round(source["rerank_score"], 4))


if __name__ == "__main__":
    main()
