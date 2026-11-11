
from src.pdf_processor import (
    extract_text_from_pdf,
    chunk_text,
    remove_references
)
from src.vector_store import VectorStore

# Step 1: Extract text and create chunks.
pages = extract_text_from_pdf("data/paper1.pdf")

# Remove bibliography
pages = remove_references(pages)

# Split cleaned text into chunks
chunks = chunk_text(pages, chunk_size=200, overlap=40)

print(f"Created {len(chunks)} chunks.")

# Step 2: Store their embeddings.
vector_store = VectorStore()
vector_store.add_chunks(chunks)

print("Embeddings stored successfully!")

# Step 3: Ask a question.
query = "Why do vision-language models struggle with word order?"

results = vector_store.search(query, top_k=3)

# Step 4: Display retrieved passages.
for i, result in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Distance:", round(result["distance"], 4))
    print("Text:", result["text"][:600])
