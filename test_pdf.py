
from src.pdf_processor import extract_text_from_pdf, chunk_text

# Extract text from the PDF.
pages = extract_text_from_pdf("data/paper1.pdf")

print("Pages extracted:", len(pages))

# Split the extracted text into chunks.
chunks = chunk_text(pages, chunk_size=200, overlap=40)

print("Chunks created:", len(chunks))

# Display the first chunk.
if chunks:
    print("\nFirst chunk:")
    print("Source:", chunks[0]["source"])
    print("Page:", chunks[0]["page"])
    print("Text:", chunks[0]["text"][:500])
else:
    print("No text was extracted from the PDF.")
