
from src.pdf_processor import (
    extract_text_from_pdf,
    remove_references,
    chunk_text
)


pdf_path = "data/paper1.pdf"

pages = extract_text_from_pdf(pdf_path)
cleaned_pages = remove_references(pages)

chunks = chunk_text(cleaned_pages)

print("Original pages:", len(pages))
print("Pages after removing references:", len(cleaned_pages))
print("Total chunks:", len(chunks))

print("\n=== FIRST 5 CHUNKS ===")

for i, chunk in enumerate(chunks[:5], start=1):
    print(f"\nCHUNK {i}")
    print("Page:", chunk["page"])
    print("Words:", len(chunk["text"].split()))
    print("Text:", chunk["text"][:500])


print("\n=== CHUNKS ON PAGE 7 ===")

page_7_chunks = [
    chunk for chunk in chunks
    if chunk["page"] == 7
]

for i, chunk in enumerate(page_7_chunks, start=1):
    print(f"\nCHUNK {i}")
    print("Words:", len(chunk["text"].split()))
    print("Beginning:", chunk["text"][:200])
    print("Ending:", chunk["text"][-200:])
