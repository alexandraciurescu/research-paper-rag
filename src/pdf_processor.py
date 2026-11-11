
import re
from pathlib import Path

import pymupdf


def clean_text(text):
    """
    Clean PDF text while preserving paragraph boundaries.
    """

    # Remove recurring conference headers.
    text = re.sub(
        r"(?im)^\s*Published as a conference paper at ICLR \d{4}\s*$",
        "",
        text
    )

    # Remove arXiv metadata lines.
    text = re.sub(
        r"(?im)^\s*arXiv:\S+.*$",
        "",
        text
    )

    # Remove standalone page numbers.
    text = re.sub(
        r"(?m)^\s*\d{1,3}\s*$",
        "",
        text
    )

    # Normalize spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Join words broken by hyphenation across lines.
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
        "",
        text
    )

    # Preserve paragraph breaks.
    paragraphs = re.split(r"\n\s*\n+", text)

    cleaned_paragraphs = []

    for paragraph in paragraphs:
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        paragraph = re.sub(r" +", " ", paragraph).strip()

        if paragraph:
            cleaned_paragraphs.append(paragraph)

    return "\n\n".join(cleaned_paragraphs)

def extract_text_from_pdf(pdf_path):
    """
    Extract PDF text while preserving page metadata.
    Detect the references section before text cleaning.
    """
    pages = []
    references_started = False

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):

            raw_text = page.get_text("text")

            # Detect a standalone References/Bibliography heading
            # before joining lines.
            match = re.search(
                r"(?im)^\s*(?:references|bibliography)\s*$",
                raw_text
            )

            if match:
                raw_text = raw_text[:match.start()]
                references_started = True

            text = clean_text(raw_text)

            if text:
                pages.append({
                    "text": text,
                    "page": page_number,
                    "source": Path(pdf_path).name
                })

            # Stop after preserving the content before References.
            if references_started:
                break

    return pages


def split_into_sentences(text):
    """
    Split text into sentences using basic punctuation rules.
    """

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text
        )
        if sentence.strip()
    ]


def split_long_text(text, max_words):
    """
    Split a long paragraph into sentence-based units.
    """

    sentences = split_into_sentences(text)

    units = []
    current = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()

        # Handle sentences longer than max_words.
        if len(words) > max_words:

            if current:
                units.append(" ".join(current))
                current = []
                current_words = 0

            # Fallback: split exceptionally long sentences by words.
            for start in range(0, len(words), max_words):
                units.append(
                    " ".join(words[start:start + max_words])
                )

            continue

        if current_words + len(words) > max_words:
            units.append(" ".join(current))
            current = []
            current_words = 0

        current.append(sentence)
        current_words += len(words)

    if current:
        units.append(" ".join(current))

    return units


def chunk_text(pages, chunk_size=250, overlap=30):
    """
    Create paragraph-aware chunks with sentence-based overlap.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size"
        )

    chunks = []
    chunk_id = 0

    for page in pages:

        paragraphs = [
            paragraph.strip()
            for paragraph in page["text"].split("\n\n")
            if paragraph.strip()
        ]

        # Convert paragraphs into manageable units.
        units = []

        for paragraph in paragraphs:
            if len(paragraph.split()) <= chunk_size:
                units.append(paragraph)
            else:
                units.extend(
                    split_long_text(paragraph, chunk_size)
                )

        current_units = []
        current_words = 0

        for unit in units:
            unit_words = len(unit.split())

            # Save the current chunk if the next unit exceeds the limit.
            if current_units and current_words + unit_words > chunk_size:

                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": " ".join(current_units),
                    "page": page["page"],
                    "source": page["source"]
                })

                chunk_id += 1

                # Keep a small amount of previous context.
                overlap_units = []
                overlap_words = 0

                for previous_unit in reversed(current_units):
                    previous_words = len(previous_unit.split())

                    if overlap_words + previous_words > overlap:
                        break

                    overlap_units.insert(0, previous_unit)
                    overlap_words += previous_words

                # Avoid exceeding chunk_size after overlap.
                while (
                    overlap_units
                    and overlap_words + unit_words > chunk_size
                ):
                    removed = overlap_units.pop(0)
                    overlap_words -= len(removed.split())

                current_units = overlap_units
                current_words = overlap_words

            current_units.append(unit)
            current_words += unit_words

        # Save the last chunk on the page.
        if current_units:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": " ".join(current_units),
                "page": page["page"],
                "source": page["source"]
            })

            chunk_id += 1

    return chunks


def remove_references(pages):
    """
    Remove the bibliography section from a research paper.
    """

    cleaned_pages = []
    references_started = False

    for page in pages:
        if references_started:
            break

        text = page["text"]

        match = re.search(
            r"(?im)(?:^|\n)\s*(?:references|bibliography)\s*(?:\n|$)",
            text
        )

        if match:
            text = text[:match.start()].strip()
            references_started = True

        if text:
            cleaned_pages.append({
                **page,
                "text": text
            })

    return cleaned_pages
