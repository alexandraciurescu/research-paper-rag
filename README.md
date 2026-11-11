# 📚 AI Research Paper Assistant

A Retrieval-Augmented Generation (RAG) application for exploring, querying, and comparing scientific research papers.

The application combines semantic search, keyword-based retrieval, reranking, and large language models to generate answers grounded in research paper excerpts, with source citations.

## Features

- **Multi-PDF Processing:** Upload and process multiple research papers.
- **Text Extraction & Chunking:** Extract text from PDFs, clean the content, and split it into manageable chunks.
- **Semantic Search:** Retrieve relevant passages using Sentence Transformers and ChromaDB.
- **Hybrid Retrieval:** Combine semantic search with BM25 keyword retrieval using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking:** Improve passage relevance using a Cross-Encoder model.
- **Diversity-Aware Retrieval:** Apply Maximal Marginal Relevance (MMR) to reduce redundancy.
- **Grounded Question Answering:** Generate answers using GPT-4.1-mini with citations to retrieved excerpts.
- **Multi-Paper Comparison:** Compare research papers by retrieving evidence separately for different aspects of each document.
- **Streamlit Interface:** Upload documents, ask questions, compare papers, and inspect supporting sources.

## Architecture

1. **Document ingestion:** PDF extraction, cleaning, and chunking.
2. **Indexing:** Generate embeddings and store them in ChromaDB.
3. **Retrieval:** Combine semantic search and BM25 using Reciprocal Rank Fusion.
4. **Reranking:** Rank candidate passages using a Cross-Encoder.
5. **Context selection:** Apply MMR for question answering or aspect-based retrieval for paper comparison.
6. **Answer generation:** Generate responses with GPT-4.1-mini using retrieved evidence.
7. **Source attribution:** Display the document names, page numbers, and supporting excerpts.

## Tech Stack

**Python · Streamlit · OpenAI API · ChromaDB · Sentence Transformers · BM25 · PyMuPDF**

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/research-paper-rag.git
cd research-paper-rag
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
```

Launch the application:

```bash
python -m streamlit run app.py
```

## Usage

**Ask Questions**

Upload research papers, select the documents you want to search, and ask questions about their content. The application retrieves relevant excerpts and generates answers with citations.

**Compare Papers**

Select at least two papers and ask a comparative question. The application retrieves evidence from each paper and generates a structured comparison of their methodologies, similarities, and differences.

## Current Limitations

- PDF text extraction may be less reliable for complex layouts and mathematical notation.
- Generated citations are not independently verified against every claim.
- Retrieval evaluation using a manually annotated benchmark is planned.

## Future Improvements

- Quantitative retrieval evaluation using Recall@K and MRR.
- Automated testing and retrieval ablation studies.
- Improved citation verification and context selection.
- Enhanced PDF processing for complex scientific documents.

## Project Motivation

This project was developed to explore practical techniques for building reliable RAG systems for scientific literature, with an emphasis on retrieval quality, source attribution, and comparing information across multiple documents.