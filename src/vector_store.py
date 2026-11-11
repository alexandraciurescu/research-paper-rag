
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, db_path="chroma_db"):
        # Load the embedding model.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Connect to the persistent vector database.
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks):
        if not chunks:
            return

        # Group chunks by their source document.
        sources = set(chunk["source"] for chunk in chunks)

        # Remove old chunks for each document before reindexing.
        for source in sources:
            self.collection.delete(
                where={"source": source}
            )

        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings locally.
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        # Generate deterministic IDs for each chunk.
        ids = []

        for i, chunk in enumerate(chunks):
            identifier = (
                f"{chunk['source']}:{chunk['page']}:{i}"
            )

            chunk_id = hashlib.sha256(
                identifier.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

        # Store embeddings and metadata.
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
                for chunk in chunks
            ]
        )

    def list_documents(self):
        # Get stored metadata.
        results = self.collection.get(
            include=["metadatas"]
        )

        return sorted({
            metadata["source"]
            for metadata in results["metadatas"]
        })

    def search(self, query, top_k=3, sources=None):
        # Convert the question into an embedding.
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # Filter by selected documents, if provided.
        where_filter = None

        if sources:
            where_filter = {
                "source": {"$in": sources}
            }

        # Search the vector database.
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        return [
            {
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": distance
            }
            for text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
