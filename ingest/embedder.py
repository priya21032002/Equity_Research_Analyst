import os
import chromadb
from chromadb.config import Settings
from ingest.pdf_loader import ParsedChunk
from typing import List
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store with Gemini embeddings."""

    COLLECTION_NAME = "equity_researcher"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir

        # Configure Gemini directly — no langchain_google_genai needed
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embed_model = "models/gemini-embedding-001"

        # Setup ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB ready. Collection size: {self.collection.count()}")
        logger.info(f"Using embedding model: {self.embed_model}")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using Gemini."""
        result = genai.embed_content(
            model=self.embed_model,
            content=texts,
            task_type="retrieval_document",
        )
        # embed_content returns {"embedding": [...]} for single
        # and {"embedding": [[...], [...]]} for batch
        embeddings = result["embedding"]

        # Normalize to always be list of lists
        if isinstance(embeddings[0], float):
            embeddings = [embeddings]

        return embeddings

    def _embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        result = genai.embed_content(
            model=self.embed_model,
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def add_chunks(self, chunks: List[ParsedChunk], batch_size: int = 20):
        """Embed and store chunks in ChromaDB."""
        logger.info(f"Embedding {len(chunks)} chunks...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
            batch = chunks[i: i + batch_size]
            texts = [c.content for c in batch]
            metadatas = [c.to_dict() for c in batch]
            ids = [
                f"{c.company_name}_{c.source_file}_{c.page_number}_{i+j}"
                for j, c in enumerate(batch)
            ]

            try:
                embeddings = self._embed_texts(texts)
                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )

            except Exception as e:
                logger.error(f"Batch {i} failed: {e} — trying one by one...")
                # Fallback: embed one at a time
                for j, (text, meta, uid) in enumerate(zip(texts, metadatas, ids)):
                    try:
                        emb = self._embed_query(text)
                        self.collection.add(
                            documents=[text],
                            embeddings=[emb],
                            metadatas=[meta],
                            ids=[uid],
                        )
                    except Exception as e2:
                        logger.warning(f"Skipping chunk {i+j}: {e2}")
                        continue

        logger.info("Embedding complete.")

    def get_companies(self) -> List[str]:
        """List all companies in the vector store."""
        results = self.collection.get(include=["metadatas"])
        companies = set()
        for m in results["metadatas"]:
            companies.add(m.get("company_name", "Unknown"))
        return sorted(list(companies))

    def collection_size(self) -> int:
        return self.collection.count()