import os
from typing import List, Dict, Tuple
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class CitedRetriever:
    """
    Retrieves relevant chunks from ChromaDB and formats them with citations.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        # Configure Gemini directly — no langchain_google_genai needed
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embed_model = "models/gemini-embedding-001"

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("equity_researcher")
        logger.info(f"Retriever ready. Using model: {self.embed_model}")

    def _embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        result = genai.embed_content(
            model=self.embed_model,
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def retrieve(
        self,
        query: str,
        company_name: str,
        top_k: int = 8,
        doc_type_filter: str = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant chunks and return:
        - context_str: formatted string for LLM
        - citations: list of source dicts for UI display
        """

        # Build filter
        where_filter = {"company_name": {"$eq": company_name}}
        if doc_type_filter:
            where_filter["doc_type"] = {"$eq": doc_type_filter}

        # Embed query
        query_embedding = self._embed_query(query)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return "No relevant information found.", []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        # Build context with inline citation markers
        context_parts = []
        citations = []

        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            cite_id = idx + 1
            relevance = round(1 - dist, 3)  # cosine similarity

            context_parts.append(
                f"[SOURCE {cite_id}] "
                f"(File: {meta.get('source_file', 'N/A')}, "
                f"Page: {meta.get('page_number', 'N/A')}, "
                f"Section: {meta.get('section_title', 'N/A')})\n"
                f"{doc}\n"
            )

            citations.append({
                "id": cite_id,
                "source_file": meta.get("source_file", "N/A"),
                "page_number": meta.get("page_number", "N/A"),
                "section_title": meta.get("section_title", "N/A"),
                "doc_type": meta.get("doc_type", "N/A"),
                "company_name": meta.get("company_name", "N/A"),
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "relevance_score": relevance,
            })

        context_str = "\n---\n".join(context_parts)
        return context_str, citations