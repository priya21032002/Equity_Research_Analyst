from typing import List
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest.pdf_loader import ParsedChunk

logger = logging.getLogger(__name__)


class FinancialChunker:
    """
    Chunks text while preserving financial context.
    Tables are NOT split — they're kept as atomic units.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, parsed_chunks: List[ParsedChunk]) -> List[ParsedChunk]:
        """Split text chunks further; keep tables and images atomic."""
        result = []

        for chunk in parsed_chunks:
            if chunk.content_type in ("table", "image_ocr"):
                result.append(chunk)
                continue

            if len(chunk.content) <= 800:
                result.append(chunk)
                continue

            splits = self.splitter.split_text(chunk.content)
            for i, split_text in enumerate(splits):
                new_chunk = ParsedChunk(
                    content=split_text,
                    content_type=chunk.content_type,
                    source_file=chunk.source_file,
                    company_name=chunk.company_name,
                    doc_type=chunk.doc_type,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    metadata={"split_index": i},
                )
                result.append(new_chunk)

        logger.info(f"Chunking: {len(parsed_chunks)} → {len(result)} chunks")
        return result