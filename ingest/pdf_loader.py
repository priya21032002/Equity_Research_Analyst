import fitz  
import os
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedChunk:
    content: str
    content_type: str
    source_file: str
    company_name: str
    doc_type: str
    page_number: int
    section_title: Optional[str] = None
    table_data: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "content_type": self.content_type,
            "source_file": self.source_file,
            "company_name": self.company_name,
            "doc_type": self.doc_type,
            "page_number": self.page_number,
            "section_title": self.section_title or "",
            "table_data": self.table_data or "",
        }


# Sections we care about in annual reports — skip boilerplate
IMPORTANT_SECTIONS = {
    "management discussion", "financial statements", "balance sheet",
    "profit and loss", "cash flow", "notes to accounts", "directors report",
    "auditors report", "key financial", "standalone", "consolidated",
    "revenue", "ebitda", "segment", "risk", "outlook", "concall",
    "credit rating", "rationale", "debt", "borrowing",
}

SKIP_SECTIONS = {
    "notice", "attendance slip", "proxy form", "agm notice",
    "map", "route map", "green initiative",
}


class ScreenerPDFLoader:
    """
    Fast PDF loader for large Screener.in documents.
    Uses only PyMuPDF — no Ghostscript, no pdfplumber table scan.
    Processes a 300-page annual report in ~15-30 seconds.
    """

    def __init__(self, company_name: str, doc_type: str):
        self.company_name = company_name
        self.doc_type = doc_type

    def load(
        self,
        pdf_path: str,
        max_pages: int = None,
        skip_pages: int = 5,       # skip first N pages (cover, TOC)
        progress_callback=None,    # optional: fn(current, total)
    ) -> List[ParsedChunk]:

        pdf_path = str(pdf_path)
        logger.info(f"Opening: {os.path.basename(pdf_path)}")

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        end_page = min(total_pages, max_pages + skip_pages) if max_pages else total_pages

        logger.info(f"Total pages: {total_pages} | Processing: {skip_pages} → {end_page}")

        chunks = []
        current_section = "General"
        skip_current_section = False

        for page_num in range(skip_pages, end_page):
            if progress_callback:
                progress_callback(page_num - skip_pages, end_page - skip_pages)

            try:
                page = doc[page_num]
                page_chunks, current_section, skip_current_section = self._process_page(
                    page=page,
                    page_num=page_num + 1,
                    pdf_path=pdf_path,
                    current_section=current_section,
                    skip_current_section=skip_current_section,
                )
                chunks.extend(page_chunks)

            except Exception as e:
                logger.warning(f"Page {page_num + 1} failed: {e} — skipping")
                continue

            # Log progress every 50 pages
            if (page_num - skip_pages) % 50 == 0 and page_num > skip_pages:
                logger.info(f"  Processed {page_num - skip_pages}/{end_page - skip_pages} pages, {len(chunks)} chunks so far")

        doc.close()
        logger.info(f"Done: {len(chunks)} chunks from {os.path.basename(pdf_path)}")
        return chunks

    # ── Page Processor ───────────────────────────────────────────────────────

    def _process_page(
        self,
        page,
        page_num: int,
        pdf_path: str,
        current_section: str,
        skip_current_section: bool,
    ):
        chunks = []

        # Get structured text blocks with font info
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        page_text_lines = []   # (text, font_size, is_bold)
        table_candidates = []  # lines that look like table rows

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = ""
                font_size = 12
                is_bold = False
                for span in line.get("spans", []):
                    t = span.get("text", "").strip()
                    if t:
                        line_text += t + " "
                        font_size = span.get("size", 12)
                        is_bold = "bold" in span.get("font", "").lower()
                line_text = line_text.strip()
                if line_text:
                    page_text_lines.append((line_text, font_size, is_bold))

        if not page_text_lines:
            return chunks, current_section, skip_current_section

        # ── Section Detection ────────────────────────────────────────────────
        para_buffer = []

        for line_text, font_size, is_bold in page_text_lines:

            # Detect heading: large font OR bold + short line
            is_heading = (
                (font_size >= 13 or is_bold)
                and len(line_text) < 120
                and not self._looks_like_number(line_text)
            )

            if is_heading:
                # Flush current buffer first
                if para_buffer and not skip_current_section:
                    flushed = self._flush_buffer(
                        para_buffer, page_num, pdf_path, current_section
                    )
                    chunks.extend(flushed)
                para_buffer = []

                # Update section
                new_section = line_text.strip()
                current_section = new_section

                # Check if this is a section we should skip
                section_lower = new_section.lower()
                skip_current_section = any(
                    s in section_lower for s in SKIP_SECTIONS
                )
                continue

            if skip_current_section:
                continue

            # ── Table Row Detection ──────────────────────────────────────────
            if self._looks_like_table_row(line_text):
                # Flush text buffer before starting table
                if para_buffer:
                    chunks.extend(self._flush_buffer(
                        para_buffer, page_num, pdf_path, current_section
                    ))
                    para_buffer = []
                table_candidates.append(line_text)
                continue

            # If we had table rows building up, flush them
            if table_candidates:
                table_chunk = self._make_table_chunk(
                    table_candidates, page_num, pdf_path, current_section
                )
                if table_chunk:
                    chunks.append(table_chunk)
                table_candidates = []

            # Normal text line
            para_buffer.append(line_text)

        # Final flushes
        if para_buffer and not skip_current_section:
            chunks.extend(self._flush_buffer(
                para_buffer, page_num, pdf_path, current_section
            ))
        if table_candidates:
            table_chunk = self._make_table_chunk(
                table_candidates, page_num, pdf_path, current_section
            )
            if table_chunk:
                chunks.append(table_chunk)

        return chunks, current_section, skip_current_section

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _flush_buffer(
        self, lines: List[str], page_num: int, pdf_path: str, section: str
    ) -> List["ParsedChunk"]:
        text = " ".join(lines).strip()
        if len(text) < 40:
            return []
        return [ParsedChunk(
            content=text,
            content_type="text",
            source_file=os.path.basename(pdf_path),
            company_name=self.company_name,
            doc_type=self.doc_type,
            page_number=page_num,
            section_title=section,
        )]

    def _make_table_chunk(
        self, rows: List[str], page_num: int, pdf_path: str, section: str
    ) -> Optional["ParsedChunk"]:
        if len(rows) < 2:
            return None

        # Join rows into readable NL string
        nl = f"Financial data from '{section}' on page {page_num}: " + " | ".join(rows[:20])

        return ParsedChunk(
            content=nl,
            content_type="table",
            source_file=os.path.basename(pdf_path),
            company_name=self.company_name,
            doc_type=self.doc_type,
            page_number=page_num,
            section_title=section,
            table_data="\n".join(rows),
        )

    def _looks_like_table_row(self, text: str) -> bool:
        """Detect lines that look like financial table rows."""
        # Has multiple numbers separated by spaces
        numbers = re.findall(r"\b[\d,]+\.?\d*\b", text)
        if len(numbers) >= 2:
            return True
        # Has currency markers
        if re.search(r"[₹$]\s*[\d,]+", text):
            return True
        # Looks like key-value: "Revenue 12,345"
        if re.match(r"^[A-Za-z\s/\(\)]+\s+[\d,]+", text):
            return True
        return False

    def _looks_like_number(self, text: str) -> bool:
        """Check if line is just a page number or numeric label."""
        return bool(re.match(r"^\d+$", text.strip()))