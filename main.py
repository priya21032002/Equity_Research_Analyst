"""
Equity Researcher AI — Main Streamlit App
"""

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from ingest.pdf_loader import ScreenerPDFLoader
from ingest.chunker import FinancialChunker
from ingest.embedder import VectorStoreManager
from agents.graph import run_query

load_dotenv()

st.set_page_config(
    page_title="Equity Researcher AI",
    page_icon="📊",
    layout="wide",
)

# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    with st.spinner("Loading vector store..."):
        st.session_state.vector_store = VectorStoreManager()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Document Ingestion")
    st.caption("Upload PDFs from Screener.in")

    company_name = st.text_input(
        "Company Name",
        placeholder="e.g. Infosys, HDFC Bank, Reliance",
    )

    doc_type = st.selectbox(
        "Document Type",
        ["annual_report", "credit_rating", "concall"],
        format_func=lambda x: {
            "annual_report": "Annual Report",
            "credit_rating": "Credit Rating",
            "concall": "Concall Transcript",
        }[x],
    )

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # ── Ingestion Settings ────────────────────────────────────────────────────
    with st.expander("Advanced Settings"):
        skip_pages = st.number_input(
            "Skip first N pages (cover/TOC)",
            min_value=0,
            max_value=20,
            value=5,
            help="Annual reports usually have 5-10 pages of cover/TOC to skip",
        )
        use_page_limit = st.checkbox("Limit pages (for testing)", value=False)
        max_pages = None
        if use_page_limit:
            max_pages = st.number_input(
                "Max pages to process",
                min_value=10,
                max_value=500,
                value=50,
            )

    ingest_clicked = st.button(
        "Ingest Documents",
        disabled=not (company_name and uploaded_files),
        use_container_width=True,
    )

    if ingest_clicked:
        all_chunks = []
        loader = ScreenerPDFLoader(company_name=company_name, doc_type=doc_type)
        chunker = FinancialChunker()

        for uploaded_file in uploaded_files:
            st.markdown(f"**Processing:** `{uploaded_file.name}`")

            # Save upload to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Progress bar for parsing
            progress_bar = st.progress(0, text="Starting PDF parse...")
            status_text = st.empty()

            def update_progress(current, total):
                if total > 0:
                    pct = int((current / total) * 100)
                    progress_bar.progress(
                        min(pct, 100),
                        text=f"Parsing page {current} of {total}..."
                    )

            try:
                # Parse PDF
                parsed = loader.load(
                    pdf_path=tmp_path,
                    max_pages=max_pages,
                    skip_pages=int(skip_pages),
                    progress_callback=update_progress,
                )
                progress_bar.progress(100, text="Parsing complete!")

                if not parsed:
                    st.warning(f"No content extracted from {uploaded_file.name}. Try reducing 'skip pages'.")
                    continue

                # Chunk
                status_text.text("Chunking content...")
                chunked = chunker.chunk(parsed)
                all_chunks.extend(chunked)

                status_text.empty()
                st.success(
                    f"`{uploaded_file.name}` → "
                    f"{len(parsed)} raw blocks → "
                    f"{len(chunked)} chunks"
                )

            except Exception as e:
                st.error(f"Failed on `{uploaded_file.name}`: {e}")
                import traceback
                st.code(traceback.format_exc(), language="python")

            finally:
                os.unlink(tmp_path)

        # Embed all chunks
        if all_chunks:
            with st.spinner(f"Embedding {len(all_chunks)} chunks into ChromaDB..."):
                try:
                    st.session_state.vector_store.add_chunks(all_chunks)
                    st.success(
                        f"Stored **{len(all_chunks)} chunks** for **{company_name}** "
                        f"in vector DB."
                    )
                    st.balloons()
                except Exception as e:
                    st.error(f"Embedding failed: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
        else:
            st.warning("No chunks to embed. Check your PDF or increase page limit.")

    # ── Indexed Companies ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Indexed Companies")
    try:
        companies = st.session_state.vector_store.get_companies()
        if companies:
            for c in companies:
                st.markdown(f"• **{c}**")
        else:
            st.caption("No companies indexed yet.")
    except Exception:
        st.caption("Loading...")

    st.divider()
    try:
        db_size = st.session_state.vector_store.collection_size()
        st.caption(f"🗄️ Vector DB: **{db_size}** chunks total")
    except Exception:
        st.caption("Vector DB: loading...")


# ── Main Chat Interface ───────────────────────────────────────────────────────
st.title("Equity Researcher AI")
st.caption("Gemini + LangGraph + ChromaDB · Citations included · No hallucinations")

# Company selector
try:
    companies = st.session_state.vector_store.get_companies()
except Exception:
    companies = []

if companies:
    selected_company = st.selectbox(
        "Select Company to Research",
        companies,
        key="company_selector",
    )
else:
    selected_company = None
    st.info("Upload and ingest a PDF in the sidebar to get started.")

st.divider()

# ── Chat History ──────────────────────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("citations"):
            with st.expander(f"{len(message['citations'])} Source Citations"):
                for cite in message["citations"]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f"**[{cite['id']}]** `{cite['source_file']}` · "
                            f"Page **{cite['page_number']}** · "
                            f"*{cite['section_title']}*"
                        )
                        st.caption(f"> {cite['content_preview']}")
                    with col2:
                        st.metric("Relevance", f"{cite['relevance_score']:.2f}")
                        st.caption(f"`{cite['doc_type']}`")
                    st.divider()

        if message.get("used_calculator"):
            st.info("Financial Calculator Agent was used.")

# ── Chat Input ────────────────────────────────────────────────────────────────
query = st.chat_input(
    placeholder="Ask about revenue, margins, management commentary, risks...",
    disabled=not selected_company,
)

if query:
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agents and show response
    with st.chat_message("assistant"):
        with st.spinner(f"Researching {selected_company}..."):
            try:
                result = run_query(query=query, company_name=selected_company)
                answer = result["answer"]
                citations = result["citations"]
                used_calculator = result.get("used_calculator", False)

                st.markdown(answer)

                if citations:
                    with st.expander(f"{len(citations)} Source Citations"):
                        for cite in citations:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(
                                    f"**[{cite['id']}]** `{cite['source_file']}` · "
                                    f"Page **{cite['page_number']}** · "
                                    f"*{cite['section_title']}*"
                                )
                                st.caption(f"> {cite['content_preview']}")
                            with col2:
                                st.metric("Relevance", f"{cite['relevance_score']:.2f}")
                                st.caption(f"`{cite['doc_type']}`")
                            st.divider()

                if used_calculator:
                    st.info("Financial Calculator Agent was used for this query.")

            except Exception as e:
                answer = f"Error: {str(e)}"
                citations = []
                used_calculator = False

                st.error(answer)
                import traceback
                st.code(traceback.format_exc(), language="python")

    # Save to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "used_calculator": used_calculator,
    })