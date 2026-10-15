import os
import shutil
import time
from datetime import datetime
from time import perf_counter

import streamlit as st
from utils.export_chat import (
    export_chat_txt,
    export_chat_markdown,
    export_chat_json,
)
from utils.pdf_reader import extract_documents_from_pdfs
from utils.text_splitter import split_documents
from utils.embeddings import get_embedding_model
from utils.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    load_documents_for_hybrid,
)
from utils.hybrid_search import HybridRetriever
from utils.llm import get_llm
from utils.rag_pipeline import stream_ask_question

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Parchment.ai | Multi-PDF RAG",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    if os.path.exists("assets/styles.css"):
        with open("assets/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# ==================================================
# Session State Initialization
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = None

if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []

if "available_docs" not in st.session_state:
    st.session_state.available_docs = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ==================================================
# Load Cached Models
# ==================================================
@st.cache_resource(show_spinner=False)
def load_cached_embedding_model():
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def load_cached_llm():
    return get_llm()


try:
    embedding_model = load_cached_embedding_model()
    llm = load_cached_llm()
except Exception as e:
    st.error("⚠️ Initialization Error: Missing or invalid GOOGLE_API_KEY. Please verify your `.env` configuration.")
    st.stop()

# ==================================================
# Load Existing Vector Database on Startup
# ==================================================
if st.session_state.vector_store is None:
    try:
        vs = load_vector_store(embedding_model)
        if vs is not None:
            docs = load_documents_for_hybrid(vs)
            st.session_state.vector_store = vs
            st.session_state.indexed_documents = docs
            st.session_state.hybrid_retriever = HybridRetriever(docs, vs)
            st.session_state.available_docs = sorted(
                list({d.metadata.get("source", "Document") for d in docs if d.metadata})
            )
    except Exception:
        pass

# ==================================================
# Sidebar: Document Management & Hybrid Search Controls
# ==================================================
with st.sidebar:
    st.markdown("## 📜 Parchment.ai")
    st.caption("Advanced Multi-Document RAG with Hybrid RRF Search")
    st.divider()

    # ----------------------------------------------
    # 1. Document Upload & Ingestion
    # ----------------------------------------------
    st.subheader("📚 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF research papers, reports, or legal documents.",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_box = st.empty()

            try:
                # Step 1: Extract Text
                status_box.info("📄 Reading uploaded PDF pages...")
                progress_bar.progress(15)
                documents = extract_documents_from_pdfs(uploaded_files)

                if not documents:
                    st.error("❌ No readable text found in the uploaded PDFs.")
                    st.stop()

                # Step 2: Split into Chunks
                status_box.info("✂️ Splitting documents into overlapping chunks...")
                progress_bar.progress(35)
                chunks = split_documents(documents)

                if not chunks:
                    st.error("❌ No document chunks generated.")
                    st.stop()

                # Step 3: Embed & Build Vector Store
                status_box.info("🧠 Generating Gemini embeddings & FAISS index...")
                progress_bar.progress(65)
                vs = create_vector_store(chunks, embedding_model)

                # Step 4: Build BM25 Hybrid Retriever
                status_box.info("🔍 Constructing BM25 sparse index & fusion engine...")
                progress_bar.progress(90)
                retriever = HybridRetriever(chunks, vs)

                # Save into state
                st.session_state.vector_store = vs
                st.session_state.indexed_documents = chunks
                st.session_state.hybrid_retriever = retriever
                st.session_state.available_docs = sorted(
                    list({c.metadata.get("source", "Document") for c in chunks if c.metadata})
                )

                progress_bar.progress(100)
                status_box.success("✅ Knowledge base built successfully!")
                time.sleep(0.8)
                st.rerun()

            except Exception as ex:
                st.error("❌ Failed to process documents.")
                st.exception(ex)

    st.divider()

    # ----------------------------------------------
    # 2. Retrieval Engine Configuration
    # ----------------------------------------------
    st.subheader("⚙️ Retrieval Settings")

    alpha_weight = st.slider(
        "Search Hybrid Balance (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="0.0 = BM25 Keyword Search Only | 0.5 = Balanced Hybrid RRF | 1.0 = FAISS Semantic Search Only",
    )

    if alpha_weight == 1.0:
        mode_label = "Dense Semantic Only (FAISS)"
    elif alpha_weight == 0.0:
        mode_label = "Sparse Keyword Only (BM25)"
    else:
        mode_label = f"Hybrid RRF ({int(alpha_weight*100)}% Dense / {int((1-alpha_weight)*100)}% BM25)"

    st.caption(f"🎯 **Active Engine:** {mode_label}")

    top_k_chunks = st.slider(
        "Retrieved Context Chunks (Top-K)",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Number of most relevant chunks passed to Gemini for answer synthesis.",
    )

    doc_filter_options = ["All Documents"] + st.session_state.available_docs
    selected_source_filter = st.selectbox(
        "Filter By Document",
        options=doc_filter_options,
        index=0,
        help="Direct queries to a specific uploaded document or search across all.",
    )

    st.divider()

    # ----------------------------------------------
    # 3. Knowledge Base Status
    # ----------------------------------------------
    st.subheader("📊 Knowledge Base Status")
    if st.session_state.vector_store is not None:
        st.success("✅ Knowledge Base Active")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Documents", len(st.session_state.available_docs))
        with col_s2:
            st.metric("Chunks", len(st.session_state.indexed_documents))

        if st.session_state.available_docs:
            with st.expander("📑 View Loaded Documents"):
                for doc_name in st.session_state.available_docs:
                    st.write(f"• `{doc_name}`")
    else:
        st.warning("⚠️ No documents indexed yet. Upload a PDF above.")

    st.divider()

    # ----------------------------------------------
    # 4. Chat Management & Export
    # ----------------------------------------------
    st.subheader("💬 Chat Options")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_c2:
        if st.button("🔄 Reset Index", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.hybrid_retriever = None
            st.session_state.indexed_documents = []
            st.session_state.available_docs = []
            st.session_state.messages = []
            if os.path.exists("vector_db"):
                shutil.rmtree("vector_db")
            st.success("Database reset.")
            st.rerun()

    if st.session_state.messages:
        st.markdown("**Export Conversation:**")
        txt_export = export_chat_txt(st.session_state.messages)
        md_export = export_chat_markdown(st.session_state.messages)
        json_export = export_chat_json(st.session_state.messages)

        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            st.download_button(
                "📄 TXT",
                data=txt_export,
                file_name="parchment_chat.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c_exp2:
            st.download_button(
                "📝 MD",
                data=md_export,
                file_name="parchment_chat.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with c_exp3:
            st.download_button(
                "💾 JSON",
                data=json_export,
                file_name="parchment_chat.json",
                mime="application/json",
                use_container_width=True,
            )

# ==================================================
# Main Content Area
# ==================================================
st.markdown(
    """
    <div class="parchment-header">
        <span class="version-pill">✨ Parchment.ai • Gemini 2.5 Flash • Hybrid RRF</span>
        <h1 class="header-title">Parchment.ai</h1>
        <p class="header-subtitle">
            Enterprise-grade Multi-Document RAG with dense FAISS semantic vectors, sparse BM25 keyword matching, 
            Reciprocal Rank Fusion, and citation-grounded real-time streaming.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metric Summary Row
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric(
        "Knowledge Base",
        "Active" if st.session_state.vector_store is not None else "Empty",
    )
with m_col2:
    st.metric("Documents", len(st.session_state.available_docs))
with m_col3:
    st.metric("Total Chunks", len(st.session_state.indexed_documents))
with m_col4:
    st.metric("Messages", len(st.session_state.messages))

st.divider()

# ==================================================
# Prompt Suggestions & Empty State
# ==================================================
if st.session_state.vector_store is None:
    st.info(
        """
        ### 👋 Welcome to Parchment.ai!
        
        To get started:
        1. Open the sidebar and **upload one or more PDF files** (e.g. research papers, technical specs, reports).
        2. Click **Process & Index Documents** to generate embeddings and build the Hybrid RAG engine.
        3. Ask questions with **real-time token streaming** and verify answers with **grounded source citations**.
        """
    )
elif len(st.session_state.messages) == 0:
    st.markdown("#### 💡 Quick Starters")
    st.caption("Click any sample query to start analyzing your uploaded documents:")

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        if st.button("📌 Summarize Key Findings", use_container_width=True):
            st.session_state.pending_prompt = "Summarize the primary objectives, methodologies, and key findings of the documents."
            st.rerun()
    with s_col2:
        if st.button("🔍 Detail Main Challenges", use_container_width=True):
            st.session_state.pending_prompt = "What are the main challenges, limitations, or problems highlighted in these documents?"
            st.rerun()
    with s_col3:
        if st.button("📊 Extract Key Data & Results", use_container_width=True):
            st.session_state.pending_prompt = "Extract and synthesize any quantitative results, statistics, and conclusions mentioned."
            st.rerun()

# ==================================================
# Display Chat History
# ==================================================
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    latency = msg.get("latency")
    docs = msg.get("docs", [])

    with st.chat_message(role):
        st.markdown(content)

        # Show citation chips & expander for assistant messages
        if role == "assistant" and sources:
            # Inline Citation Badges
            chip_html = "<div style='margin-top: 8px; margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 6px;'>"
            for s in sources:
                conf = s.get("confidence", "N/A")
                mode = s.get("mode", "Hybrid")
                chip_html += f"""
                    <span class="citation-chip">
                        📄 <b>{s.get('source')}</b> (p. {s.get('page')}) 
                        <span class="match-badge-high">{conf}% {mode}</span>
                    </span>
                """
            chip_html += "</div>"
            st.markdown(chip_html, unsafe_allow_html=True)

            # Retrieved Chunks Details
            if docs:
                with st.expander(f"📚 Inspect Retrieved Sources ({len(docs)} Chunks)", expanded=False):
                    for idx, doc in enumerate(docs, start=1):
                        meta = doc.metadata or {}
                        score = meta.get("confidence_score", "N/A")
                        mode = meta.get("retrieval_mode", "Hybrid")
                        source = meta.get("source", "Unknown")
                        page = meta.get("page", "?")
                        chunk = meta.get("chunk", idx)

                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-card-header">
                                    <span class="source-card-title">Chunk #{idx} • 📄 {source} (Page {page}, Part {chunk})</span>
                                    <span class="match-badge-high">Relevance: {score}% [{mode}]</span>
                                </div>
                                <div class="source-card-snippet">{doc.page_content.strip()}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            if latency is not None:
                st.caption(f"⚡ *Synthesized in {latency:.2f}s using Gemini 2.5 Flash*")

# ==================================================
# Chat Input & Real-Time Token Streaming
# ==================================================
user_prompt = st.chat_input("Ask a question about your documents...")

# Support click from starter prompt buttons
if st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_prompt:
    if st.session_state.hybrid_retriever is None:
        st.warning("⚠️ Please upload and index documents before asking questions.")
    else:
        # Append User Message
        timestamp_now = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": user_prompt,
            "timestamp": timestamp_now,
        })

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate Assistant Response with Real-Time Streaming
        with st.chat_message("assistant"):
            try:
                start_clock = perf_counter()

                # Stream token generator & retrieve docs
                stream_generator, retrieved_docs = stream_ask_question(
                    question=user_prompt,
                    retriever=st.session_state.hybrid_retriever,
                    llm=llm,
                    messages=st.session_state.messages[:-1],  # pass prior conversation
                    k=top_k_chunks,
                    alpha=alpha_weight,
                    source_filter=selected_source_filter,
                )

                # Real token streaming into Streamlit
                full_response = st.write_stream(stream_generator)

                end_clock = perf_counter()
                latency_val = round(end_clock - start_clock, 2)

                # Prepare source metadata
                cited_sources = []
                seen_sources = set()
                for doc in retrieved_docs:
                    m = doc.metadata or {}
                    key = (m.get("source"), m.get("page"))
                    if key not in seen_sources:
                        seen_sources.add(key)
                        cited_sources.append({
                            "source": m.get("source", "Document"),
                            "page": m.get("page", "?"),
                            "chunk": m.get("chunk", "?"),
                            "confidence": m.get("confidence_score", "N/A"),
                            "mode": m.get("retrieval_mode", "Hybrid"),
                        })

                # Display Citation Badges
                if cited_sources:
                    chip_html = "<div style='margin-top: 8px; margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 6px;'>"
                    for s in cited_sources:
                        conf = s.get("confidence", "N/A")
                        mode = s.get("mode", "Hybrid")
                        chip_html += f"""
                            <span class="citation-chip">
                                📄 <b>{s.get('source')}</b> (p. {s.get('page')}) 
                                <span class="match-badge-high">{conf}% {mode}</span>
                            </span>
                        """
                    chip_html += "</div>"
                    st.markdown(chip_html, unsafe_allow_html=True)

                # Display Inspector Expander
                if retrieved_docs:
                    with st.expander(f"📚 Inspect Retrieved Sources ({len(retrieved_docs)} Chunks)", expanded=False):
                        for idx, doc in enumerate(retrieved_docs, start=1):
                            meta = doc.metadata or {}
                            score = meta.get("confidence_score", "N/A")
                            mode = meta.get("retrieval_mode", "Hybrid")
                            source = meta.get("source", "Unknown")
                            page = meta.get("page", "?")
                            chunk = meta.get("chunk", idx)

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div class="source-card-header">
                                        <span class="source-card-title">Chunk #{idx} • 📄 {source} (Page {page}, Part {chunk})</span>
                                        <span class="match-badge-high">Relevance: {score}% [{mode}]</span>
                                    </div>
                                    <div class="source-card-snippet">{doc.page_content.strip()}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.caption(f"⚡ *Synthesized in {latency_val:.2f}s using Gemini 2.5 Flash*")

                # Save assistant response with full citations and docs to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "latency": latency_val,
                    "sources": cited_sources,
                    "docs": retrieved_docs,
                })

            except Exception as e:
                st.error("❌ Failed to generate response.")
                st.exception(e)