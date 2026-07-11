import os
import shutil
import time
from time import perf_counter

import streamlit as st
from utils.export_chat import (
    export_chat_txt,
    export_chat_markdown
)
from utils.pdf_reader import extract_documents_from_pdfs
from utils.text_splitter import split_documents
from utils.embeddings import get_embedding_model
from utils.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
)
from utils.llm import get_llm
from utils.rag_pipeline import ask_question
# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Multi-PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)
def load_css():
    with open("assets/styles.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()
# ==================================================
# Session State Initialization
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "keyword_retriever" not in st.session_state:
    st.session_state.keyword_retriever = None

sources = []
docs = []

# ==================================================
# Load Embedding Model
# ==================================================
embedding_model = get_embedding_model()

# ==================================================
# Load Existing Vector Database
# ==================================================
if st.session_state.vector_store is None:

    try:

        vector_store = load_vector_store(
            embedding_model
        )

        if vector_store is not None:

            st.session_state.vector_store = vector_store
            st.session_state.embedding_model = embedding_model
            st.session_state.llm = get_llm()

    except Exception:

        # Ignore loading errors if the database
        # doesn't exist yet.
        pass

# ==================================================
# Main Title
# ==================================================
with st.container():
    st.caption("Version 1.0 • June 2026 • developed by Anushka Dwivedi")
    st.title("Parchment.ai")

    st.markdown(
        """
Upload multiple PDF documents and ask questions
using Retrieval-Augmented Generation (RAG).
"""
    )
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Knowledge Base",
        "Loaded" if st.session_state.vector_store else "Not Loaded"
    )

with col2:
    st.metric(
        "Messages",
        len(st.session_state.messages)
    )

with col3:
    st.metric(
        "Model",
        "Gemini"
    )


if st.session_state.vector_store is not None:

    st.success("✅ Knowledge Base Loaded")

else:

    st.warning("⚠️ No Knowledge Base Loaded")

st.divider()

# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    
    st.header("📚 Upload PDFs")
    if st.button(
    "🗑️ Clear Chat",
    use_container_width=True
):

      st.session_state.messages = []

      st.success("Chat history cleared.")

      st.rerun()

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.divider()

    st.subheader("📂 Database")

    if st.session_state.vector_store is not None:

        st.success(
    f"✅ Knowledge Base Loaded ({len(st.session_state.messages)} chat messages)"
                  )

    else:

        st.warning("Not Loaded")
    st.divider()

    st.subheader("📊 Status")

    if st.session_state.vector_store:

      st.success("Knowledge Base Ready")

      st.metric(
        "💬 Chat Messages",
        len(st.session_state.messages)
    )

    else:

      st.warning("Upload PDFs")

    if st.button("🔄 Rebuild Vector Database"):

        st.session_state.vector_store = None
        st.session_state.embedding_model = None
        st.session_state.llm = None
        st.session_state.messages = []

        if os.path.exists("vector_db"):

            shutil.rmtree("vector_db")

        st.success("Vector Database Deleted")

        st.rerun()
    st.divider()
    st.subheader("📥 Export Chat")
    txt_data = export_chat_txt(
        st.session_state.messages
          )

    md_data = export_chat_markdown(
        st.session_state.messages
      )
    st.divider()

    st.subheader("💬 Chat Management")
    st.download_button(
      label="📄 Download TXT",
      data=txt_data,
      file_name="chat_history.txt",
      mime="text/plain",
      use_container_width=True
      )
    st.download_button(
       label="📝 Download Markdown",
       data=md_data,
       file_name="chat_history.md",
       mime="text/markdown",
       use_container_width=True
       )
    st.metric(
    "Unique Sources",
    len(set(sources))
)

st.divider()
with st.container():

    st.info("Sources Used")

    for source, page in sorted(sources):

        st.write(
            f"📄 {source} | Page {page}"
        )
# ==================================================
# Build Knowledge Base
# ==================================================
if uploaded_files and st.session_state.vector_store is None:

    st.info("📄 Processing uploaded PDFs...")
    progress=st.progress(0)
    status=st.empty()

    # ----------------------------------------------
    # Step 1 - Extract Documents
    # ----------------------------------------------
    with st.spinner("Extracting text from PDFs..."):
        status.info("📄 Reading uploaded PDFs...")
        progress.progress(10)
        documents = extract_documents_from_pdfs(uploaded_files)

    if len(documents) == 0:

        st.error("❌ No readable text found in the uploaded PDFs.")

        st.stop()

    st.success(f"✅ Extracted {len(documents)} pages.")

    # ----------------------------------------------
    # Preview Extracted Text
    # ----------------------------------------------
    preview_text = "\n\n".join(
        doc.page_content
        for doc in documents[:3]
    )

    with st.expander("📄 Preview Extracted Text"):

        st.write(preview_text[:5000])

    # ----------------------------------------------
    # Step 2 - Split Documents
    # ----------------------------------------------
    with st.spinner("Splitting documents into chunks..."):
        status.info("✂️ Splitting documents into chunks...")
        progress.progress(30)
        chunks = split_documents(documents)

    if len(chunks) == 0:

        st.error("❌ No chunks were created.")

        st.stop()

    st.success(f"✅ Created {len(chunks)} chunks.")

    # ----------------------------------------------
    # Preview Chunks
    # ----------------------------------------------
    with st.expander("🧩 Preview First 3 Chunks"):

        for i, chunk in enumerate(chunks[:3]):

            st.markdown(f"### Chunk {i+1}")

            st.write(chunk.page_content)

            st.markdown("###metadata")
            st.json(chunk.metadata)

            st.divider()

    # ----------------------------------------------
    # Step 3 - Load Embedding Model
    # ----------------------------------------------
    st.info("Loading Gemini Embedding Model...")
    status.info("🧠 Loading Gemini Embedding Model...")
    progress.progress(50)
    embedding_model = get_embedding_model()

    st.success("✅ Embedding model loaded.")

    # ----------------------------------------------
    # Step 4 - Create Vector Store
    # ----------------------------------------------
    with st.spinner("Creating FAISS Vector Database..."):

        try:
            status.info("📚 Creating FAISS Vector Database...")
            progress.progress(70)
            vector_store = create_vector_store(
                chunks,
                embedding_model
            )

        except Exception as e:

            st.error("❌ Failed to create the vector database.")

            st.exception(e)

            st.stop()

    st.success("✅ Vector Database Created")

    # ----------------------------------------------
    # Step 5 - Save Vector Store
    # ----------------------------------------------
    with st.spinner("Saving Vector Database..."):
        status.info("💾 Saving Vector Database...")
        progress.progress(90)
        save_vector_store(vector_store)

    st.success("✅ Vector Database Saved")

    # ----------------------------------------------
    # Step 6 - Save Session State
    # ----------------------------------------------
    st.session_state.vector_store = vector_store
    st.session_state.embedding_model = embedding_model

    if st.session_state.llm is None:

        st.session_state.llm = get_llm()
    progress.progress(100)
    st.rerun()
# ==================================================
# Chat Interface
# ==================================================
if st.session_state.vector_store is not None:

    st.divider()
    with st.container():

      st.subheader("💬 CHAT with  your PDFs")
      st.caption(
    "Ask questions based on the uploaded documents."
       )
      st.write(
        "Ask questions about the uploaded documents."
    )

    # ----------------------------------------------
    # Display Chat History
    # ----------------------------------------------
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # ----------------------------------------------
    # Chat Input
    # ----------------------------------------------
    question = st.chat_input(
        "Ask your question..."
    )

    if question:

        # ------------------------------------------
        # Show User Message
        # ------------------------------------------
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # ------------------------------------------
        # Ask RAG Pipeline
        # ------------------------------------------
        try:
            start_time = perf_counter()
            with st.spinner("Thinking..."):

                answer, docs = ask_question(
                    question=question,
                    vector_store=st.session_state.vector_store,
                    llm=st.session_state.llm,
                    messages=st.session_state.messages,
                )
                end_time = perf_counter()
                response_time = round(end_time - start_time, 2)
                chunk_count = len(docs)

                sources = []
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    chunk = doc.metadata.get("chunk", "Unknown")

                    st.markdown(f"### Source {i + 1}")
                    st.markdown(f"**📄 File:** {source}")
                    st.markdown(f"**📑 Page:** {page}")
                    st.markdown(f"**🔹 Chunk:** {chunk}")
                    st.write(doc.page_content)
                    st.divider()

                    sources.append(source)

                sources = sorted(set(sources))

        except Exception as e:

            st.error("❌ Failed to generate response.")

            st.exception(e)

            st.stop()

        # ------------------------------------------
        # Show Assistant Response
        # ------------------------------------------
        with st.chat_message("assistant"):

           message_placeholder = st.empty()

           full_response = ""

           for word in answer.split():

              full_response += word + " "

              message_placeholder.markdown(full_response + "▌")

              time.sleep(0.02)

           message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        with st.expander("📊 Response Details", expanded=True):

            col1, col2, col3 = st.columns(3)

            col1.metric(
                 "Response Time",
                 f"{response_time:.2f}s"
                       )

            col2.metric(
                  "Chunks",
                  chunk_count
                      )

            col3.metric(
                 "Documents",
                  len(sources)
                       )

            st.divider()

            st.write("🧠 **Model:** Gemini 2.5 Flash")

            st.write("📄 **Documents Used:**")

            for source in sources:

                st.write(f"• {source}")
        st.markdown("### 📚 Sources Used")
        source_pages = set()
        for doc in docs:

          source = doc.metadata.get("source", "Unknown")
          page = doc.metadata.get("page", "Unknown")

          source_pages.add((source, page))
        for source, page in sorted(source_pages):

           st.markdown(
        f"- 📄 **{source}** (Page {page})"
    )

        # ------------------------------------------
        # Retrieved Sources
        # ------------------------------------------
        with st.expander("📚 Retrieved Sources", expanded=False):

            for i, doc in enumerate(docs):

                st.markdown(f"## 📄 Result {i + 1}")

                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        f"**Source:** {doc.metadata.get('source', 'Unknown')}"
                    )

                    st.write(
                        f"**Page:** {doc.metadata.get('page', 'Unknown')}"
                    )

                with col2:
                    st.write(
                        f"**Chunk:** {doc.metadata.get('chunk', 'N/A')}"
                    )

                    st.write(
                        f"**Rank:** #{i + 1}"
                    )

                st.markdown("### Content")

                st.write(doc.page_content)

                st.divider()

                st.write(
                    f"**Words:** {doc.metadata.get('words', 'Unknown')}"
                )

                st.write(
                    f"**Characters:** {doc.metadata.get('characters', 'Unknown')}"
                )

else:

       st.info(
         """
           # 👋 Welcome!

              To begin:

                 1. Upload one or more PDFs.
                 2. Wait for the knowledge base to be created.
                 3. Ask questions in natural language.

     The chatbot will retrieve the most relevant document chunks and generate answers using Gemini + FAISS.
"""
)
st.divider()

st.markdown(
    """
<div style="text-align:center; color:gray; padding:20px;">
Built with ❤️ using <b>Streamlit</b> • <b>LangChain</b> • <b>Gemini</b> • <b>FAISS</b>
</div>
""",
    unsafe_allow_html=True,
)