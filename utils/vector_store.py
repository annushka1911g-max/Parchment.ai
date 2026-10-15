import os
import pickle
import time
from typing import List, Tuple, Optional
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

VECTOR_DB_PATH = "vector_db"
DOCUMENTS_FILE = "documents.pkl"


def create_vector_store(
    documents: List[Document],
    embedding_model,
    persist_directory: str = VECTOR_DB_PATH,
    max_retries: int = 3,
) -> FAISS:
    """
    Create and save a FAISS vector store along with raw document chunks for hybrid search.
    """
    os.makedirs(persist_directory, exist_ok=True)

    for attempt in range(max_retries):
        try:
            vector_store = FAISS.from_documents(
                documents,
                embedding_model,
            )

            vector_store.save_local(persist_directory)

            # Persist document chunks for BM25 hybrid search
            doc_file_path = os.path.join(persist_directory, DOCUMENTS_FILE)
            with open(doc_file_path, "wb") as f:
                pickle.dump(documents, f)

            return vector_store

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(3)


def save_vector_store(
    vector_store: FAISS,
    documents: Optional[List[Document]] = None,
    persist_directory: str = VECTOR_DB_PATH
):
    """
    Saves the FAISS vector store and optionally documents pickle.
    """
    os.makedirs(persist_directory, exist_ok=True)
    vector_store.save_local(persist_directory)

    if documents is not None:
        doc_file_path = os.path.join(persist_directory, DOCUMENTS_FILE)
        with open(doc_file_path, "wb") as f:
            pickle.dump(documents, f)


def load_vector_store(
    embedding_model,
    persist_directory: str = VECTOR_DB_PATH,
) -> Optional[FAISS]:
    """
    Loads FAISS index from local disk.
    """
    if not os.path.exists(os.path.join(persist_directory, "index.faiss")):
        return None

    return FAISS.load_local(
        persist_directory,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def load_documents_for_hybrid(
    vector_store: Optional[FAISS] = None,
    persist_directory: str = VECTOR_DB_PATH,
) -> List[Document]:
    """
    Loads document chunks for BM25 from documents.pkl or extracts them from vector_store docstore.
    """
    doc_file_path = os.path.join(persist_directory, DOCUMENTS_FILE)
    if os.path.exists(doc_file_path):
        try:
            with open(doc_file_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    if vector_store is not None and hasattr(vector_store, "docstore"):
        try:
            return list(vector_store.docstore._dict.values())
        except Exception:
            pass

    return []