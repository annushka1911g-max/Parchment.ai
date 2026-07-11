import os
import time

from langchain_community.vectorstores import FAISS

VECTOR_DB_PATH = "vector_db"

def create_vector_store(documents, embedding_model, persist_directory="vector_db"):
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(persist_directory)
    return vector_store


def save_vector_store(vector_store, persist_directory="vector_db"):
    vector_store.save_local(persist_directory)


def load_vector_store(embedding_model, persist_directory="vector_db"):
    return FAISS.load_local(
        persist_directory,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

