from typing import List, Tuple, Generator, Optional
from langchain_core.documents import Document
from utils.prompt import get_prompt
from utils.chat_memory import get_chat_history


def build_context_string(docs: List[Document]) -> str:
    """
    Builds a structured context string from retrieved documents for the LLM.
    """
    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown Document")
        page = meta.get("page", "?")
        chunk_idx = meta.get("chunk", i)
        confidence = meta.get("confidence_score", "N/A")
        mode = meta.get("retrieval_mode", "Hybrid")

        chunk_header = (
            f"=== Context Item [{i}] | Document: {source} | Page: {page} | "
            f"Match: {mode} ({confidence}%) ==="
        )
        formatted_chunks.append(f"{chunk_header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted_chunks)


def stream_ask_question(
    question: str,
    retriever,
    llm,
    messages: list,
    k: int = 4,
    alpha: float = 0.65,
    source_filter: Optional[str] = None,
) -> Tuple[Generator[str, None, None], List[Document]]:
    """
    Retrieves relevant documents using HybridRetriever and streams the generated answer.

    Returns:
        (stream_generator, retrieved_docs)
    """
    # 1. Retrieve documents via HybridRetriever
    docs = retriever.search(
        query=question,
        k=k,
        alpha=alpha,
        source_filter=source_filter,
    )

    # 2. Build structured context and chat history
    context = build_context_string(docs)
    chat_history = get_chat_history(messages)

    # 3. Create chain
    prompt = get_prompt()
    chain = prompt | llm

    # 4. Stream generator
    def token_stream():
        for chunk in chain.stream({
            "context": context,
            "question": question,
            "chat_history": chat_history,
        }):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)

    return token_stream(), docs


def ask_question(
    question: str,
    retriever,
    llm,
    messages: list,
    k: int = 4,
    alpha: float = 0.65,
    source_filter: Optional[str] = None,
) -> Tuple[str, List[Document]]:
    """
    Synchronous fallback to retrieve documents and return the full generated answer.
    """
    stream_gen, docs = stream_ask_question(
        question=question,
        retriever=retriever,
        llm=llm,
        messages=messages,
        k=k,
        alpha=alpha,
        source_filter=source_filter,
    )

    full_answer = "".join(list(stream_gen))
    return full_answer, docs
