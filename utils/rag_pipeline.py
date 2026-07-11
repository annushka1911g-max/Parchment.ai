from utils.prompt import get_prompt
from utils.chat_memory import get_chat_history

def ask_question(question, vector_store, llm, messages):
    chat_history = get_chat_history(messages)

    semantic_docs = vector_store.similarity_search(question, k=3)

    merged_docs = []
    seen = set()

    for doc in semantic_docs:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk"),
        )

        if key not in seen:
            seen.add(key)
            merged_docs.append(doc)

    docs = merged_docs[:5]
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = get_prompt()
    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history,
    })

    return response.content, docs
