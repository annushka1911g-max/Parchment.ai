from langchain_core.prompts import ChatPromptTemplate


def get_prompt():

    template = """
You are a helpful AI assistant.

Use ONLY the provided document context and previous conversation.

If the answer cannot be found, say:

"I couldn't find that information in the uploaded documents."

-----------------------
Previous Conversation:

{chat_history}

-----------------------
Retrieved Context:

{context}

-----------------------
Question:

{question}

Answer:
"""

    return ChatPromptTemplate.from_template(template)