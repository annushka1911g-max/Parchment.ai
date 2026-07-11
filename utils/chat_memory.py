def get_chat_history(messages, max_messages=6):
    """
    Returns the recent chat history as a formatted string.
    """

    history = []

    for message in messages[-max_messages:]:

        role = message["role"].capitalize()

        content = message["content"]

        history.append(f"{role}: {content}")

    return "\n".join(history)