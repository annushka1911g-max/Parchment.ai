def export_chat_txt(messages):
    """
    Convert chat messages into plain text.
    """

    lines = []

    for message in messages:

        role = message["role"].capitalize()

        content = message["content"]

        lines.append(f"{role}:")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)
def export_chat_markdown(messages):
    """
    Convert chat messages into Markdown.
    """

    md = []

    md.append("# Multi-PDF RAG Chat Export\n")

    for message in messages:

        role = message["role"].capitalize()

        content = message["content"]

        md.append(f"## {role}")
        md.append(content)
        md.append("")

    return "\n".join(md)