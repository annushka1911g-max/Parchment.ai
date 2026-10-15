import json
from typing import List, Dict, Any


def export_chat_txt(messages: List[Dict[str, Any]]) -> str:
    """
    Convert chat messages into plain text with source citations if present.
    """
    lines = []
    lines.append("=== Parchment.ai Chat Export ===")
    lines.append("")

    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"[{role}]:")
        lines.append(content)

        # Include citations if present
        sources = msg.get("sources", [])
        if sources:
            lines.append("\nSources Referenced:")
            for s in sources:
                lines.append(f" - {s.get('source', 'Unknown')} (Page {s.get('page', '?')}, Match: {s.get('confidence', 'N/A')}%)")

        lines.append("-" * 40)
        lines.append("")

    return "\n".join(lines)


def export_chat_markdown(messages: List[Dict[str, Any]]) -> str:
    """
    Convert chat messages into formatted Markdown with source annotations.
    """
    md = []
    md.append("# 🤖 Parchment.ai - RAG Chat Export\n")

    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        latency = msg.get("latency")

        header = f"## 💬 {role}"
        if latency is not None:
            header += f" *({latency}s)*"
        md.append(header)
        md.append(content)
        md.append("")

        sources = msg.get("sources", [])
        if sources:
            md.append("### 📚 Cited Sources")
            for s in sources:
                md.append(f"- **{s.get('source', 'Unknown')}** (Page {s.get('page', '?')}) — *Confidence: {s.get('confidence', 'N/A')}% [{s.get('mode', 'Hybrid')}]*")
            md.append("")

    return "\n".join(md)


def export_chat_json(messages: List[Dict[str, Any]]) -> str:
    """
    Convert chat messages into clean structured JSON.
    """
    return json.dumps(messages, indent=2, ensure_ascii=False)