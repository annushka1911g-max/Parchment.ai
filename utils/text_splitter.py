from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split documents into chunks while preserving and enriching metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):

        chunk.metadata["chunk"] = index

        chunk.metadata["words"] = len(
            chunk.page_content.split()
        )

        chunk.metadata["characters"] = len(
            chunk.page_content
        )

    return chunks