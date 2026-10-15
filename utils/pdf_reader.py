from pypdf import PdfReader
from langchain_core.documents import Document


def extract_documents_from_pdfs(pdf_docs):

    documents = []

    for pdf in pdf_docs:

        reader = PdfReader(pdf)

        pdf_name = pdf.name

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": pdf_name,
                            "page": page_number
                        }
                    )
                )

    return documents