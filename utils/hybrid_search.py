from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, documents):

        self.documents = documents

        self.tokenized_docs = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_docs)

    def keyword_search(self, query, k=3):

        tokenized_query = query.lower().split()

        results = self.bm25.get_top_n(
            tokenized_query,
            self.documents,
            n=k
        )

        return results