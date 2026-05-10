from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)

docs = retriever.invoke("large language models")

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print(f"\nTitle : ", doc.metadata.get("Title"))
    print(f"\nAuthor : ", doc.metadata.get("Author"))
    print(f"\nSummary : ", doc.page_content)