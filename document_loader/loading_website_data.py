from langchain_community.document_loaders import WebBaseLoader

data = WebBaseLoader('https://www.petrinadarrah.com/posts/best-places-to-visit-in-new-zealand')

docs = data.load()

print(len(docs)) # it gives 1 because in website only one object is made , each has a meta_data and page_content

print(docs[0].page_content)

