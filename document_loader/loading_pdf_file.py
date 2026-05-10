from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader('research_papers.pdf')
docs = data.load()
#print(docs[0].page_content) #it loads only first page because we wrote of docs[0]
print(len(docs))  # it gives 15 beacuse pdf has 15 pages means each page is one doc object , and each doc object has a meta_data and page_content