from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from rich import print

data = TextLoader("../document_loader/chatrapati_shivaji_raje.txt")

docs = data.load()

splitter = CharacterTextSplitter(
    separator = "",   # means do not count /n/n funda means count if there is no lines 
    chunk_size = 10,
    chunk_overlap = 1
)

chunks = splitter.split_documents(docs)

print(chunks)

print('===============================================')

print(len(chunks))


print('------------------ Printing each chunks ------------------')
for i in chunks:
    print(i.page_content)

#print(len(docs))
#note that this is very basic and do not use in industry because it chunk randomy accprding to the chunk_size 
#👉 It does NOT care about meaning
#👉 It just cuts raw characters
#✅ Pros
#Simple
#Fast
#No dependency on tokenizer
#❌ Cons
#Breaks words/sentences randomly
#Bad for semantic understanding
#Not ideal for LLMs