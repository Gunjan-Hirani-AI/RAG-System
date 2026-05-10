from langchain_community.document_loaders import TextLoader
from rich import print
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = TextLoader('../document_loader/chatrapati_shivaji_raje.txt')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=1
)

docs = data.load()

chunk = splitter.split_documents(docs)

print(chunk)
print(len(chunk))

# this RecursiveCharacterTextSplitter we can use this in the production because it less depend on the chunk_size and more focus on meaning ,

# but how RecursiveCharacterTextSplitter will do this ?

# it use a ["\n\n", "\n", " ", ""] means it see the first double line(\n\n) if found then it start splitting from that point , if not then it see s the one line(/n) , if not then it see the " " , if not then it see the "" , if found then start the splitting from that point