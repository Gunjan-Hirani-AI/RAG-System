# the purpose of this file is following :

# you do not want that if you ask a question then every time the docs is loading then chunking(splitting) then embeddings then storing , means we want only one time we do these thing and after i send a question then only search is perform in the DB okay so here that one time creaded file so we do not need created those thong again and again okay

#first we do loading 
#second is chunkking(splitting)
#third is create a embeddings
#fourth is storing in chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader('document_loader/deeplearning.pdf')

docs = data.load()

print(len(docs))    # it gives 534 because our pdf has a 534 pages

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

print(len(chunks))

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory="chroma-db"
)
