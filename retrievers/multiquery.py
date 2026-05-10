from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


# OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# Create Vector Store
vectorstore = Chroma.from_documents(docs, embeddings)


# Base Retriever
retriever = vectorstore.as_retriever()


# OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# Multi Query Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


query = "What is gradient descent?"


retrieved_docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")


for doc in retrieved_docs:
    print(doc.page_content)