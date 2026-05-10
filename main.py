from dotenv import load_dotenv
load_dotenv()
from rich import print
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


embedding_model = OpenAIEmbeddings()

#loading the chroma db here.
vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm = init_chat_model(
    model="gpt-5-mini"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","""
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document.
"""),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
    ]
)

print(" RAG is created ")

print("print 0 to exit")

while True:
    query = input("YOU : ")
    if query == "0":
        break

    docs= retriever.invoke(query)

    context = "/n/n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    })

    response = llm.invoke(final_prompt)

    print(f"AI : {response.content}")