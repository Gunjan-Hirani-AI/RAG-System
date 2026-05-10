from langchain_community.vectorstores import Chroma
from rich import print
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Chhatrapati Shivaji Maharaj was a महान warrior and the founder of the Maratha Empire. He was known for his excellent military strategies and strong administration.",
        metadata={"source": "Shivaji_book"}
    ),
    Document(
        page_content="Shivaji Maharaj respected all religions and promoted justice and good governance in his kingdom.",
        metadata={"source": "Shivaji_book"}
    ),
    Document(
        page_content="Maharana Pratap was a brave Rajput king who is remembered for his courage and resistance against the Mughal Empire.",
        metadata={"source": "Mewar_book"}
    ),
]  # just suppose that this are the chunkkings okay means here we skip loading and chunking because of focus only embedding and storing in chroma DB.

embeddings_model = OpenAIEmbeddings()

vectore_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    persist_directory="chroma-db"
)

result = vectore_store.similarity_search("who is maratha king?",k=2)  # here k means nearest 2 search in the DB , means see run this file you got the idea

for r in result:
    print(r.page_content)
    print(r.metadata)

# see when i run this file then automatically the chroma-db is created in this folder that has folder has :
# one folder and chroma.sqlite3
# so first folder is for storing the embeddings which we created here and onther chroma.sqlite3 file is storing for the page_content and metadata
