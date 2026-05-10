# app.py

import os
import uuid
import tempfile

from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stApp {
    background: linear-gradient(135deg,#0E1117,#111827);
    color: white;
}

.title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
}

.stButton>button {
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<p class="title">📚 PDF RAG Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Upload your PDF and ask questions using AI</p>',
    unsafe_allow_html=True
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")

    chunk_size = st.slider(
        "Chunk Size",
        500,
        2000,
        1000
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        0,
        500,
        200
    )

    st.divider()

    st.markdown("### 📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

# ==========================================
# SESSION STATE
# ==========================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# ==========================================
# CACHE MODELS
# ==========================================

@st.cache_resource
def load_embedding_model():
    return OpenAIEmbeddings()

@st.cache_resource
def load_llm():
    return init_chat_model(
        model="gpt-5-mini"
    )

embedding_model = load_embedding_model()
llm = load_llm()

# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file is not None:

    # process only new uploaded pdf
    if st.session_state.current_pdf != uploaded_file.name:

        with st.spinner("📖 Reading PDF..."):

            # create temp directory
            temp_dir = tempfile.mkdtemp()

            pdf_path = os.path.join(
                temp_dir,
                uploaded_file.name
            )

            # save uploaded pdf
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # load pdf
            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            st.success(
                f"✅ PDF Loaded Successfully ({len(docs)} pages)"
            )

            # split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = splitter.split_documents(docs)

            st.info(
                f"✂️ Total Chunks Created: {len(chunks)}"
            )

            # unique chroma db path
            db_path = f"temp-chroma-db/{uuid.uuid4()}"

            # create vectorstore
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=db_path
            )

            # save vectorstore
            st.session_state.vectorstore = vectorstore

            # save current pdf name
            st.session_state.current_pdf = uploaded_file.name

            # clear old chats
            st.session_state.chat_history = []

            st.success(
                "✅ Embeddings Created & Stored Successfully"
            )

# ==========================================
# DISPLAY OLD CHAT HISTORY
# ==========================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ==========================================
# CHAT INPUT
# ==========================================

query = st.chat_input(
    "Ask question from your PDF..."
)

# ==========================================
# HANDLE QUESTION
# ==========================================

if query:

    if st.session_state.vectorstore is None:

        st.warning("⚠️ Please upload a PDF first")

    else:

        # save user message
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query
            }
        )

        # show user message instantly
        with st.chat_message("user"):

            st.markdown(query)

        # assistant message
        with st.chat_message("assistant"):

            thinking_placeholder = st.empty()

            thinking_placeholder.markdown(
                "🤖 AI is thinking..."
            )

            # retriever
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )

            # retrieve docs
            docs = retriever.invoke(query)

            # build context
            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            # prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:
"I could not find the answer in the document."
                        """
                    ),
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

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            # llm response
            response = llm.invoke(final_prompt)

            answer = response.content

            # remove thinking text
            thinking_placeholder.empty()

            # show answer
            st.markdown(answer)

        # save ai response
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <center>
    Made with ❤️ using LangChain + Streamlit + OpenAI
    </center>
    """,
    unsafe_allow_html=True
)