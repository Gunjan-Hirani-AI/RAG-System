from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from rich import print

data = PyPDFLoader('../document_loader/research_papers.pdf')

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

docs = data.load()

chunks = splitter.split_documents(docs)

print(len(chunks))
# this is used by chatgpt 2 , but still it is not prefrable to use this 
#👉 How it works
#Splits based on tokens (LLM units), not characters

#Tokens ≠ words
#Example:

#"ChatGPT is awesome"
#→ ["Chat", "G", "PT", " is", " awesome"]
# means here the tokention wise word not character

#i have a question means how the TokenTextSplitter knows this is token ? means they can not use llm then how it decide ?

#answer :  Short answer: it use a tiktoken liabrary which has already workds knowlgae to according to that it see your text and token it simple.
#👉 TokenTextSplitter does NOT use an LLM. It uses a tokenizer.

#🔹 What actually happens

#When you do:

#from langchain_text_splitters import TokenTextSplitter

#Internally it uses a tokenizer library (usually tiktoken for OpenAI models).

#🔍 So what is a tokenizer?

#A tokenizer is just a rule-based + pre-trained algorithm that converts text into tokens.

#👉 It does:

#"ChatGPT is awesome"
#↓
#["Chat", "G", "PT", " is", " awesome"]

#👉 Then maps to IDs:

#["Chat", "G", "PT"] → [1234, 567, 89]

#🔹 How does it "decide" tokens?

#It doesn’t think. It follows predefined vocabulary + rules.

#Example:

#The tokenizer already has a vocabulary like:

#"Chat" ✅
#"GPT" ❌ (so split into "G" + "PT")
#"is" ✅
#"awesome" ✅

#So it breaks text into the closest known pieces.
