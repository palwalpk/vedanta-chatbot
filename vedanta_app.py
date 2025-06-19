import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# Initialize model and retriever
loader = PyMuPDFLoader("01-Atma-Bodha-Class-Notes.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

llm = ChatOpenAI(temperature=0, model="gpt-4")
#In-Memory Chroma: Doesn’t use SQLite since Streamlit Cloud is not supporting it
#db = Chroma.from_documents(chunks, embeddings, persist_directory="./db")

db = Chroma.from_documents(
    chunks,  # ← this should be your list of Document objects
    OpenAIEmbeddings(),
    persist_directory=None  # in-memory, not persisted
)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Streamlit UI
st.set_page_config(page_title="Vedanta Chatbot", layout="wide")
st.title("🕉️ Vedanta Chatbot")
st.markdown("Ask anything about Chinmay Mission : Atma Bodha Class Notes")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        result = qa.invoke({"query": query})
        st.markdown("### 📜 Answer:")
        st.write(result)
