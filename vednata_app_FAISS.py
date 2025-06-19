import os
import streamlit as st
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load .env or Streamlit secrets
load_dotenv()
folder_path = "./vedanta_texts"
api_key = st.secrets.get("OPENAI_API_KEY")
logging.info("api_key from Secrets" ,api_key)
# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4", api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key)


def load_all_pdfs_from_folder(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs


# Cache vector DB so it's not reloaded every time
@st.cache_resource
def load_faiss():
    # Load and split PDF
   # loader = PyMuPDFLoader("01-Atma-Bodha-Class-Notes.pdf")  # Make sure this PDF is in your root folder
    #docs = loader.load()
    docs = load_all_pdfs_from_folder(folder_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create FAISS index (in-memory)
    return FAISS.from_documents(chunks, embeddings)

db = load_faiss()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Streamlit UI
st.title("🕉️ Vedanta Chatbot")
query = st.text_input("Ask your question about Vedanta:")

if query:
    with st.spinner("Thinking..."):
        result = qa.invoke({"query": query})
        st.write("📜", result)
