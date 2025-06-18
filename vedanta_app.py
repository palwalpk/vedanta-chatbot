import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize model and retriever
llm = ChatOpenAI(temperature=0, model="gpt-4")
db = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
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
