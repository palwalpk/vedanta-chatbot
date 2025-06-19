from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()
loader = PyMuPDFLoader("01-Atma-Bodha-Class-Notes.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
llm = ChatOpenAI(temperature=0, model="gpt-4")
#In-Memory Chroma: Doesn’t use SQLite since Streamlit Cloud is not supporting it
#db = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
db = Chroma.from_documents(
    chunks,  # ← this should be your list of Document objects
    OpenAIEmbeddings(),
    persist_directory=None  # in-memory, not persisted
)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

while True:
    query = input("Ask VedantaBot: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa.invoke({"query": query})
    print("\n📜 Answer:", result, "\n")
