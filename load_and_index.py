from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = PyMuPDFLoader("01-Atma-Bodha-Class-Notes.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
#In-Memory Chroma: Doesn’t use SQLite since Streamlit Cloud is not supporting it
#db = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
db = Chroma.from_documents(chunks, embeddings, persist_directory=None)
db.persist()
print("Indexing complete.")
