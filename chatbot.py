from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-4")
db = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

while True:
    query = input("Ask VedantaBot: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa.invoke({"query": query})
    print("\n📜 Answer:", result, "\n")
