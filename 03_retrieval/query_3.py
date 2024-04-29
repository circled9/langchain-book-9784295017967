import os
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    model="text-embedding-ada-002",
)

database = Chroma(persist_directory="./.data", embedding_function=embeddings)

retriever = database.as_retriever()

qa = RetrievalQA.from_llm(
    llm=chat,
    retriever=retriever,
    return_source_documents=True,
)

result = qa.invoke(input="飛行車の最高速度を教えて")

print(result["result"])

print(result["source_documents"])
