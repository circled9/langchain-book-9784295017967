import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    model="text-embedding-ada-002",
)

database = Chroma(persist_directory="./.data", embedding_function=embeddings)

documents = database.similarity_search("飛行機の最高速度は？")

print(f"ドキュメントの数: {len(documents)}")

for document in documents:
    print(f"ドキュメントの内容: {document.page_content}")
