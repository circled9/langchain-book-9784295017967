import os
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores.chroma import Chroma

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load()

text_splitter = SpacyTextSplitter(
    chunk_size=300,
    pipeline="ja_core_news_sm",
)
splitted_documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    model="text-embedding-ada-002",
)

database = Chroma(persist_directory="./.data", embedding_function=embeddings)

database.add_documents(
    splitted_documents,
)

print("データベースの作成が完了しました。")
