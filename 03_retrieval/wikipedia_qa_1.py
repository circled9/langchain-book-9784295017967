import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.retrievers.wikipedia import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500,
    top_k_results=2,
)

prompt = ChatPromptTemplate.from_messages([("human", "{context}")])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)

chain = create_retrieval_chain(retriever, combine_docs_chain)

result = chain.invoke({"input": "バーボンウィスキーとは？"})

source_documents = result["context"]

print(f"検索結果: {len(source_documents)}件")

for document in source_documents:
    print("--------取得したメタデータ--------")
    print(document.metadata)
    print("--------取得したテキスト--------")
    print(document.page_content[:100])
print("--------返答--------")
print(result["answer"])
