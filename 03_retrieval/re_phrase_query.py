import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.retrievers.wikipedia import WikipediaRetriever
from langchain.retrievers.re_phraser import RePhraseQueryRetriever
from langchain_core.prompts import PromptTemplate

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500,
)

llm = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

prompt_template = "以下の質問からWikipediaで検索するべきキーワードを抽出してください\n\n質問: {context}"
prompt = PromptTemplate(
    input_variables=["context"],
    template=prompt_template,
)

re_phrase_query_retriever = RePhraseQueryRetriever.from_llm(
    llm=llm,
    prompt=prompt,
    retriever=retriever,
)

documents = re_phrase_query_retriever.get_relevant_documents(
    "私はラーメンが好きです。ところでバーボンウイスキーとは何ですか？"
)

print(documents)
