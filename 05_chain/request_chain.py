import os
from dotenv import load_dotenv
import requests
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

prompt = PromptTemplate(
    input_variables=["query", "requests_result"],
    template="""以下の文章を元に答えてください。
文章: {requests_result}
質問: {query}""",
)

llm_chain = prompt | chat

url = "https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json"
res = requests.get(url)
json_data = res.json()
requests_result = json.dumps(json_data)

print(
    llm_chain.invoke(
        {
            "query": "東京の天気について教えて",
            "requests_result": requests_result,
        }
    ).content
)
