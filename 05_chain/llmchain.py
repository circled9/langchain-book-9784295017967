import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか？", input_variables=["product"]
)

chain = prompt | chat

result = chain.invoke({"product": "iPhone"})

print(result.content)
