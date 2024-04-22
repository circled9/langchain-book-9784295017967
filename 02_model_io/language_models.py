import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

result = chat.invoke([HumanMessage(content="こんにちは！")])
print(result.content)
