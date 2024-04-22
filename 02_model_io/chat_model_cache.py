import os
from dotenv import load_dotenv
import time
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

set_llm_cache(InMemoryCache())

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)
start = time.time()
result = chat.invoke([HumanMessage(content="こんにちは！")])

end = time.time()
print(result.content)
print(f"実行時間: {end - start}秒")

start = time.time()
result = chat.invoke([HumanMessage(content="こんにちは！")])

end = time.time()
print(result.content)
print(f"実行時間: {end - start}秒")
