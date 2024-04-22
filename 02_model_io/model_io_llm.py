import os
from dotenv import load_dotenv
from langchain_openai.llms import OpenAI

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo-instruct",
)

result = llm.invoke("美味しいラーメンを", stop="。")
print(result)
