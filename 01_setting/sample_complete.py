import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="今日の天気がとても良くて、気分が",
    stop="。",
    max_tokens=100,
    n=2,
    temperature=0.5,
)

print(response)
