import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "iPhone8のリリース日を教えて"},
    ],
)

print(response)
