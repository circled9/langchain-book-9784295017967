import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema import HumanMessage

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

output_parser = CommaSeparatedListOutputParser()

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

result = chat.invoke(
    [
        HumanMessage(content="Appleが開発した代表的な製品を3つ教えてください"),
        HumanMessage(content=output_parser.get_format_instructions()),
    ]
)

output = output_parser.parse(result.content)

for item in output:
    print("代表的な製品 => " + item)
