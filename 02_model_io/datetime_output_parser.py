import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain.schema import HumanMessage

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

output_parser = DatetimeOutputParser()

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

prompt = PromptTemplate.from_template("{product}のリリース日を教えて")

result = chat.invoke(
    [
        HumanMessage(content=prompt.format(product="iPhone8")),
        HumanMessage(content=output_parser.get_format_instructions()),
    ]
)

output = output_parser.parse(result.content)

print(output)
