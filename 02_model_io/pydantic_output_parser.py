import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, field_validator

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)


class Smartphone(BaseModel):
    release_date: str = Field(description="スマートフォンの発売日")
    screen_inches: float = Field(description="スマートフォンの画面サイズ（インチ）")
    os_installed: str = Field(description="スマートフォンにインストールされているOS")
    type_name: str = Field(description="スマートフォンのモデル名")

    @field_validator("screen_inches")
    def validate_screen_inches(cls, field):
        if field <= 0:
            raise ValueError("Screen inches must be a positive number")
        return field


parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=Smartphone), llm=chat
)

result = chat.invoke(
    [
        HumanMessage(content="Androidでリリースしたスマートフォンを1個挙げて"),
        HumanMessage(content=parser.get_format_instructions()),
    ]
)

parsed_result = parser.parse(result.content)

print(f"モデル名: {parsed_result.type_name}")
print(f"画面サイズ: {parsed_result.screen_inches}インチ")
print(f"OS: {parsed_result.os_installed}")
print(f"スマートフォンの発売日: {parsed_result.release_date}")
