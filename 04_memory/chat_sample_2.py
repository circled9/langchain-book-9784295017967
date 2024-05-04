import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

result = chat.invoke(
    [
        HumanMessage(content="茶碗蒸しを作るのに必要な食材を教えて"),
        AIMessage(
            content="""
茶碗蒸しを作るのに必要な食材は以下の通りです。
- 卵
- 鶏肉や野菜などの具材
- だし汁
- 醤油
- 塩
- 砂糖
- 酒

これらの食材を使って、茶碗蒸しを作ることができます。
"""
        ),
        HumanMessage(content="前の回答を英語に翻訳して"),
    ]
)
print(result.content)
