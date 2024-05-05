import os
from dotenv import load_dotenv
import chainlit as cl
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

history = RedisChatMessageHistory(
    session_id="chat_history",
    url=REDIS_URL,
)

memory = ConversationBufferMemory(
    return_messages=True,
    chat_memory=history,
)

chain = ConversationChain(
    memory=memory,
    llm=chat,
)


@cl.on_chat_start
async def on_start_chat():
    await cl.Message(
        content="私は会話の文脈を考慮した返答ができるチャットボットです。メッセージを入力してください。"
    ).send()


@cl.on_message
async def on_message(message):
    result = chain.invoke(message.content)

    await cl.Message(content=result["response"]).send()
