import os
from dotenv import load_dotenv
import chainlit as cl
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

memory = ConversationSummaryMemory(
    llm=chat,
    return_messages=True,
)

chain = ConversationChain(
    memory=memory,
    llm=chat,
)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="私は会話の文脈を考慮した返答ができるチャットボットです。メッセージを入力してください。"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    messages = chain.memory.load_memory_variables({})["history"]

    print(f"保存されているメッセージの数: {len(messages)}")

    for saved_message in messages:
        print(saved_message.content)

    result = chain.invoke(message.content)

    await cl.Message(content=result["response"]).send()
