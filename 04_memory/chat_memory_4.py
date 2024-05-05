import os
from dotenv import load_dotenv
import chainlit as cl
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.schema import HumanMessage

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)


@cl.on_chat_start
async def on_start_chat():
    thread_id = None
    while not thread_id:
        res = await cl.AskUserMessage(
            content="私は会話の文脈を考慮した返答ができるチャットボットです。スレッドIDを入力してください。",
            timeout=600,
        ).send()
        if res:
            thread_id = res["output"]

    history = RedisChatMessageHistory(
        session_id=thread_id,
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

    memory_message_result = chain.memory.load_memory_variables({})

    messages = memory_message_result["history"]

    for message in messages:
        if isinstance(message, HumanMessage):
            await cl.Message(
                author="User",
                content=f"{message.content}",
            ).send()
        else:
            await cl.Message(
                author="ChatBot",
                content=f"{message.content}",
            ).send()
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message):
    chain = cl.user_session.get("chain")

    result = chain.invoke(message.content)

    await cl.Message(content=result["response"]).send()
