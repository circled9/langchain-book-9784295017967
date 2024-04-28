import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.vectorstores.chroma import Chroma

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    model="text-embedding-ada-002",
)

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。
文章:
{document}

質問: {query}
""",
    input_variables=["document", "query"],
)

database = Chroma(persist_directory="./.data", embedding_function=embeddings)


@cl.on_chat_start
async def on_start_chat():
    await cl.Message(content="準備ができました！メッセージを入力してください！").send()


@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ: " + input_message.content)
    documents = database.similarity_search(input_message.content)

    documents_string = ""

    for document in documents:
        documents_string += f"""
    ---------------------------
    {document.page_content}
    """

    result = chat.invoke(
        [
            HumanMessage(
                content=prompt.format(
                    document=documents_string, query=input_message.content
                )
            )
        ]
    )
    await cl.Message(content=result.content).send()
