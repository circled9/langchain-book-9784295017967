import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.text_splitter import SpacyTextSplitter
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

text_splitter = SpacyTextSplitter(
    chunk_size=300,
    pipeline="ja_core_news_sm",
)


@cl.on_chat_start
async def on_start_chat():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            max_size_mb=20,
            content="PDFを選択してください",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()
    file = files[0]

    documents = PyMuPDFLoader(file.path).load()

    splitted_documents = text_splitter.split_documents(documents)

    database = Chroma(embedding_function=embeddings)

    database.add_documents(splitted_documents)

    cl.user_session.set("database", database)

    await cl.Message(
        content=f"`{file.name}`の読み込みが完了しました。質問を入力してください。"
    ).send()


@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ: " + input_message.content)

    database = cl.user_session.get("database")

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
