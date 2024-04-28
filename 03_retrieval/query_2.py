import os
from dotenv import load_dotenv
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

database = Chroma(persist_directory="./.data", embedding_function=embeddings)

query = "飛行機の最高速度は？"

documents = database.similarity_search(query)

documents_string = ""

for document in documents:
    documents_string += f"""
---------------------------
{document.page_content}
"""

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。
文章:
{document}

質問: {query}
""",
    input_variables=["document", "query"],
)

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

result = chat.invoke(
    [HumanMessage(content=prompt.format(document=documents_string, query=query))]
)

print(result.content)
