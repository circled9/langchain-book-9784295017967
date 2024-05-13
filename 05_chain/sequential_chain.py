import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv("../.env")
API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    api_key=API_KEY,
    model="gpt-3.5-turbo",
)

input = {"input": RunnablePassthrough()}

output_parser = StrOutputParser()

write_article_prompt = PromptTemplate(
    template="{input}についての記事を書いてください。",
    input_variables=["input"],
)
write_article_chain = input | write_article_prompt | chat | output_parser

translate_prompt = PromptTemplate(
    template="以下の文章を英語に翻訳してください。\n{input}",
    input_variables=["input"],
)
translate_chain = input | translate_prompt | chat | output_parser

chain = write_article_chain | translate_chain

print(
    chain.invoke(
        {
            "input": "エレキギターの選び方",
        }
    )
)
