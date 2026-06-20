from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        prompt
    )
])

llm = ChatOpenAI(model="gpt-5-nano")

generation_chain = prompt_template | llm | StrOutputParser()
