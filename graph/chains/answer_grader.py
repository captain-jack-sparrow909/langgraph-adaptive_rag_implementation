
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano")

class GradeAnswer(BaseModel):
    """Answer addresses the question or not"""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

system = """you are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "generation: {generation} \n question: {question}")
])

answer_grader_chain = prompt_template | llm.with_structured_output(GradeAnswer)
