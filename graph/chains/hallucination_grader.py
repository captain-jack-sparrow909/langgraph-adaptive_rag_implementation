from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class GradeHallucination(BaseModel):
     """Binary score for hallucination present in generation answer."""
     binary_score: str = Field(description="Answer is grounded in facts, 'yes' or 'no'")

llm = ChatOpenAI(model="gpt-5-nano")


system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


prompt_template = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts: {documents} \n LLM Generation: {generation}")
])

hallucination_grader_chain = prompt_template | llm.with_structured_output(GradeHallucination)
# because here we must want the binary_score to be returned that is why .with_structured_output()
# when not needed you can just use JsonOutputParser
