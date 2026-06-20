from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ..., #means this field will be required while instantiating an object of this class
        description="Given a user question choose to route it to web search or a vectorstore."
        )

llm = ChatOpenAI(model="gpt-5-nano")
llm_structured_output = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}")
])

router_chain = prompt | llm_structured_output


