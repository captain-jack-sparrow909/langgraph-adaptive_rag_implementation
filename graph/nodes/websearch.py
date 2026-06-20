from typing import Any, Dict
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

from graph.state import GraphState

search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState)->Dict[str, Any]:
    print("--WEB Search")
    question = state['question']
    documents = state['documents']

    tavily_search = search_tool.invoke(input=question)  # it'll return a array of dict where each dict has content key
    joined_tavily_result = "\n".join([tv_result['content'] for tv_result in tavily_search]) # making it 1 long string
    web_results = Document(page_content=joined_tavily_result)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
    



