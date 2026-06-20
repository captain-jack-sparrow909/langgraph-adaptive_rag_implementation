from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever

def retrive(state: GraphState)->Dict[str, Any]:
    print("--RETRIEVE--")
    question = state['question']
    docs = retriever.invoke(question)
    return {"documents": docs, "question": question}


