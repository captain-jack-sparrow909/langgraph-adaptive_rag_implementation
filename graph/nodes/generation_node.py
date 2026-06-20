

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState):
    documents = state['documents']
    question = state['question']

    response = generation_chain.invoke({
        "context": documents, "question": question
    })

    return {"question": question, "documents": documents, "generation": response}