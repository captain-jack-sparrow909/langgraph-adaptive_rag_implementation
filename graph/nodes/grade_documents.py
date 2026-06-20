from graph.chains.retrieval_grader import retrieval_grader_chain
from typing import Dict, Any

from graph.state import GraphState

def grade_documents(state: GraphState)->Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    filtered_docs = []
    web_search = False
    question = state['question']
    documents = state['documents']

    for doc in documents: # doc is of type Document
        response = retrieval_grader_chain.invoke({"question": question, "document": doc.page_content})  #since it'll return pydantic object
        if response.binary_score.lower() is "yes":
            print("Document is relevant")
            filtered_docs.append(doc)
        elif response.binary_score.lower() is "no":
            print("Document is not relevant")
            web_search = True
            continue
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


    # Loader → [Document, Document, ...]
    #        ↓
    #   TextSplitter → [smaller Document chunks]
    #        ↓
    #   Embeddings + VectorStore (stores page_content, indexes metadata)
    #        ↓
    #   Retriever → returns relevant [Document, Document, ...]
    #        ↓
    #   LLM chain (uses page_content as context)
    

