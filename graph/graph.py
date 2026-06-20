from dotenv import load_dotenv
from graph.chains.answer_grader import answer_grader_chain
from graph.chains.hallucination_grader import hallucination_grader_chain
from graph.const import GENERATE, GRADE_DOCUMENTS, WEB_SEARCH, RETRIEVE
from langgraph.graph import END, StateGraph
from graph.nodes import generate, grade_documents, retrive, web_search
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state: GraphState):
    print("---Assess the Documents--")
    if state['web_search']:
        return WEB_SEARCH
    else :
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState):
    question = state['question']
    documents = state['documents']
    generation = state['generation']
    score = hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
    hallucination_binary_score = score.binary_score

    if hallucination_binary_score.lower() == "yes":
        print("documents are based on facts")
        score_answer = answer_grader_chain.invoke({"question": question, "generation": generation})
        answer_grader_binary_score = score_answer.binary_score
        if answer_grader_binary_score.lower() == 'yes':
            print("generation is based of question")
            return "useful"
        elif answer_grader_binary_score.lower() == 'no':
            print("answer ins't grounded in the question")
            return "not useful"
    else:
        print("documents isn't based of the question")
        return "not supported"
        



workflow = StateGraph(GraphState)
workflow.add_node(GENERATE, generate)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(RETRIEVE, retrive)

workflow.set_entry_point(RETRIEVE)

workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {
    WEB_SEARCH: WEB_SEARCH, 
    GENERATE: GENERATE
})

workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question, {
    "not supported": GENERATE,
    "not useful": WEB_SEARCH,
    "useful": END
})

workflow.add_edge(RETRIEVE, GENERATE)
workflow.add_edge(WEB_SEARCH, GENERATE)

workflow.add_edge(GENERATE, END)

graph = workflow.compile()
