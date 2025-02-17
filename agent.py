import os
from pprint import pprint
from typing import List, Dict, Any
from typing_extensions import TypedDict
from prompts import retrieval_grader,hallucination_grader,answer_grader,rag_chain


os.environ["LANGCHAIN_API_KEY"] = "876ee77b782f4c959618c48cdcb1f1b7"


# embeddings = HuggingFaceEmbeddings(
#             model_name="BAAI/bge-large-en-v1.5")



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    not_relevant: str
    generation: str
    documents: List[str]
    retrieval_model: Any
    collection_name : str
    qdrnt_retriever : str
    


### Nodes


def service_retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    # client = QdrantClient(path="/tmp/langchain_qdrant")

    question = state["question"]
    qdrnt_retriever = state["qdrnt_retriever"]
    # collection_name = state["collection_name"]
    # qdrant_vector_store = QdrantVectorStore(
    #     client=client,
    #     collection_name=collection_name,
    #     embedding=embeddings,
    # )
    # qdrnt_retriever = qdrant_vector_store.as_retriever(search_kwargs={"score_threshold": 0.5,"k": 1})

    # Step 3: Ensure Collection Exists with Correct Vector Size

    # Retrieval
    documents = qdrnt_retriever.invoke(question)
    return {"documents": documents, "question": question, "retrieval_model":qdrnt_retriever}


def grade_documents(state):
    
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to end

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    # generation = state["generation"]
    
    # Score each doc
    filtered_docs = []
    not_relevant = "Yes"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            not_relevant = "No"
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            not_relevant = "Yes"
            # generation = "Sorry, I couldn't find relevant information in the document."
            continue
    return {"documents": filtered_docs, "question": question, "not_relevant": not_relevant,"generation" : "Sorry, I couldn't find relevant information in the document."}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}



## Conditional edge

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or end

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    not_relevant = state["not_relevant"]
    state["documents"]

    if not_relevant == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, END ---"
        )
        return "end"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

### Conditional edge


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    # Initialize custom dictionary if it does not exist

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    print("score",score)
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            state["generation"] = f"Sorry, I couldn't fully answer the question. However, here's what I found: {generation}"
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        state["generation"] = "I could not find relevant information in the document. Please check and try again."
        return "not supported"
    



