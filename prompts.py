from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os
from pathlib import Path

abs_path = os.path.dirname(os.path.abspath(__file__))

env_path = os.path.join(abs_path, ".env")
dotenv_path = Path(env_path)
load_dotenv(dotenv_path=dotenv_path)

api_key = os.getenv("OPEN_API_KEY")

llm = ChatOpenAI(api_key=api_key,model="gpt-4o-mini", temperature=0)



### Retrieval Grader

retrieval_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_prompt | llm | JsonOutputParser()

### Hallucination Grader

hallucination_prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts.Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and NO preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = hallucination_prompt | llm | JsonOutputParser()
### Answer Grader



# Prompt
answer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = answer_prompt | llm | JsonOutputParser()
#generate answer
rag_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a helpful chatbot that can answer questions based on the provided context. 
    You need not use the entire context provided to you.
    When interpreting general questions, you can rephrase the content without altering its meaning.
    For questions that require steps, processes, or procedures, maintain the original content as closely as possible.
    If the context includes Red Hat specific knowledge, include that in your answer.
    Avoid repeating answers.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    ### If the question demands steps or a list of points, guide the model to respond accordingly:
    - Please provide the answer in points or a list format where appropriate.
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)



# Chain
rag_chain = rag_prompt | llm | StrOutputParser()

