from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

# embeddings = HuggingFaceEmbeddings(
#                 model_name="BAAI/bge-large-en-v1.5")
embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs = {'device': 'cpu'})
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def embedd_and_store(file_path):
    # file_path = "./static/uploads/sample_ai.pdf"

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    print(len(docs))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    print(len(all_splits))

   
    # 
    # embeddings = OpenAIEmbeddings(api_key = api_key,model="text-embedding-3-small")

    qdrant_vector_store = QdrantVectorStore.from_documents(
        all_splits,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="medium-qdrnt-data",
    )


    qdrnt_retriever = qdrant_vector_store.as_retriever(search_kwargs={"score_threshold": 0.5,"k": 1})
    print("qdrnt_retriever: ",qdrnt_retriever)
    return qdrnt_retriever