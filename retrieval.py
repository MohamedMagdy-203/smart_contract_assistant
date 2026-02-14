import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def load_vector_db():
    """
    Load the existing vector database from local directory
    """
    persist_directory = "chroma_db"
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    return db

def load_llm():
    """
    Intilaize the llm from huggingface
    """
    repo_id = "Qwen/Qwen2.5-7B-Instruct"
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=.1, max_new_tokens=512, huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    return llm

