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

def create_rag_chain(vector_db, llm):
    """
    Create RAG Chain
    """
    retriever = vector_db.as_retriever(search_kwargs={"k":3})

    system_prompt = (
    "You are a contract analysis AI assistant.\n"
    "Your task is to analyze legal contracts strictly using ONLY the provided context.\n\n"
    
    "Rules:\n"
    "1. Do not rely on prior knowledge.\n"
    "2. Do not infer missing information.\n"
    "3. Do not summarize unless asked.\n"
    "4. If the answer is not explicitly stated, respond exactly with:\n"
    "'I do not know based on the provided document.'\n\n"
    
    "Answer format:\n"
    "- Relevant Clause: (quote exact text)\n"
    "- Location: (section/clause number if available)\n"
    "- Explanation: (brief legal explanation based only on text)\n\n"
    
    "Context:\n"
    "{context}"
)

    prompt = ChatPromptTemplate.from_messages([('system',system_prompt),('human','{input}')])
    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever,question_answer_chain)

    return rag_chain
