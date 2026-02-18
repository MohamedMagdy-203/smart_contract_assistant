from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

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
    Initialize Groq LLM (Free & Super Fast!)
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    return llm

def create_rag_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})  
    
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
        "Previous Conversation History:{chat_history}"
        "Context:\n"
        "{context}\n\n"
        "Question: {input}"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(f"[Document {i+1}]\n{doc.page_content}")
        result = "\n\n".join(formatted)
        
        print("\n[DEBUG] Context sent to LLM:")
        print(result[:500])  
        print("...\n")
        
        return result
    
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs, "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain