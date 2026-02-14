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
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Main Execution Block for Testing ---
if __name__ == "__main__":
    # Check if the vector database exists before running
    if not os.path.exists("chroma_db"):
        print("Error: 'chroma_db' directory not found.")
        print("Please run 'python ingestion.py' first to create the vector store.")
    else:
        print("--- Initializing RAG Pipeline ---")
        
        # 1. Load the Vector Database
        print("1. Loading Vector Store...")
        db = load_vector_db()
        
        # 2. Load the LLM (Qwen)
        print("2. Loading LLM (Qwen/Qwen2.5-7B-Instruct)...")
        llm = load_llm()
        
        # 3. Create the RAG Chain
        print("3. Creating Retrieval Chain...")
        chain = create_rag_chain(db, llm)
        
        print("\n--- RAG Pipeline Ready! (Type 'exit' to quit) ---")
        
        # Start a loop to ask questions interactively
        while True:
            query = input("\nUser Question: ")
            
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break
            
            if not query.strip():
                continue
            
            print("\nThinking...")
            
            # Invoke the chain with the user's query
            try:
                print(f"\n[Query]: {query}")
                
                retriever = db.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(query)
                
                print(f"\n[Retrieved {len(docs)} documents]")
                for i, doc in enumerate(docs):
                    print(f"\n--- Document {i+1} ---")
                    print(doc.page_content[:300])  
                    print("...")
                
                response = chain.invoke(query)
                
                print("-" * 50)
                print("AI Response:")
                print(response)
                print("-" * 50)
                
            except Exception as e:
                print(f"An error occurred: {e}")