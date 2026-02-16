import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import shutil

load_dotenv()

# Global variables to store the database and chain
vector_db = None
rag_chain = None

def load_documents(file_path):
    """Load documents from PDF or DOCX file."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("File type not supported. Please upload PDF or DOCX.")
    
    documents = loader.load()
    return documents

def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    return texts

def create_vector_db(text, persist_directory="chroma_db"):
    """Create a vector database from the given text."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(
        documents=text, 
        embedding=embedding_model, 
        persist_directory=persist_directory
    )
    return db

def load_vector_db(persist_directory="chroma_db"):
    """Load the existing vector database."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding_model
    )
    return db

def load_llm():
    """Initialize Groq LLM."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm

def create_rag_chain(vector_db, llm):
    """Create the RAG chain."""
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    system_prompt = (
        "You are a contract analysis AI assistant with conversation memory.\n"
        "Your task is to analyze legal contracts strictly using ONLY the provided context.\n\n"
        
        "Rules:\n"
        "1. Pay attention to the conversation history to understand follow-up questions.\n"
        "2. When the user refers to previous answers (e.g., 'what about that clause?'), use the conversation context.\n"
        "3. Do not rely on prior knowledge beyond the document.\n"
        "4. Do not infer missing information.\n"
        "5. If the answer is not explicitly stated, respond with:\n"
        "'I do not know based on the provided document.'\n\n"
        
        "Answer format:\n"
        "- For follow-up questions, acknowledge the previous context\n"
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
        return "\n\n".join(formatted)
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def process_document(file):
    """Process uploaded document and create vector database."""
    global vector_db, rag_chain
    
    try:
        if file is None:
            return "❌ Please upload a document first.", "", gr.update(selected=0)
        
        # Get the file path
        file_path = file.name
        
        # Remove existing database if it exists
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
        
        status = "📄 Loading document...\n"
        yield status, "", gr.update(selected=0)
        
        # Load and process the document
        docs = load_documents(file_path)
        status += f"✅ Loaded {len(docs)} page(s)\n"
        status += "✂️ Splitting into chunks...\n"
        yield status, "", gr.update(selected=0)
        
        chunks = split_documents(docs)
        status += f"✅ Created {len(chunks)} chunks\n"
        status += "🔮 Creating vector database...\n"
        yield status, "", gr.update(selected=0)
        
        vector_db = create_vector_db(chunks)
        status += "✅ Vector database created\n"
        status += "🤖 Initializing LLM...\n"
        yield status, "", gr.update(selected=0)
        
        llm = load_llm()
        rag_chain = create_rag_chain(vector_db, llm)
        
        status += "✅ RAG pipeline ready!\n"
        status += "💬 Switching to chat...\n"
        
        # Switch to chat tab (index 1)
        yield status, gr.update(interactive=True), gr.update(selected=1)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n"
        error_msg += "Please make sure you have set your GROQ_API_KEY in the .env file."
        yield error_msg, gr.update(interactive=False), gr.update(selected=0)

def chat_with_document(message, history):
    """Chat function for Gradio ChatInterface with memory."""
    global rag_chain, vector_db
    
    if rag_chain is None:
        return "⚠️ Please upload and process a document first using the 'Upload Document' tab."
    
    if not message.strip():
        return "Please enter a question."
    
    try:
        # Build conversation context from history
        conversation_context = ""
        if history and len(history) > 0:
            conversation_context = "Previous conversation:\n"
            # Handle different history formats
            for exchange in history[-3:]:  # Last 3 exchanges
                if isinstance(exchange, (list, tuple)) and len(exchange) >= 2:
                    user_msg = exchange[0]
                    assistant_msg = exchange[1]
                    conversation_context += f"User: {user_msg}\n"
                    conversation_context += f"Assistant: {assistant_msg}\n"
            conversation_context += "\n"
        
        # Combine context with current message
        full_query = conversation_context + f"Current question: {message}"
        
        # Get the response from RAG chain
        response = rag_chain.invoke(full_query)
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def show_retrieved_docs(query):
    """Show retrieved documents for transparency."""
    global vector_db
    
    if vector_db is None:
        return "⚠️ Please upload and process a document first."
    
    if not query.strip():
        return "Please enter a query."
    
    try:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        
        result = f"📚 Retrieved {len(docs)} relevant chunks:\n\n"
        for i, doc in enumerate(docs):
            result += f"--- Chunk {i+1} ---\n"
            result += doc.page_content[:500]
            if len(doc.page_content) > 500:
                result += "..."
            result += "\n\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio Interface
with gr.Blocks(title="Contract Analysis AI Assistant") as demo:
    
    gr.Markdown(
        """
        # 📄 Contract Analysis AI Assistant
        
        Upload your legal contract (PDF or DOCX) and ask questions about it!
        
        ### How to use:
        1. **Upload Document**: Go to the 'Upload Document' tab and upload your contract
        2. **Wait for Processing**: The system will process and index your document
        3. **Ask Questions**: Switch to the 'Chat' tab and start asking questions
        """
    )
    
    with gr.Tabs() as tabs:
        # Tab 1: Document Upload
        with gr.Tab("📤 Upload Document", id=0):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Contract (PDF or DOCX)",
                        file_types=[".pdf", ".docx"]
                    )
                    upload_btn = gr.Button("Process Document", variant="primary", size="lg")
                
                with gr.Column():
                    status_output = gr.Textbox(
                        label="Processing Status",
                        lines=10,
                        interactive=False
                    )
            
            upload_btn.click(
                fn=process_document,
                inputs=[file_input],
                outputs=[status_output, upload_btn, tabs]
            )
        
        # Tab 2: Chat Interface
        with gr.Tab("💬 Chat", id=1):
            chatbot = gr.ChatInterface(
                fn=chat_with_document,
                examples=[
                    "What are the key terms of this contract?",
                    "What is the termination clause?",
                    "What are the payment terms?",
                    "Who are the parties involved?",
                    "What are the obligations of each party?"
                ],
                title="Ask Questions About Your Contract",
                description="Ask any question about the uploaded contract. The AI remembers the conversation and can answer follow-up questions."
            )
        
        # Tab 3: Document Retrieval (Advanced)
        with gr.Tab("🔍 Document Retrieval (Advanced)", id=2):
            gr.Markdown(
                """
                ### See what the AI retrieves
                This tab shows which document chunks are retrieved for your query.
                Useful for understanding and debugging the AI's responses.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    retrieval_query = gr.Textbox(
                        label="Enter Query",
                        placeholder="What is the payment clause?",
                        lines=2
                    )
                    retrieval_btn = gr.Button("Show Retrieved Chunks", variant="secondary")
                
                with gr.Column():
                    retrieval_output = gr.Textbox(
                        label="Retrieved Document Chunks",
                        lines=15,
                        interactive=False
                    )
            
            retrieval_btn.click(
                fn=show_retrieved_docs,
                inputs=[retrieval_query],
                outputs=[retrieval_output]
            )
    
    gr.Markdown(
        """
        ---
        ### 📝 Notes:
        - The AI only answers based on the uploaded document
        - For best results, ask specific questions about clauses, terms, or parties
        - Processing time depends on document size
        - Make sure to set your `GROQ_API_KEY` in a `.env` file
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )