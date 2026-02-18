import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_documents(file_path):
    """
    Load the documents from the givin file path.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("File type not supported. Please upload PDF or DOCX.")
    
    documents = loader.load()
    return documents


def split_documents(documents):
    """
    Split the documents to chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200 , length_function = len)
    texts = text_splitter.split_documents(documents)
    return texts


def create_vector_db(text):
    """
    Create a vector database from the given text.
    """
    persist_directory = "chroma_db"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents=text, embedding=embedding_model, persist_directory=persist_directory)
    return db
