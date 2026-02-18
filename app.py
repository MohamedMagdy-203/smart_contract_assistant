import os
from ingestion import load_documents, split_documents, create_vector_db
from retrieval import load_vector_db, load_llm, create_rag_chain
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient


vector_db = None
llm = None
rag_chain = None

@asynccontextmanager
async def lifespan(app : FastAPI):
    global vector_db, llm, rag_chain
    llm = load_llm()
    if os.path.exists("chroma_db"):
        vector_db = load_vector_db()
        rag_chain = create_rag_chain(llm, vector_db)
    else:
        vector_db = None
        rag_chain = None
    yield


import os
from ingestion import load_documents, split_documents, create_vector_db
from retrieval import load_vector_db, load_llm, create_rag_chain
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient


vector_db = None
llm = None
rag_chain = None

@asynccontextmanager
async def lifespan(app : FastAPI):
    global vector_db, llm, rag_chain
    llm = load_llm()
    if os.path.exists("chroma_db"):
        vector_db = load_vector_db()
        rag_chain = create_rag_chain(vector_db, llm)
    
    yield

app = FastAPI(lifespan= lifespan)

