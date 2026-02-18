import os
from ingestion import load_documents, split_documents, create_vector_db
from retrieval import load_vector_db, load_llm, create_rag_chain
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient
import shutil
from fastapi import exceptions
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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_db, rag_chain
    file_path = file.filename

    try:
        with open(file_path, "wb")as buffer:
            shutil.copyfileobj(file.file, buffer)
        docs = load_documents(file_path)
        chunks = split_documents(docs)
        vector_db = create_vector_db(chunks)
        rag_chain = create_rag_chain(vector_db, llm)
        return JSONResponse(content={"status": "success", "message": f"File '{file_path}' processed and vector DB updated."})
    except Exception as e :
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)