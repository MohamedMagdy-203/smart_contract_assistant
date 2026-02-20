# Smart Contract Assistant

---
## DEMO


https://github.com/user-attachments/assets/ab3a4a4e-acdb-4696-bb40-b697df7a92b7

## Evaluation_demo


https://github.com/user-attachments/assets/2a1c3448-dcd3-4082-be5f-4cf54487c49d


---

## Project Overview

The Smart Contract Assistant is a web application that allows users to upload legal contracts (PDF or DOCX) and interact with them through a conversational AI interface. The system uses Retrieval-Augmented Generation (RAG) to extract, chunk, and embed document content into a vector store, then uses a large language model to answer questions based strictly on what the document says. No hallucinations, no external knowledge, only the contract itself.

---

## Project Specification vs. What Was Built

The following table maps each requirement from the project specification to its implementation status.

| Requirement | Status | Details |
|---|---|---|
| File ingestion: PDF and DOCX | Done | ingestion.py uses PyPDFLoader and Docx2txtLoader |
| Text chunking and embedding | Done | RecursiveCharacterTextSplitter with 1000 char chunks and 200 overlap, embedded via sentence-transformers/all-MiniLM-L6-v2 |
| Vector store setup | Done | ChromaDB with local persistence in chroma_db/ directory |
| Semantic retrieval | Done | Top 5 most relevant chunks retrieved per query |
| LLM-based Q&A with source grounding | Done | Groq API using Llama 3.3 70B, answers strictly from context |
| Chat history | Done | chat_history_buffer in app.py carries previous turns into each prompt |
| Guard-rails for factuality | Done | System prompt enforces citation-only answers and explicit "I do not know" fallback |
| Contract summarization | Not implemented | Listed as optional in the specification |
| FastAPI backend | Done | app.py exposes /upload and /ask endpoints |
| Gradio frontend | Done | ui.py communicates with the FastAPI backend via HTTP |
| Evaluation pipeline | Partially implemented | See Evaluation section below |

---

## Technology Stack

- LangChain: Framework for building the RAG pipeline and chaining components
- ChromaDB: Local vector database for storing and querying document embeddings
- HuggingFace Embeddings: sentence-transformers/all-MiniLM-L6-v2 for converting text to vectors
- Groq API: Fast inference using Llama 3.3 70B Versatile
- FastAPI: Backend REST API for file upload and question answering
- Gradio: Frontend web interface for upload and chat
- PyPDF and python-docx: Document loading and parsing

---

## Project Structure

```
contract-analysis-ai/
    app.py              Main FastAPI application with /upload and /ask endpoints
    ingestion.py        Document loading, chunking, and vector DB creation
    retrieval.py        RAG chain, retrieval logic, and LLM initialization
    ui.py               Gradio frontend that communicates with the FastAPI backend
    requirements.txt    All Python dependencies with pinned versions
    .env                Environment variables (not committed to version control)
    README.md           This file
```

---

## Prerequisites

- Python 3.8 or higher
- A Groq API key (free at https://console.groq.com)

---

## Installation

Step 1: Clone or download the project files to your local machine.

Step 2: Create and activate a virtual environment.

On Windows:
```
python -m venv venv
venv\Scripts\activate
```

On macOS and Linux:
```
python -m venv venv
source venv/bin/activate
```

Step 3: Install all dependencies.

```
pip install -r requirements.txt
```

Step 4: Create a .env file in the project root with your Groq API key.

```
GROQ_API_KEY=your_groq_api_key_here
```

To get your API key, go to https://console.groq.com, sign up, navigate to API Keys, and create a new key.

---

## How to Run

You need to run two processes: the FastAPI backend and the Gradio frontend.

Terminal 1 - Start the backend:
```
uvicorn app:app --reload --port 8000
```

Terminal 2 - Start the frontend:
```
python ui.py
```

Then open your browser at http://127.0.0.1:7860

---

## How to Use

Step 1: Go to the interface in your browser.

Step 2: Upload a PDF or DOCX contract using the file input at the top. The system will process it and store it in the vector database.

Step 3: Ask questions in the chat box. The assistant will respond with the relevant clause, its location in the document, and a brief explanation based strictly on the contract text.

Example questions you can ask:
- What are the obligations of the first party?
- What is the commission rate for each area?
- What happens if the second party delivers cash late?
- How are disputes resolved?
- How many days notice is required to terminate the agreement?

---

## How It Works

Document Processing Pipeline:

When you upload a file, the system loads the document using the appropriate loader (PDF or DOCX), splits it into overlapping chunks of 1000 characters, converts each chunk into a vector embedding using the HuggingFace model, and stores all embeddings in ChromaDB on disk.

Question Answering Pipeline:

When you ask a question, the query is converted to an embedding, the top 5 most semantically similar chunks are retrieved from ChromaDB, those chunks are assembled into a context block, and the question plus context plus chat history are sent to the Groq LLM. The model responds with a structured answer that includes a direct quote, a location reference, and a brief explanation.

The system prompt enforces three strict rules: the model must only use the provided context, it must not infer or guess missing information, and if the answer is not in the document it must respond with exactly "I do not know based on the provided document."

---

## Evaluation

The project includes a basic evaluation approach to measure retrieval and answer quality.

Retrieval Evaluation:

You can test retrieval quality manually using the vector database directly. Load the DB and run similarity searches to check whether the top-k chunks returned are relevant to your query. A well-functioning retriever should return the clause that contains the answer within the top 3 results.

Answer Quality Evaluation:

The following test cases were run against the sample_contract.pdf included in this repository. Each question was asked, and the answer was evaluated against the ground truth from the document.

| Question | Expected Answer | Result |
|---|---|---|
| What is the objective of the agreement? | Transfer money from Turkey into Syria for humanitarian activities | Correct - cited Article 1 |
| How many days in advance must the first party inform the second party of a transfer? | 7 days before the due date | Correct - cited Article 2.3 |
| What commission rate applies to Sarmada? | 0.6% | Correct - cited Article 3.4 |
| What happens if cash is delivered more than 24 hours late? | The exchange rate is revised at the time of delivery | Correct - cited Article 3.7 |
| What is the termination notice period? | Two weeks written notice | Correct - cited Article 5.2 |
| How are disputes resolved? | First through friendly consultation, then a mutually agreed mediator on a cost-sharing basis | Correct - cited Article 6 |
| What is the contract duration? | Blank, to be filled at signing | Correct - cited Article 4.1 |
| What currency is payment made in? | USD or Euro based on a pre-agreed exchange rate | Correct - cited Article 2.8 |

Memory-Based Follow-Up Test (Chat History):

After asking "What is the commission rate for Sarmada?" the user then asked "What about the other areas?" without repeating context. The system correctly used the chat history buffer to understand that the follow-up referred to commission rates in Article 3.4 and returned the rates for the other areas (xxx: 1.0% and xxx: 2.5%).

Limitations:

The evaluation is manual and qualitative. There is no automated scoring pipeline such as RAGAS or TruLens integrated yet. Adding an automated evaluation framework would be a natural next step for production use.

---

## Known Limitations

- Only PDF and DOCX formats are supported
- One document is processed at a time; uploading a new file replaces the previous vector database
- No image analysis; only text-based contracts are supported
- Requires an active internet connection for the Groq API
- The evaluation pipeline is manual; no automated metrics are computed

---

## Security Notes

- Never commit your .env file to version control
- The application runs locally by default and does not send data to any external service except the Groq API for inference
- Keep your Groq API key confidential

---

## Future Enhancements

- Multi-document search and comparison
- Automated evaluation with RAGAS metrics
- Export answers to PDF or DOCX
- Multi-language contract support
- Cloud deployment with Docker

---

## Troubleshooting

GROQ_API_KEY not found: Make sure your .env file exists in the project root and contains the key with no extra spaces.

File type not supported: Only .pdf and .docx extensions are accepted.

Backend not running: Make sure you started uvicorn before launching the Gradio UI.

Slow first run: The HuggingFace embedding model downloads on first use. Subsequent runs are faster.

---

## Credits


Built with LangChain, ChromaDB, HuggingFace Sentence Transformers, Groq API, FastAPI, and Gradio.

