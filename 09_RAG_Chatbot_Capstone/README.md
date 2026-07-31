# Project 09 — RAG Chatbot Capstone

**Author:** Rudhra Sitholey
**Registration No:** 23BCY10296 | **Application No:** IN26012560
**Email:** rudhra.23bcy10296@vitbhopal.ac.in

---

## Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from custom documents. The system ingests documents, chunks them into passages, embeds them into a ChromaDB vector store, and retrieves the most relevant context for each user query. Retrieved passages are sent alongside the question to a Groq-hosted LLM (LLaMA / Mixtral) to generate grounded, citation-backed answers. The frontend is a polished single-page chat interface served via FastAPI.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM Provider | Groq Cloud (LLaMA 3, Mixtral) |
| Vector Database | ChromaDB (persistent local storage) |
| Embeddings | Sentence-level embeddings for semantic search |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML5, CSS3, Vanilla JavaScript (single-page chat UI) |
| Configuration | python-dotenv, Pydantic |

## Project Structure

```
09_RAG_Chatbot_Capstone/
├── app.py                # CLI chatbot interface with sample queries
├── main.py               # FastAPI web server with chat endpoints
├── config.py             # Environment configuration and model settings
├── ingest.py             # Document ingestion and ChromaDB population
├── test_backend.py       # Backend API tests
├── index.html            # Chat UI (served by FastAPI)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template (Groq API key)
├── Procfile              # Deployment process configuration
├── docs/
│   └── ai_ml_handbook.txt  # Default knowledge base document
├── data/                 # Additional data files
├── chroma_db/            # Persistent vector store
└── src/
    ├── ingest.py         # DocumentVectorStore — chunking + embedding
    └── rag_chain.py      # RAGQuestionAnsweringChain — retrieval + generation
```

## How It Works

1. **Document Ingestion** — Documents (`.txt`, `.pdf`, `.md`) are loaded, split into overlapping chunks (~500 tokens), and embedded into ChromaDB.
2. **Semantic Search** — User queries are embedded and matched against the vector store using cosine similarity. The top-k most relevant passages are retrieved.
3. **Answer Generation** — Retrieved passages + the user question are formatted into a prompt and sent to a Groq-hosted LLM (LLaMA 3 / Mixtral). The model generates a grounded answer.
4. **Citations** — Each answer includes references to the source document chunks used for generation.

## Getting Started

### 1. Set Up Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key
# Get your key at: https://console.groq.com/keys
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ingest Documents

```bash
python ingest.py
# → Processes docs/ folder and populates ChromaDB
```

### 4. Launch the Web Application

```bash
python main.py
# → Open http://localhost:8000
```

### 5. (Optional) Run the CLI Chatbot

```bash
python app.py
# → Interactive terminal chatbot with sample queries
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/api/chat` | Send a message, receive RAG-generated answer |
| `GET` | `/api/health` | Health check |

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | API key from [Groq Console](https://console.groq.com/keys) | ✅ Yes |

## Features

- **Grounded answers** — Responses are always backed by retrieved document context
- **Citation tracking** — Every answer shows which source chunks were used
- **Persistent vector store** — ChromaDB stores embeddings locally, no re-ingestion needed
- **Custom documents** — Drop any `.txt` file into `docs/` and re-run ingestion
- **Modern chat UI** — Clean, responsive single-page interface

