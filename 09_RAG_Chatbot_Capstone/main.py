import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions

import config
import ingest

app = FastAPI(
    title="Domain-Agnostic RAG Chatbot Backend",
    description="A FastAPI RAG backend that can dynamically adapt to any FAQ dataset.",
    version="1.0.0"
)

# In-memory flag to ensure database is ready
db_ready = False

# Initialize ChromaDB client and embedding function
try:
    chroma_client = chromadb.PersistentClient(path=config.DB_DIR)
    embedding_fn = ingest.get_embedding_function()
except Exception as e:
    print(f"Error initializing database: {e}")
    chroma_client = None
    embedding_fn = None

def get_groq_client():
    groq_key = config.get_groq_api_key()
    if groq_key:
        try:
            from groq import Groq
            return Groq(api_key=groq_key)
        except Exception as e:
            print(f"Failed to initialize Groq client: {e}")
    return None

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message author: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The student/user query to answer")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Previous conversation history")
    top_k: Optional[int] = Field(2, description="Number of context documents to retrieve")

class IngestRequest(BaseModel):
    dataset_path: Optional[str] = Field(None, description="Path to custom FAQ JSON file. If None, uses configured default.")

class StatusResponse(BaseModel):
    dataset_path: str
    chatbot_domain: str
    welcome_message: str
    collection_name: str
    llm_configured: bool
    llm_provider: str
    db_ready: bool

def ensure_db():
    """Checks if the ChromaDB collection is populated. If not, auto-ingests data."""
    global db_ready
    if db_ready:
        return
        
    if not chroma_client:
        raise HTTPException(status_code=500, detail="Database client not initialized.")
        
    try:
        # Check if collection exists and has documents
        collection = chroma_client.get_collection(name=config.COLLECTION_NAME, embedding_function=embedding_fn)
        if collection.count() > 0:
            db_ready = True
            return
    except Exception:
        # Collection does not exist or has issues
        pass
        
    # Auto-ingest
    try:
        print("ChromaDB collection not found or empty. Performing automatic ingestion of default dataset...")
        ingest.ingest_data()
        db_ready = True
    except Exception as e:
        print(f"Failed to auto-ingest default dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Database not initialized and auto-ingestion failed: {e}")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the premium dynamic RAG chatbot frontend."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="index.html not found on server")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/status", response_model=StatusResponse)
def get_status():
    """Retrieve the current configuration status of the RAG engine."""
    try:
        ensure_db()
    except Exception as e:
        print(f"ensure_db failed during status check: {e}")
        
    has_docs = False
    if chroma_client:
        try:
            col = chroma_client.get_collection(name=config.COLLECTION_NAME, embedding_function=embedding_fn)
            has_docs = col.count() > 0
        except Exception:
            pass
            
    active_provider = "groq" if get_groq_client() else "ollama"
    return {
        "dataset_path": config.DATASET_PATH,
        "chatbot_domain": config.get_chatbot_domain(),
        "welcome_message": config.get_welcome_message(),
        "collection_name": config.COLLECTION_NAME,
        "llm_configured": True,
        "llm_provider": active_provider,
        "db_ready": has_docs
    }

@app.post("/ingest")
def trigger_ingest(request: IngestRequest):
    """Trigger ingestion of a new dataset dynamically."""
    global db_ready
    dataset_to_load = request.dataset_path or config.DATASET_PATH
    
    if not os.path.exists(dataset_to_load):
        raise HTTPException(status_code=404, detail=f"Dataset file not found at: {dataset_to_load}")
        
    try:
        # Temporarily update config dataset path if customized
        if request.dataset_path:
            config.DATASET_PATH = request.dataset_path
            
        ingest.ingest_data(dataset_to_load)
        db_ready = True
        return {
            "status": "success",
            "message": f"Successfully ingested and indexed dataset from {dataset_to_load}",
            "domain": config.get_chatbot_domain()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")



def condense_query(query: str, history: List[ChatMessage]) -> str:
    """Condense chat history and follow-up question into a standalone query."""
    chitchat_keywords = {
        "hey", "hi", "hello", "good morning", "good evening", "good afternoon",
        "thanks", "thank you", "thanks!", "thank you!", "thats good", "that's good",
        "okay", "ok", "okay thats good", "ok thanks", "great", "awesome", "cool",
        "got it", "perfect", "nice", "sounds good", "alright", "super", "wonderful", "good"
    }
    cleaned_query = query.strip().lower().rstrip(".!?,")
    if cleaned_query in chitchat_keywords or (len(cleaned_query.split()) <= 3 and any(w in cleaned_query for w in ["good", "ok", "okay", "thanks", "great", "cool", "nice", "hello", "hey", "hi"])):
        return query

    if not history:
        return query

    # Limit history to immediate active context (last 4 messages / 2 turns)
    recent_history = history[-4:]
    history_lines = []
    for msg in recent_history:
        role_label = "User" if msg.role == "user" else "Assistant"
        history_lines.append(f"{role_label}: {msg.content}")
    history_text = "\n".join(history_lines)

    prompt = (
        "Given the conversation history and a user message:\n"
        "- If the user message is a simple greeting, thank you, acknowledgement, or general praise (e.g. 'hello', 'thanks', 'thats good', 'okay'), return it as-is without rephrasing.\n"
        "- If the user message is a follow-up question requiring context from history, rephrase it into a standalone search question.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"User Message: {query}\n\n"
        "Output:"
    )

    # Try Groq rephrasing if available
    g_client = get_groq_client()
    if g_client:
        try:
            completion = g_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            condensed = completion.choices[0].message.content.strip().strip('"\'')
            if condensed:
                print(f"Condensation (Groq): '{query}' -> '{condensed}'")
                return condensed
        except Exception as e:
            print(f"Groq condensation failed: {e}.")

    # Fallback: combine
    last_user_msg = ""
    for msg in reversed(history):
        if msg.role == "user":
            last_user_msg = msg.content
            break
    if last_user_msg:
        combined = f"{last_user_msg} {query}"
        print(f"Condensation (Local Fallback Combine): '{query}' -> '{combined}'")
        return combined
        
    return query

@app.post("/query")
def query_rag(request: QueryRequest):
    """Query the RAG engine with a question, retrieving context and generating an answer."""
    ensure_db()
    
    # 1. Condense the query if chat history exists
    search_query = condense_query(request.query, request.history)
    
    try:
        collection = chroma_client.get_collection(name=config.COLLECTION_NAME, embedding_function=embedding_fn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to access database collection: {e}")
        
    # 2. Retrieve most relevant contexts from ChromaDB using rephrased query
    results = collection.query(
        query_texts=[search_query],
        n_results=request.top_k
    )
    
    retrieved_contexts = []
    context_strings = []
    if results and results["documents"] and len(results["documents"][0]) > 0:
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        ids = results["ids"][0]
        for idx, doc in enumerate(docs):
            context_strings.append(doc)
            meta = metadatas[idx]
            retrieved_contexts.append({
                "id": ids[idx],
                "category": meta.get("category", "General"),
                "question": meta.get("question", ""),
                "answer": meta.get("answer", "")
            })
            
    if not context_strings:
        return {
            "answer": "I am sorry, but I do not have any information in my knowledge base that could help answer your question.",
            "retrieved_contexts": [],
            "mode": "empty_db"
        }
        
    # Construct context block
    context_block = ""
    for idx, ctx in enumerate(context_strings):
        context_block += f"[Context {idx+1}]\n{ctx}\n\n"
        
    system_prompt = config.get_system_prompt().format(context=context_block)
    
    # Build prompt with chat history if present
    if request.history:
        history_lines = []
        for msg in request.history:
            role_label = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg.content}")
        history_text = "\n".join(history_lines)
        
        prompt = (
            f"{system_prompt}\n"
            f"Conversation History:\n{history_text}\n\n"
            f"User: {request.query}\n"
            "Assistant:"
        )
    else:
        prompt = f"{system_prompt}\nUser Question: {request.query}\nAnswer:"

    # 4. Answer generation using Groq API
    g_client = get_groq_client()
    if g_client:
        try:
            completion = g_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer_text = completion.choices[0].message.content.strip()
            return {
                "answer": answer_text,
                "retrieved_contexts": retrieved_contexts,
                "mode": "groq"
            }
        except Exception as e:
            print(f"Groq LLM generation failed: {e}. Falling back to local retrieval.")

    # 6. Local Retrieval Fallback Mode
    fallback_answers = []
    for item in retrieved_contexts:
        fallback_answers.append(
            f"**Q: {item['question']}**\n"
            f"A: {item['answer']} *(Category: {item['category']}, ID: {item['id']})*"
        )
        
    disclaimer = (
        "⚠️ *Running in Local Retrieval Fallback Mode (Cloud & Local LLM providers were unavailable).* \n"
        "Here are the most relevant answers found in the database:\n\n"
    )
    answer_text = disclaimer + "\n\n---\n\n".join(fallback_answers)
    
    return {
        "answer": answer_text,
        "retrieved_contexts": retrieved_contexts,
        "mode": "retrieval_fallback"
    }

async def event_generator(query: str, history: List[ChatMessage], top_k: int):
    # Ensure database is loaded
    ensure_db()
    
    # 1. Condense the query if chat history exists
    search_query = condense_query(query, history)
    
    # 2. Query ChromaDB
    try:
        collection = chroma_client.get_collection(name=config.COLLECTION_NAME, embedding_function=embedding_fn)
        results = collection.query(query_texts=[search_query], n_results=top_k)
    except Exception as e:
        yield json.dumps({"type": "error", "data": f"Database query failed: {e}"}) + "\n"
        return

    retrieved_contexts = []
    context_strings = []
    if results and results["documents"] and len(results["documents"][0]) > 0:
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        ids = results["ids"][0]
        for idx, doc in enumerate(docs):
            context_strings.append(doc)
            meta = metadatas[idx]
            retrieved_contexts.append({
                "id": ids[idx],
                "category": meta.get("category", "General"),
                "question": meta.get("question", ""),
                "answer": meta.get("answer", "")
            })

    # Yield contexts metadata first
    yield json.dumps({"type": "contexts", "data": retrieved_contexts}) + "\n"
    # Yield provider info
    g_client = get_groq_client()
    active_mode = "groq" if g_client else "ollama"
    yield json.dumps({"type": "provider", "data": active_mode}) + "\n"

    if not context_strings:
        yield json.dumps({"type": "token", "data": "I am sorry, but I do not have any information in my knowledge base to answer your question."}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
        return

    # Construct prompt
    context_block = ""
    for idx, ctx in enumerate(context_strings):
        context_block += f"[Context {idx+1}]\n{ctx}\n\n"
        
    system_prompt = config.get_system_prompt().format(context=context_block)
    
    # Build prompt with chat history if present (limit to last 2 messages / 1 turn to save tokens)
    if history:
        recent_history = history[-2:]
        history_lines = []
        for msg in recent_history:
            role_label = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg.content}")
        history_text = "\n".join(history_lines)
        
        prompt = (
            f"{system_prompt}\n"
            f"History:\n{history_text}\n\n"
            f"User: {query}\n"
            "Assistant:"
        )
    else:
        prompt = f"{system_prompt}\nUser: {query}\nAssistant:"

    # 3. Answer generation using Groq stream (with max_tokens=300 to preserve token limit)
    if g_client:
        try:
            stream = g_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield json.dumps({"type": "token", "data": content}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            return
        except Exception as e:
            print(f"Groq streaming failed: {e}. Falling back to local retrieval.")

    # 5. Fallback to Local search if LLMs failed/unreachable
    fallback_answers = []
    for item in retrieved_contexts:
        fallback_answers.append(
            f"**Q: {item['question']}**\n"
            f"A: {item['answer']} *(Category: {item['category']}, ID: {item['id']})*"
        )
    disclaimer = (
        "⚠️ *Running in Local Retrieval Fallback Mode (Ollama service was unavailable).* \n"
        "Here are the most relevant answers found in the database:\n\n"
    )
    answer_text = disclaimer + "\n\n---\n\n".join(fallback_answers)
    
    # Send word-by-word with a tiny delay to simulate a real streaming effect for the fallback
    import asyncio
    words = answer_text.split(" ")
    for i in range(0, len(words), 3):
        chunk_words = " ".join(words[i:i+3]) + (" " if i+3 < len(words) else "")
        yield json.dumps({"type": "token", "data": chunk_words}) + "\n"
        await asyncio.sleep(0.02)
        
    yield json.dumps({"type": "done"}) + "\n"

@app.post("/query/stream")
def query_rag_stream(request: QueryRequest):
    """Query the RAG engine and stream the response word-by-word with dynamic history."""
    return StreamingResponse(
        event_generator(request.query, request.history, request.top_k), 
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
