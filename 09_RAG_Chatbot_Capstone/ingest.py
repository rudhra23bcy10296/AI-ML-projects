import os
import json
import chromadb
from chromadb.utils import embedding_functions
import config

def load_dataset(file_path: str):
    """Load the JSON or JSONL FAQ dataset dynamically."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
    # Check if JSONL
    if file_path.lower().endswith(".jsonl"):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
        
    # Try normal JSON first, fallback to JSONL parsing if it fails
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON list of items.")
        return data
    except json.JSONDecodeError:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except Exception:
                        pass
        if data:
            return data
        raise

def get_embedding_function():
    """Get lightweight ONNX embedding function for low-memory deployment (<50MB RAM)."""
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        print("Using ChromaDB lightweight ONNXMiniLM_L6_V2 embedding function...")
        return ONNXMiniLM_L6_V2()
    except Exception as e:
        print(f"Fallback to default embedding function: {e}")
        return embedding_functions.DefaultEmbeddingFunction()

def ingest_data(file_path: str = None):
    if file_path is None:
        file_path = config.DATASET_PATH
        
    print(f"Loading dataset from: {file_path}")
    dataset = load_dataset(file_path)
    
    # Initialize Chroma Client
    db_dir = config.DB_DIR
    print(f"Initializing ChromaDB in directory: {db_dir}")
    chroma_client = chromadb.PersistentClient(path=db_dir)
    
    # Get embedding function
    embedding_fn = get_embedding_function()
    
    # Get or create collection
    collection_name = config.COLLECTION_NAME
    # If collection exists, delete it first to ensure clean ingestion
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass  # Collection didn't exist or deletion was skipped
        
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, item in enumerate(dataset):
        # Extract fields dynamically to support any domain
        # Check for standard 'question' and 'answer' fields
        question = item.get("question") or item.get("q") or ""
        answer = item.get("answer") or item.get("a") or item.get("text") or item.get("content") or ""
        
        # If neither is found, serialize the whole item
        if not question and not answer:
            document_content = json.dumps(item)
            question = f"Item {idx}"
            answer = document_content
        else:
            document_content = f"Question: {question}\nAnswer: {answer}"
            
        item_id = item.get("id") or item.get("uuid") or f"FAQ-{idx:03d}"
        category = item.get("category") or item.get("cat") or "General"
        
        documents.append(document_content)
        metadatas.append({
            "id": item_id,
            "category": category,
            "question": question,
            "answer": answer
        })
        ids.append(item_id)
        
    print(f"Indexing {len(documents)} items into ChromaDB...")
    
    # Add in batches of 100 to handle large datasets safely
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"Indexed batch {i} to {end_idx}")
        
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    ingest_data()
