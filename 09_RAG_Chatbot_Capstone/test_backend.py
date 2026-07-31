import os
import chromadb
import config
import ingest
from main import app

def test_rag_flow():
    print("=== Testing Ingestion ===")
    try:
        # Ingest the university dataset
        ingest.ingest_data()
        print("Ingestion completed successfully!\n")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return

    print("=== Testing Local ChromaDB Query ===")
    try:
        # Initialize client
        chroma_client = chromadb.PersistentClient(path=config.DB_DIR)
        embedding_fn = ingest.get_embedding_function()
        collection = chroma_client.get_collection(name=config.COLLECTION_NAME, embedding_function=embedding_fn)
        
        # Test Query
        test_query = "What is the minimum GPA requirement for undergraduate admission?"
        print(f"Query: '{test_query}'")
        
        results = collection.query(
            query_texts=[test_query],
            n_results=2
        )
        
        print("\nRetrieved Results:")
        for idx, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][idx]
            print(f"\n[{idx+1}] ID: {results['ids'][0][idx]} (Category: {meta.get('category')})")
            print(doc)
            
    except Exception as e:
        print(f"ChromaDB Query failed: {e}")

    print("\n=== Testing FastAPI Query Endpoint ===")
    try:
        from main import query_rag, QueryRequest
        req = QueryRequest(query="What is the GPA requirement and what scholarships are available?", top_k=2)
        response = query_rag(req)
        print(f"Response Mode: {response.get('mode')}")
        print("Answer output:")
        # Encode to avoid Windows console UnicodeEncodeError on emoji print
        answer_to_print = response.get("answer", "")
        print(answer_to_print.encode('ascii', errors='replace').decode('ascii'))
    except Exception as e:
        print(f"FastAPI Query Endpoint test failed: {e}")

if __name__ == "__main__":
    test_rag_flow()
