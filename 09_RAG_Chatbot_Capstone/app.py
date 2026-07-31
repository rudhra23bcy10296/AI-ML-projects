"""
RAG Chatbot Capstone - Interactive CLI / Web User Interface
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import os
from src.ingest import DocumentVectorStore
from src.rag_chain import RAGQuestionAnsweringChain


def run_chatbot_interface(doc_path=None):
    print("=" * 65)
    print(" Project 9: RAG Chatbot (Capstone Project)")
    print(" Student: Rudhra Sitholey | Reg: 23BCY10296 | App: IN26012560")
    print("=" * 65)
    
    vectorstore = DocumentVectorStore()
    
    if doc_path is None:
        doc_path = os.path.join("docs", "ai_ml_handbook.txt")
        
    print(f"\n[1] Initializing Vector Store Ingestion on Document: {doc_path}...")
    vectorstore.load_and_chunk_document(doc_path)
    
    rag_chain = RAGQuestionAnsweringChain(vectorstore)
    print("\n[2] RAG Chatbot Online! Type your queries below (or 'exit' to quit):")
    print("-" * 65)
    
    sample_queries = [
        "What is Supervised Learning?",
        "Explain Convolutional Neural Networks and image classification",
        "How does Proximal Policy Optimization work in RL?"
    ]
    
    for q in sample_queries:
        print(f"\nUser Query: {q}")
        response = rag_chain.answer_question(q)
        print(f"RAG Response:\n{response['answer']}")
        print(f"Citations: {', '.join(response['citations'])}")
        print("." * 40)


if __name__ == '__main__':
    run_chatbot_interface()
