"""
RAG Chatbot Capstone - Document Ingestion & JSONL / Text Vectorstore Indexing Module
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentVectorStore:
    """
    Semantic chunker and vector retriever supporting both TXT handbooks and JSONL dataset files.
    """
    def __init__(self, chunk_size=200, chunk_overlap=30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def load_and_chunk_document(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at path: {file_path}")
            
        filename = os.path.basename(file_path)
        
        if filename.endswith(".jsonl"):
            # Load JSONL dataset records
            chunks = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        text = item.get('text') or item.get('content') or json.dumps(item)
                        chunks.append(text)
            self.chunks = chunks
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            raw_sections = [s.strip() for s in text.split("\n\n") if s.strip()]
            self.chunks = raw_sections
            
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        print(f"Indexed {len(self.chunks)} semantic dataset records from {filename}")

    def similarity_search(self, query, top_k=2):
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': self.chunks[idx],
                'score': round(float(sim_scores[idx]), 4),
                'chunk_id': idx + 1
            })
        return results
