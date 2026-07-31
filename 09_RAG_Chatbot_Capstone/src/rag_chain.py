"""
RAG Chatbot Capstone - Retrieval-Augmented Generation Synthesis Chain
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

class RAGQuestionAnsweringChain:
    """
    RAG QA Synthesizer combining retrieved context with source attribution.
    """
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def answer_question(self, user_query, top_k=2):
        retrieved_docs = self.vectorstore.similarity_search(user_query, top_k=top_k)
        
        if not retrieved_docs or retrieved_docs[0]['score'] == 0.0:
            return {
                'answer': "I'm sorry, no relevant information was found in the indexed document handbook.",
                'citations': []
            }
            
        context_str = "\n---\n".join([doc['content'] for doc in retrieved_docs])
        
        # Rule-based synthesis reflecting retrieved knowledge
        synthesis = f"Based on the handbook context:\n{retrieved_docs[0]['content']}"
        
        citations = [f"Chunk #{doc['chunk_id']} (Relevance Score: {doc['score']})" for doc in retrieved_docs]
        
        return {
            'query': user_query,
            'answer': synthesis,
            'retrieved_context': context_str,
            'citations': citations
        }
