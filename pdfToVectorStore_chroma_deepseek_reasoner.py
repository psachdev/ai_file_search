import os
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI  # DeepSeek uses OpenAI-style API
from dotenv import load_dotenv

load_dotenv()

class PDFAnalyzer:
    def __init__(self):
        # ChromaDB setup (local embeddings)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="pdf_docs",
            embedding_function=self.embedding_function
        )
        
        # DeepSeek client (OpenAI format)
        self.deepseek = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"  # Official endpoint
        )

    def search_and_reason(self, query, top_k=3):
        """Search ChromaDB, then reason with DeepSeek."""
        # Step 1: Get relevant chunks from ChromaDB
        chroma_results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format context for DeepSeek
        context = "\n\n".join([
            f"--- Excerpt {i+1} ---\n{text}"
            for i, text in enumerate(chroma_results["documents"][0])
        ])

        # Step 2: Query DeepSeek with context
        response = self.deepseek.chat.completions.create(
            model="deepseek-reasoner",  # Or "deepseek-v3"
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst. Use the provided document excerpts to answer questions precisely. If data is missing, say so."
                },
                {
                    "role": "user",
                    "content": f"Document Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            temperature=0.1  # Keep responses factual
        )
        
        return response.choices[0].message.content

# Usage Example
if __name__ == "__main__":
    analyzer = PDFAnalyzer()
    
    # Example: Complex financial query
    query = "Based on the net losses and shares outstanding, what was the approximate P/E ratio in 2013?"
    answer = analyzer.search_and_reason(query)
    
    print(f"🔍 Query: {query}")
    print(f"💡 DeepSeek Analysis:\n{answer}")