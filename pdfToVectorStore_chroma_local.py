import os
from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

class PDFVectorIndexer:
    def __init__(self):
        # Use ChromaDB's default embedding model (no API needed)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="pdf_docs",
            embedding_function=self.embedding_function
        )

    def extract_text_chunks(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        """Extract text from PDF and split into chunks."""
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - chunk_overlap
        return chunks

    def index_pdf(self, pdf_path, metadata=None):
        """Index a PDF into ChromaDB."""
        chunks = self.extract_text_chunks(pdf_path)
        doc_ids = [f"doc_{i}" for i in range(len(chunks))]
        
        # Ensure metadata matches the number of chunks
        if metadata is None:
            metadata = {}
        metadatas = [metadata.copy() for _ in range(len(chunks))]  # Create one metadata dict per chunk
        
        self.collection.add(
            documents=chunks,
            ids=doc_ids,
            metadatas=metadatas  # Now matches length of documents
        )
        print(f"✅ Indexed {len(chunks)} chunks from {pdf_path}")

    def search(self, query, top_k=3):
        """Search the indexed PDFs."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "score": results["distances"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
            })
        return formatted_results

# Example Usage
if __name__ == "__main__":
    indexer = PDFVectorIndexer()
    
    # Index a PDF with metadata
    pdf_path = "sample_10k.pdf"
    indexer.index_pdf(pdf_path, metadata={"source": pdf_path, "type": "financial"})
    
    # Search
    query = "What is the P/E ratio?"
    results = indexer.search(query)
    
    print(f"🔍 Results for: '{query}'\n")
    for i, res in enumerate(results, 1):
        print(f"📄 Result {i} (Score: {res['score']:.2f}):")
        print(f"Metadata: {res['metadata']}")
        print(res["text"][:200] + "...\n")