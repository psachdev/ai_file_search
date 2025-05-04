import os
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

class PDFToVectorStore:
    def __init__(self, api_key=None):
        """
        Initialize the PDF processor with OpenAI client
        
        Args:
            api_key (str): OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.vector_store_id = None
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
    def create_vector_store(self, name="PDF Documents"):
        """
        Create a new vector store
        
        Args:
            name (str): Name for the vector store
            
        Returns:
            str: ID of the created vector store
        """
        vector_store = self.client.vector_stores.create(name=name)
        self.vector_store_id = vector_store.id
        print(f"Vector store created with ID: {self.vector_store_id}")
        return vector_store.id
    
    def upload_pdf_to_vector_store(self, pdf_path, vector_store_id=None):
        """
        Upload a PDF file to the vector store
        
        Args:
            pdf_path (str): Path to the PDF file
            vector_store_id (str): ID of the vector store. If None, uses the one created earlier
            
        Returns:
            tuple: (file_id, vector_store_file_id)
        """
        if vector_store_id is None:
            if self.vector_store_id is None:
                raise ValueError("No vector store ID provided and none created yet")
            vector_store_id = self.vector_store_id
        
        # Step 1: Upload the file
        file = self.client.files.create(
            file=open(pdf_path, "rb"),
            purpose="assistants"
        )
        
        # Step 2: Add to vector store
        vector_store_file = self.client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file.id
        )
        
        return file.id, vector_store_file.id
    
    def search_vector_store(self, query, vector_store_id=None, limit=5):
        """
        Search the vector store
        
        Args:
            query (str): Search query
            vector_store_id (str): ID of the vector store
            limit (int): Maximum number of results to return
            
        Returns:
            list: Search results
        """
        if vector_store_id is None:
            if self.vector_store_id is None:
                raise ValueError("No vector store ID provided and none created yet")
            vector_store_id = self.vector_store_id
        
        return self.client.vector_stores.search(
            vector_store_id=vector_store_id,
            query=query,
            max_num_results=limit
        )

    def analyze_with_gpt4o(self, query, context):

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst. Summarize operational expenses from the provided 10K document excerpts. Be concise and highlight key figures."
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext:\n{context}"
                }
            ],
            temperature=0.3  # Keep outputs factual
        )
        return response.choices[0].message.content
    
    def delete_vector_store(self, vector_store_id=None):
        """
        Delete the vector store
        
        Args:
            vector_store_id (str): ID of the vector store to delete
        """
        if vector_store_id is None:
            if self.vector_store_id is None:
                raise ValueError("No vector store ID provided and none created yet")
            vector_store_id = self.vector_store_id
        
        self.client.vector_stores.delete(vector_store_id)
        if vector_store_id == self.vector_store_id:
            self.vector_store_id = None

# Example Usage
if __name__ == "__main__":
    # Initialize with your API key (or set OPENAI_API_KEY environment variable)
    processor = PDFToVectorStore()
    
    # Path to your PDF file
    pdf_path = "sample_10k.pdf"
    
    try:
        # Step 1: Create a vector store
        vector_store_id = processor.create_vector_store(name="My PDF Documents")
        print(f"Created vector store with ID: {vector_store_id}")
        
        # Step 2: Upload PDF to vector store
        file_id, vs_file_id = processor.upload_pdf_to_vector_store(pdf_path)
        print(f"Uploaded PDF. File ID: {file_id}, Vector Store File ID: {vs_file_id}")
        
        # Wait a bit for processing (you might want to implement proper polling in production)
        import time
        print("Waiting for file to be processed...")
        time.sleep(30)  # Adjust based on file size
        
        # Step 3 (Optional): Search the vector store
        #query = "Was apple stock a buy in 2023?"
        #results = processor.search_vector_store(query)
        #print("\nSearch Results: " + str(results))
    
            
    finally:
        print("\nEND")
