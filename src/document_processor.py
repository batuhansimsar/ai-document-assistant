"""
Document Processor Module
Handles PDF and TXT file reading, text extraction, and intelligent chunking.
"""

import os
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Process and chunk documents for RAG pipeline."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def read_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")
    
    def read_txt(self, file_path: str) -> str:
        """
        Read text from TXT file.
        
        Args:
            file_path: Path to TXT file
            
        Returns:
            File content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading TXT file: {str(e)}")
    
    def process_file(self, file_path: str) -> str:
        """
        Process file based on extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self.read_pdf(file_path)
        elif ext in ['.txt', '.md']:
            return self.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Additional metadata to attach
            
        Returns:
            List of chunks with metadata
        """
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "content": chunk,
                "metadata": {
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
            }
            documents.append(doc)
        
        return documents
    
    def process_and_chunk(self, file_path: str) -> List[Dict]:
        """
        Complete pipeline: read file and create chunks.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of document chunks with metadata
        """
        # Extract text
        text = self.process_file(file_path)
        
        # Create metadata
        metadata = {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            "file_type": os.path.splitext(file_path)[1]
        }
        
        # Chunk text
        return self.chunk_text(text, metadata)


if __name__ == "__main__":
    # Test the processor
    processor = DocumentProcessor()
    print("Document Processor initialized successfully!")
    print(f"Chunk size: {processor.chunk_size}")
    print(f"Chunk overlap: {processor.chunk_overlap}")
