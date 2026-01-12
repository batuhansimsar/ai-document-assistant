"""
RAG Engine Module
Orchestrates the complete RAG pipeline: document ingestion, retrieval, and generation.
"""

import os
from typing import List, Dict, Optional
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_handler import OllamaLLM


class RAGEngine:
    """Complete RAG pipeline orchestrator."""
    
    def __init__(
        self,
        ollama_model: str = "llama3.2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_context_chunks: int = 5
    ):
        """
        Initialize RAG engine.
        
        Args:
            ollama_model: Ollama model name
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            max_context_chunks: Max chunks to use for context
        """
        self.max_context_chunks = max_context_chunks
        
        # Initialize components
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = VectorStore()
        self.llm = OllamaLLM(model=ollama_model)
    
    def ingest_document(self, file_path: str) -> Dict:
        """
        Ingest a document into the system.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Ingestion statistics
        """
        # Process and chunk document
        chunks = self.doc_processor.process_and_chunk(file_path)
        
        # Add to vector store
        num_added = self.vector_store.add_documents(chunks)
        
        return {
            "file_name": os.path.basename(file_path),
            "chunks_created": len(chunks),
            "chunks_stored": num_added,
            "status": "success"
        }
    
    def query(
        self,
        question: str,
        source_filter: Optional[str] = None,
        stream: bool = False
    ):
        """
        Query the RAG system.
        
        Args:
            question: User question
            source_filter: Optional filter by source file
            stream: Whether to stream response
            
        Returns:
            Response dict with answer and sources
        """
        # Retrieve relevant chunks
        filter_dict = {"source": source_filter} if source_filter else None
        
        relevant_docs = self.vector_store.similarity_search(
            query=question,
            k=self.max_context_chunks,
            filter_dict=filter_dict
        )
        
        # Extract context
        context = [doc["content"] for doc in relevant_docs]
        
        if not context:
            return {
                "answer": "Henüz hiç doküman yüklenmemiş. Lütfen önce bir doküman yükleyin!",
                "sources": [],
                "stream": False
            }
        
        # Generate response
        if stream:
            # Return generator for streaming
            return {
                "answer": self.llm.generate_stream(question, context),
                "sources": relevant_docs,
                "stream": True
            }
        else:
            answer = self.llm.generate(question, context)
            return {
                "answer": answer,
                "sources": relevant_docs,
                "stream": False
            }
    
    def delete_document(self, source: str) -> int:
        """
        Delete a document from the system.
        
        Args:
            source: Source file name
            
        Returns:
            Number of chunks deleted
        """
        return self.vector_store.delete_by_source(source)
    
    def list_documents(self) -> List[str]:
        """
        List all documents in the system.
        
        Returns:
            List of unique source file names
        """
        stats = self.vector_store.get_stats()
        
        # Get all unique sources
        results = self.vector_store.collection.get()
        
        if not results['metadatas']:
            return []
        
        sources = set(meta.get('source', 'Unknown') for meta in results['metadatas'])
        return sorted(list(sources))
    
    def get_stats(self) -> Dict:
        """
        Get system statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self.vector_store.get_stats(),
            "model": self.llm.model,
            "ollama_health": self.llm.check_health()
        }
    
    def clear_all(self):
        """Clear all documents from the system."""
        self.vector_store.clear_all()


if __name__ == "__main__":
    # Test RAG engine
    engine = RAGEngine()
    print("RAG Engine initialized successfully!")
    print(f"Stats: {engine.get_stats()}")
