"""
Vector Store Module
Handles ChromaDB operations, embeddings, and similarity search.
"""

import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStore:
    """Vector database for document storage and retrieval."""
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: HuggingFace embedding model name
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Dict]) -> int:
        """
        Add documents to vector store.
        
        Args:
            documents: List of document chunks with content and metadata
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Generate embeddings
        texts = [doc["content"] for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Prepare data for ChromaDB
        ids = [f"doc_{i}_{doc['metadata'].get('chunk_id', i)}" 
               for i, doc in enumerate(documents)]
        
        metadatas = [doc["metadata"] for doc in documents]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(documents)
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Dict = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of similar documents with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            doc = {
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            }
            documents.append(doc)
        
        return documents
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source.
        
        Args:
            source: Source file name
            
        Returns:
            Number of documents deleted
        """
        # Get all documents with this source
        results = self.collection.get(
            where={"source": source}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        
        return 0
    
    def get_stats(self) -> Dict:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with stats
        """
        count = self.collection.count()
        
        return {
            "total_documents": count,
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory
        }
    
    def clear_all(self):
        """Clear all documents from the vector store."""
        # Delete and recreate collection
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )


if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()
    print("Vector Store initialized successfully!")
    print(f"Stats: {store.get_stats()}")
