"""
ChromaDB Vector Store Wrapper for SEMRAG

Provides a clean interface for storing and querying chunk and community embeddings
using ChromaDB as the vector database.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from pathlib import Path


class ChromaVectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "data/processed/chroma_db"):
        """
        Initialize ChromaDB client and collections.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collections
        self.chunks_collection = self.client.get_or_create_collection(
            name="chunks",
            metadata={"description": "Chunk embeddings for semantic search"}
        )
        
        self.communities_collection = self.client.get_or_create_collection(
            name="communities",
            metadata={"description": "Community summary embeddings for global search"}
        )
    
    def reset(self):
        """Reset all collections (useful for rebuilding index)."""
        self.client.delete_collection("chunks")
        self.client.delete_collection("communities")
        
        self.chunks_collection = self.client.get_or_create_collection(
            name="chunks",
            metadata={"description": "Chunk embeddings for semantic search"}
        )
        
        self.communities_collection = self.client.get_or_create_collection(
            name="communities",
            metadata={"description": "Community summary embeddings for global search"}
        )
    
    def add_chunk_embeddings(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Add chunk embeddings to ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: NumPy array of embeddings [num_chunks, embedding_dim]
        """
        if len(chunks) == 0:
            return
        
        # Prepare data for ChromaDB
        ids = [str(chunk["chunk_id"]) for chunk in chunks]
        embeddings_list = embeddings.tolist()
        
        # Store text and metadata
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk["chunk_id"],
                "token_count": chunk.get("token_count", 0),
                "num_sentences": len(chunk.get("sentences", []))
            }
            for chunk in chunks
        ]
        
        # Add to collection
        self.chunks_collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(chunks)} chunk embeddings to ChromaDB")
    
    def add_community_embeddings(
        self,
        community_ids: List[int],
        embeddings: np.ndarray,
        summaries: Dict[int, str]
    ):
        """
        Add community summary embeddings to ChromaDB.
        
        Args:
            community_ids: List of community IDs
            embeddings: NumPy array of embeddings [num_communities, embedding_dim]
            summaries: Dictionary mapping community_id to summary text
        """
        if len(community_ids) == 0:
            return
        
        # Prepare data for ChromaDB
        ids = [f"community_{cid}" for cid in community_ids]
        embeddings_list = embeddings.tolist()
        
        # Store summaries and metadata
        documents = [summaries.get(cid, "") for cid in community_ids]
        metadatas = [
            {
                "community_id": cid,
                "summary_length": len(summaries.get(cid, ""))
            }
            for cid in community_ids
        ]
        
        # Add to collection
        self.communities_collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(community_ids)} community embeddings to ChromaDB")
    
    def query_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Query chunk embeddings for similar vectors.
        
        Args:
            query_embedding: Query vector [embedding_dim]
            top_k: Number of results to return
            where: Optional metadata filter
        
        Returns:
            Tuple of (chunk_ids, distances, metadatas)
        """
        # Query ChromaDB
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where
        )
        
        # Extract results
        if not results['ids'] or len(results['ids'][0]) == 0:
            return [], [], []
        
        chunk_ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        # Convert distances to similarities (ChromaDB returns L2 distances by default)
        # For cosine similarity: similarity = 1 - distance
        similarities = [1.0 - d for d in distances]
        
        return chunk_ids, similarities, metadatas
    
    def query_communities(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> Tuple[List[int], List[float], List[str]]:
        """
        Query community embeddings for similar vectors.
        
        Args:
            query_embedding: Query vector [embedding_dim]
            top_k: Number of results to return
        
        Returns:
            Tuple of (community_ids, similarities, summaries)
        """
        # Query ChromaDB
        results = self.communities_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Extract results
        if not results['ids'] or len(results['ids'][0]) == 0:
            return [], [], []
        
        # Parse community IDs from "community_X" format
        community_ids = [
            int(cid.replace("community_", ""))
            for cid in results['ids'][0]
        ]
        
        distances = results['distances'][0]
        summaries = results['documents'][0]
        
        # Convert distances to similarities
        similarities = [1.0 - d for d in distances]
        
        return community_ids, similarities, summaries
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Chunk ID to retrieve
        
        Returns:
            Chunk data dictionary or None if not found
        """
        try:
            result = self.chunks_collection.get(
                ids=[str(chunk_id)],
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not result['ids']:
                return None
            
            return {
                "chunk_id": chunk_id,
                "text": result['documents'][0],
                "embedding": np.array(result['embeddings'][0]),
                "metadata": result['metadatas'][0]
            }
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get statistics about stored vectors."""
        return {
            "total_chunks": self.chunks_collection.count(),
            "total_communities": self.communities_collection.count()
        }
