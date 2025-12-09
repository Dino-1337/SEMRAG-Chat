"""Semantic chunking implementation following SEMRAG Algorithm 1."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a semantic chunk of text."""
    text: str
    sentences: List[str]
    sentence_indices: List[int]
    embedding: np.ndarray = None
    token_count: int = 0
    chunk_id: int = 0


class SemanticChunker:
    """Implements semantic chunking using cosine similarity of sentence embeddings."""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7,
                 max_chunk_tokens: int = 1024,
                 sub_chunk_tokens: int = 128,
                 chunk_overlap: int = 20):
        """
        Initialize semantic chunker.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            similarity_threshold: Threshold for grouping sentences
            max_chunk_tokens: Maximum tokens per chunk
            sub_chunk_tokens: Size of sub-chunks when splitting large chunks
            chunk_overlap: Overlap between sub-chunks in tokens
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.sub_chunk_tokens = sub_chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute embeddings for all sentences."""
        embeddings = self.embedding_model.encode(
            sentences,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def semantic_chunking(self, sentences: List[str], embeddings: np.ndarray) -> List[Chunk]:
        """
        Algorithm 1: Semantic chunking via cosine similarity.
        
        Groups sentences into semantically coherent chunks based on
        cosine similarity between consecutive sentences.
        """
        if len(sentences) == 0:
            return []
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_indices = [0]
        
        for i in range(1, len(sentences)):
            # Compute cosine similarity with previous sentence
            if i > 0:
                similarity = cosine_similarity(
                    embeddings[i-1:i],
                    embeddings[i:i+1]
                )[0][0]
            else:
                similarity = 1.0
            
            # If similarity is above threshold, add to current chunk
            if similarity >= self.similarity_threshold:
                current_chunk_sentences.append(sentences[i])
                current_chunk_indices.append(i)
            else:
                # Start a new chunk
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunk_embedding = np.mean(
                        embeddings[current_chunk_indices],
                        axis=0
                    )
                    token_count = self.count_tokens(chunk_text)
                    
                    chunks.append(Chunk(
                        text=chunk_text,
                        sentences=current_chunk_sentences.copy(),
                        sentence_indices=current_chunk_indices.copy(),
                        embedding=chunk_embedding,
                        token_count=token_count
                    ))
                
                # Start new chunk
                current_chunk_sentences = [sentences[i]]
                current_chunk_indices = [i]
        
        # Add the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_embedding = np.mean(
                embeddings[current_chunk_indices],
                axis=0
            )
            token_count = self.count_tokens(chunk_text)
            
            chunks.append(Chunk(
                text=chunk_text,
                sentences=current_chunk_sentences.copy(),
                sentence_indices=current_chunk_indices.copy(),
                embedding=chunk_embedding,
                token_count=token_count
            ))
        
        return chunks
    
    def enforce_token_limits(self, chunks: List[Chunk]) -> List[Chunk]:
        """Enforce token limits by splitting large chunks."""
        final_chunks = []
        
        for chunk in chunks:
            if chunk.token_count <= self.max_chunk_tokens:
                chunk.chunk_id = len(final_chunks)
                final_chunks.append(chunk)
            else:
                # Split into sub-chunks
                sub_chunks = self._split_large_chunk(chunk)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = len(final_chunks)
                    final_chunks.append(sub_chunk)
        
        return final_chunks
    
    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """Split a large chunk into smaller sub-chunks with overlap."""
        sentences = chunk.sentences
        sub_chunks = []
        
        i = 0
        while i < len(sentences):
            current_sentences = []
            current_indices = []
            
            while i < len(sentences):
                test_sentences = current_sentences + [sentences[i]]
                test_text = " ".join(test_sentences)
                test_tokens = self.count_tokens(test_text)
                
                if test_tokens <= self.sub_chunk_tokens:
                    current_sentences.append(sentences[i])
                    current_indices.append(chunk.sentence_indices[i])
                    i += 1
                else:
                    break
            
            if current_sentences:
                sub_text = " ".join(current_sentences)
                sub_embedding = np.mean(
                    [chunk.embedding] if len(current_sentences) == 1
                    else [self.embedding_model.encode(s)[0] for s in current_sentences],
                    axis=0
                )
                token_count = self.count_tokens(sub_text)
                
                sub_chunks.append(Chunk(
                    text=sub_text,
                    sentences=current_sentences,
                    sentence_indices=current_indices,
                    embedding=sub_embedding,
                    token_count=token_count
                ))
                
                # Move back for overlap
                if i < len(sentences):
                    overlap_sentences = max(1, len(current_sentences) // 4)
                    i = max(0, i - overlap_sentences)
            else:
                i += 1
        
        return sub_chunks

