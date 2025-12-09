"""Buffer merging to preserve contextual continuity."""

from typing import List
import numpy as np
from .semantic_chunker import Chunk


class BufferMerger:
    """Applies buffer merging to preserve contextual continuity."""
    
    def __init__(self, buffer_size: int = 2, max_chunk_tokens: int = 1024):
        """
        Initialize buffer merger.
        
        Args:
            buffer_size: Number of sentences to merge at boundaries
            max_chunk_tokens: Maximum tokens per chunk
        """
        self.buffer_size = buffer_size
        self.max_chunk_tokens = max_chunk_tokens
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def apply_buffer_merging(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply buffer merging to preserve contextual continuity.
        
        Merges chunks at boundaries to maintain context.
        """
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            merged_sentences = current_chunk.sentences.copy()
            merged_indices = current_chunk.sentence_indices.copy()
            
            # Merge with next chunks if within buffer size
            j = i + 1
            while j < len(chunks) and (j - i) <= self.buffer_size:
                next_chunk = chunks[j]
                # Check if merging would exceed token limit
                combined_text = " ".join(merged_sentences + next_chunk.sentences)
                if self.count_tokens(combined_text) <= self.max_chunk_tokens:
                    merged_sentences.extend(next_chunk.sentences)
                    merged_indices.extend(next_chunk.sentence_indices)
                    j += 1
                else:
                    break
            
            # Create merged chunk
            merged_text = " ".join(merged_sentences)
            merged_embedding = np.mean(
                [chunks[k].embedding for k in range(i, j)],
                axis=0
            )
            token_count = self.count_tokens(merged_text)
            
            merged_chunks.append(Chunk(
                text=merged_text,
                sentences=merged_sentences,
                sentence_indices=merged_indices,
                embedding=merged_embedding,
                token_count=token_count
            ))
            
            i = j
        
        return merged_chunks

