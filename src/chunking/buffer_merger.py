from typing import List
import numpy as np
from .semantic_chunker import Chunk


class BufferMerger:
    def __init__(self, buffer_size: int = 2, max_chunk_tokens: int = 1024):
        self.buffer_size = buffer_size
        self.max_chunk_tokens = max_chunk_tokens
        try:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is None:
            return len(text.split())
        return len(self._encoding.encode(text))

    def apply_buffer_merging(self, chunks: List[Chunk]) -> List[Chunk]:
        if len(chunks) <= 1:
            return chunks

        merged_chunks = []
        i = 0
        n = len(chunks)

        while i < n:
            merged_sentences = chunks[i].sentences.copy()
            merged_indices = chunks[i].sentence_indices.copy()
            j = i + 1

            while j < n and (j - i) <= self.buffer_size:
                next_chunk = chunks[j]
                combined_text = " ".join(merged_sentences + next_chunk.sentences)
                if self.count_tokens(combined_text) <= self.max_chunk_tokens:
                    merged_sentences.extend(next_chunk.sentences)
                    merged_indices.extend(next_chunk.sentence_indices)
                    j += 1
                else:
                    break

            merged_text = " ".join(merged_sentences)
            # compute merged embedding: average available chunk embeddings
            embeddings = []
            for k in range(i, j):
                if getattr(chunks[k], "embedding", None) is not None:
                    embeddings.append(chunks[k].embedding)
            if embeddings:
                merged_embedding = np.mean(embeddings, axis=0)
            else:
                merged_embedding = None

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
