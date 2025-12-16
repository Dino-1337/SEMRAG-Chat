from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    sentences: List[str]
    sentence_indices: List[int]
    embedding: np.ndarray = None
    token_count: int = 0
    chunk_id: int = -1  # -1 indicates unassigned, makes bugs obvious


class SemanticChunker:
    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.72,
                 max_chunk_tokens: int = 1024,
                 sub_chunk_tokens: int = 128,
                 chunk_overlap: int = 20):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.sub_chunk_tokens = sub_chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.array([])
        return self.embedding_model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def semantic_chunking(self, sentences: List[str], embeddings: np.ndarray) -> List[Chunk]:
        if not sentences:
            return []

        chunks = []
        curr_sents = [sentences[0]]
        curr_indices = [0]

        for i in range(1, len(sentences)):
            prev_emb = embeddings[i - 1:i]
            cur_emb = embeddings[i:i + 1]
            sim = float(cosine_similarity(prev_emb, cur_emb)[0][0])

            if sim >= self.similarity_threshold:
                curr_sents.append(sentences[i])
                curr_indices.append(i)
            else:
                chunk_text = " ".join(curr_sents)
                chunk_emb = np.mean(embeddings[curr_indices], axis=0)
                token_count = self.count_tokens(chunk_text)
                chunks.append(Chunk(text=chunk_text,
                                    sentences=curr_sents.copy(),
                                    sentence_indices=curr_indices.copy(),
                                    embedding=chunk_emb,
                                    token_count=token_count))
                curr_sents = [sentences[i]]
                curr_indices = [i]

        if curr_sents:
            chunk_text = " ".join(curr_sents)
            chunk_emb = np.mean(embeddings[curr_indices], axis=0)
            token_count = self.count_tokens(chunk_text)
            chunks.append(Chunk(text=chunk_text,
                                sentences=curr_sents.copy(),
                                sentence_indices=curr_indices.copy(),
                                embedding=chunk_emb,
                                token_count=token_count))

        return chunks

    def enforce_token_limits(self, chunks: List[Chunk]) -> List[Chunk]:
        final = []
        for chunk in chunks:
            if chunk.token_count <= self.max_chunk_tokens:
                chunk.chunk_id = len(final)
                final.append(chunk)
            else:
                subs = self._split_large_chunk(chunk)
                for sub in subs:
                    sub.chunk_id = len(final)
                    final.append(sub)
        return final

    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        sentences = chunk.sentences
        if not sentences:
            return []

        sent_embs = self.compute_sentence_embeddings(sentences)
        final_subs = []
        n = len(sentences)
        i = 0

        while i < n:
            j = i
            current_sents = []
            current_indices = []

            while j < n:
                test_sents = current_sents + [sentences[j]]
                test_text = " ".join(test_sents)
                if self.count_tokens(test_text) <= self.sub_chunk_tokens:
                    current_sents.append(sentences[j])
                    current_indices.append(chunk.sentence_indices[j])
                    j += 1
                else:
                    break

            if not current_sents:
                # if single sentence exceeds sub_chunk_tokens, include it anyway
                current_sents = [sentences[j]]
                current_indices = [chunk.sentence_indices[j]]
                j = j + 1

            sub_text = " ".join(current_sents)
            # compute embedding for sub-chunk
            try:
                sub_emb = np.mean(sent_embs[[k - chunk.sentence_indices[0] for k in current_indices]], axis=0)
            except Exception:
                sub_emb = None

            token_count = self.count_tokens(sub_text)
            final_subs.append(Chunk(
                text=sub_text,
                sentences=current_sents,
                sentence_indices=current_indices,
                embedding=sub_emb,
                token_count=token_count
            ))

            # advance with overlap
            if j < n:
                overlap = max(1, len(current_sents) // 4)
                i = max(0, j - overlap)
            else:
                i = j

        return final_subs
