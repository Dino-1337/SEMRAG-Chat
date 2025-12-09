"""Tests for semantic chunking module."""

import unittest
from src.semantic_chunking import SemanticChunker, Chunk


class TestSemanticChunking(unittest.TestCase):
    """Test cases for semantic chunking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = SemanticChunker(
            similarity_threshold=0.7,
            max_chunk_tokens=100,
            sub_chunk_tokens=50
        )
    
    def test_count_tokens(self):
        """Test token counting."""
        text = "This is a test sentence."
        token_count = self.chunker.count_tokens(text)
        self.assertGreater(token_count, 0)
    
    def test_semantic_chunking_basic(self):
        """Test basic semantic chunking."""
        sentences = [
            "Dr. Ambedkar was a great leader.",
            "He fought for social justice.",
            "The weather is nice today.",
            "I like to read books."
        ]
        
        embeddings = self.chunker.compute_sentence_embeddings(sentences)
        chunks = self.chunker.semantic_chunking(sentences, embeddings)
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Chunk)
    
    def test_chunk_has_required_attributes(self):
        """Test that chunks have required attributes."""
        sentences = ["Test sentence one.", "Test sentence two."]
        embeddings = self.chunker.compute_sentence_embeddings(sentences)
        chunks = self.chunker.semantic_chunking(sentences, embeddings)
        
        if chunks:
            chunk = chunks[0]
            self.assertIsNotNone(chunk.text)
            self.assertIsNotNone(chunk.sentences)
            self.assertIsNotNone(chunk.sentence_indices)
            self.assertIsNotNone(chunk.embedding)
            self.assertGreater(chunk.token_count, 0)


if __name__ == '__main__':
    unittest.main()

