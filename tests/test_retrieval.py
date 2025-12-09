"""Tests for retrieval module."""

import unittest
import numpy as np
from src.retrieval import RetrievalSystem
from src.semantic_chunking import Chunk
import networkx as nx


class TestRetrieval(unittest.TestCase):
    """Test cases for retrieval system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retrieval = RetrievalSystem(
            local_top_k=3,
            local_threshold=0.5
        )
    
    def test_compute_query_embedding(self):
        """Test query embedding computation."""
        query = "What is social justice?"
        embedding = self.retrieval.compute_query_embedding(query)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertGreater(embedding.shape[0], 0)
    
    def test_rank_chunks(self):
        """Test chunk ranking."""
        chunks = [
            Chunk(
                text="Test chunk one",
                sentences=["Test chunk one"],
                sentence_indices=[0],
                embedding=np.random.rand(384),
                token_count=10
            ),
            Chunk(
                text="Test chunk two",
                sentences=["Test chunk two"],
                sentence_indices=[1],
                embedding=np.random.rand(384),
                token_count=10
            )
        ]
        
        query_embedding = self.retrieval.compute_query_embedding("test query")
        results = self.retrieval.rank_chunks(query_embedding, chunks)
        
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 2)  # (chunk, score)
    
    def test_combine_results(self):
        """Test result combination."""
        chunks = [
            Chunk(
                text="Test",
                sentences=["Test"],
                sentence_indices=[0],
                embedding=np.random.rand(384),
                token_count=5
            )
        ]
        
        local_results = [(chunks[0], 0.8)]
        global_results = [(chunks[0], 0.7)]
        
        combined = self.retrieval.combine_results(
            local_results,
            global_results,
            strategy="union"
        )
        
        self.assertGreater(len(combined), 0)


if __name__ == '__main__':
    unittest.main()

