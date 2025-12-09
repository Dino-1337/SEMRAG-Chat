"""Tests for knowledge graph module."""

import unittest
from src.knowledge_graph import KnowledgeGraphBuilder, Entity, Relationship
from src.semantic_chunking import Chunk
import numpy as np


class TestKnowledgeGraph(unittest.TestCase):
    """Test cases for knowledge graph construction."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.kg_builder = KnowledgeGraphBuilder(
                entity_types=["PERSON", "ORG", "GPE"],
                min_entity_frequency=1
            )
        except RuntimeError as e:
            self.skipTest(f"spaCy model not available: {e}")
    
    def test_extract_entities(self):
        """Test entity extraction."""
        chunks = [
            Chunk(
                text="Dr. Ambedkar was born in India.",
                sentences=["Dr. Ambedkar was born in India."],
                sentence_indices=[0],
                embedding=np.random.rand(384),
                token_count=10
            )
        ]
        
        entities = self.kg_builder.extract_entities(chunks)
        self.assertIsInstance(entities, list)
    
    def test_build_graph(self):
        """Test graph construction."""
        # Create mock entities
        entities = [
            Entity(
                text="Ambedkar",
                label="PERSON",
                chunk_ids=[0],
                frequency=1
            ),
            Entity(
                text="India",
                label="GPE",
                chunk_ids=[0],
                frequency=1
            )
        ]
        
        # Create mock relationships
        relationships = [
            Relationship(
                source="Ambedkar",
                target="India",
                relation="born_in",
                chunk_id=0
            )
        ]
        
        graph = self.kg_builder.build_knowledge_graph(entities, relationships)
        
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreaterEqual(graph.number_of_edges(), 0)


if __name__ == '__main__':
    unittest.main()

