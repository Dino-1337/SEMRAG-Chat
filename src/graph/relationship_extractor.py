"""Relationship extraction using dependency parsing."""

from typing import List, Dict
from dataclasses import dataclass
from src.graph.entity_extractor import Entity
from src.chunking.semantic_chunker import Chunk


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    target: str
    relation: str
    chunk_id: int


class RelationshipExtractor:
    """Extracts relationships between entities using dependency parsing."""
    
    def __init__(self, nlp):
        """
        Initialize relationship extractor.
        
        Args:
            nlp: spaCy language model
        """
        self.nlp = nlp
    
    def extract_relationships(self, chunks: List[Chunk], entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships between entities.
        
        Args:
            chunks: List of text chunks
            entities: List of extracted entities
            
        Returns:
            List of relationships
        """
        relationships = []
        entity_set = {e.text.lower() for e in entities}
        
        print("Extracting relationships...")
        for chunk_idx, chunk in enumerate(chunks):
            doc = self.nlp(chunk.text)
            
            # Find entities in this chunk
            chunk_entities = []
            for ent in doc.ents:
                if ent.text.lower() in entity_set:
                    chunk_entities.append((ent.text, ent.start_char, ent.end_char))
            
            # Extract relationships between nearby entities
            for i, (ent1_text, start1, end1) in enumerate(chunk_entities):
                for j, (ent2_text, start2, end2) in enumerate(chunk_entities[i+1:], start=i+1):
                    ent1_span = doc.char_span(start1, end1)
                    ent2_span = doc.char_span(start2, end2)
                    
                    if ent1_span and ent2_span:
                        sent = ent1_span.sent
                        if ent2_span.sent == sent:
                            relation = self._extract_relation(ent1_span, ent2_span, sent)
                            
                            if relation:
                                rel = Relationship(
                                    source=ent1_text,
                                    target=ent2_text,
                                    relation=relation,
                                    chunk_id=chunk_idx
                                )
                                relationships.append(rel)
        
        print(f"Extracted {len(relationships)} relationships")
        return relationships
    
    def _extract_relation(self, ent1, ent2, sent) -> str:
        """Extract relationship type between two entities."""
        try:
            root1 = ent1.root
            root2 = ent2.root
            
            path = []
            current = root1
            while current != root2 and current.head != current:
                path.append(current.dep_)
                current = current.head
                if len(path) > 10:
                    break
            
            if any(dep in ["nsubj", "nsubjpass"] for dep in path):
                return "subject_of"
            elif any(dep in ["dobj", "pobj"] for dep in path):
                return "object_of"
            elif any(dep in ["prep"] for dep in path):
                return "related_to"
            else:
                return "co_occurs_with"
        except:
            return "co_occurs_with"

