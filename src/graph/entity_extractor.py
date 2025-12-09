"""Entity extraction using spaCy NER."""

from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass
import spacy
from src.chunking.semantic_chunker import Chunk


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str  # Entity type (PERSON, ORG, etc.)
    chunk_ids: List[int]  # Chunks where this entity appears
    frequency: int = 1


class EntityExtractor:
    """Extracts entities from text chunks using spaCy NER."""
    
    def __init__(self, entity_types: List[str] = None, min_entity_frequency: int = 1):
        """
        Initialize entity extractor.
        
        Args:
            entity_types: List of entity types to extract
            min_entity_frequency: Minimum frequency for entity to be included
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy English model not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
        
        self.entity_types = entity_types or ["PERSON", "ORG", "GPE", "DATE", "EVENT", "WORK_OF_ART"]
        self.min_entity_frequency = min_entity_frequency
        self.entities: Dict[str, Entity] = {}
    
    def extract_entities(self, chunks: List[Chunk]) -> List[Entity]:
        """
        Extract entities from chunks using spaCy NER.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of extracted entities
        """
        entity_counter = defaultdict(lambda: {"label": None, "chunk_ids": set()})
        
        print("Extracting entities from chunks...")
        for chunk_idx, chunk in enumerate(chunks):
            doc = self.nlp(chunk.text)
            
            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    entity_text = ent.text.strip()
                    if len(entity_text) > 1:  # Filter very short entities
                        entity_counter[entity_text]["label"] = ent.label_
                        entity_counter[entity_text]["chunk_ids"].add(chunk_idx)
        
        # Create Entity objects
        entities = []
        for text, data in entity_counter.items():
            if len(data["chunk_ids"]) >= self.min_entity_frequency:
                entity = Entity(
                    text=text,
                    label=data["label"],
                    chunk_ids=list(data["chunk_ids"]),
                    frequency=len(data["chunk_ids"])
                )
                entities.append(entity)
                self.entities[text] = entity
        
        print(f"Extracted {len(entities)} unique entities")
        return entities

