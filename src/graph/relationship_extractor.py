"""
Improved Relationship Extraction for SEMRAG
- Canonical entity alignment
- Sentence-window co-occurrence
- Optional syntactic relation scoring
- Clean noun-chunk handling
"""

from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict
from src.graph.entity_extractor import Entity
from src.chunking.semantic_chunker import Chunk
from src.graph.graph_builder import canonicalize


@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    weight: int
    chunk_id: int


class RelationshipExtractor:
    def __init__(self, nlp, window_sentences: int = 1, syntax_boost: bool = True):
        self.nlp = nlp
        self.window_sentences = window_sentences
        self.syntax_boost = syntax_boost

    def extract_relationships(self, chunks: List[Chunk], entities: List[Entity]) -> List[Relationship]:
        print("Extracting relationships (clean + canonical + windowed)...")

        entity_map = {canonicalize(e.text): e for e in entities}
        rel_counter = defaultdict(lambda: {"weight": 0, "chunks": set()})

        for cid, chunk in enumerate(chunks):
            doc = self.nlp(chunk.text)

            mentions = []

            for ent in doc.ents:
                c = canonicalize(ent.text)
                if c in entity_map:
                    mentions.append((c, ent.start, ent.end, ent.sent.start))

            for nc in doc.noun_chunks:
                c = canonicalize(nc.text)
                if c in entity_map and len(c.split()) >= 2:
                    mentions.append((c, nc.start, nc.end, nc.sent.start))

            if len(mentions) < 2:
                continue

            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    e1, s1, e1_end, sent1 = mentions[i]
                    e2, s2, e2_end, sent2 = mentions[j]

                    if abs(sent1 - sent2) <= self.window_sentences:
                        key = tuple(sorted((e1, e2)))

                        rel_counter[key]["weight"] += 1
                        rel_counter[key]["chunks"].add(cid)

                        if self.syntax_boost:
                            tok1 = doc[s1]
                            tok2 = doc[s2]
                            if tok1.head == tok2 or tok2.head == tok1:
                                rel_counter[key]["weight"] += 1

        relationships = []
        for (e1, e2), data in rel_counter.items():
            relationships.append(
                Relationship(
                    source=e1,
                    target=e2,
                    relation="co_occurs",
                    weight=data["weight"],
                    chunk_id=min(data["chunks"])
                )
            )

        print(f"Extracted {len(relationships)} strong relationships")
        return relationships
