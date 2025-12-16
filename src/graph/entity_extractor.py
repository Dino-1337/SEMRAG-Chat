"""
Entity Extractor for Option B:
- High-value NER entities
- Meaningful concepts from noun chunks
- Noise suppression (letters, greetings, months, generic nouns)
- Canonicalization for deduplication
"""

from typing import List, Dict
from collections import defaultdict
from dataclasses import dataclass
import spacy
import re

from src.chunking.semantic_chunker import Chunk


VALID_LABELS = {
    "PERSON", "ORG", "GPE", "NORP", "EVENT", "WORK_OF_ART"
}

GENERIC_BLACKLIST = {
    "chapter", "page", "section", "note", "review",
    "letter", "speech", "remarks", "appendix", "introduction",
    "conference", "committee", "country", "nation",
    "society", "people", "population", "problem",
    "date", "time", "belief", "object", "shadow", "neck",
    "spit", "travels", "journeys", "thing", "opinion",
    "address", "answer", "being", "structure"
}

MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
}

PRONOUN_PREFIXES = {"my", "his", "her", "its", "their", "our", "your"}

GENERIC_PREFIXES = {"any ", "some ", "every ", "only ", "even "}

VALID_SINGLE_WORDS = {
    "buddhism", "caste", "varna", "dalit", "gandhi", "ambedkar",
    "hinduism", "brahmin", "untouchable", "congress", "india",
    "constitution", "parliament", "democracy", "equality"
}

BAD_PATTERNS = [
    r"^\d+$",
    r"^\d+\.\d+$",
    r"^[ivxlcdm]+$",
    r"^[A-Za-z]\.$",
]

STOPWORD_PREFIXES = {"the", "a", "an", "this", "that", "these", "those"}


@dataclass
class Entity:
    text: str
    label: str
    chunk_ids: List[int]
    frequency: int = 1
    canonical: str = None


class EntityExtractor:
    def __init__(self, min_entity_frequency: int = 3):
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except:
            raise RuntimeError(
                "spaCy model not installed. Run: python -m spacy download en_core_web_lg"
            )

        self.min_freq = min_entity_frequency
        self.entities: Dict[str, Entity] = {}

    def _clean(self, text: str) -> str:
        """Normalize entity text."""
        t = text.strip()
        t = re.sub(r"\s+", " ", t)
        t = t.strip(".,:;()[]\"'")

        parts = t.split()
        if parts and parts[0].lower() in STOPWORD_PREFIXES:
            t = " ".join(parts[1:])

        return t.strip()

    def _valid_candidate(self, t: str) -> bool:
        tl = t.lower()

        if len(t) < 3:
            return False

        if tl in GENERIC_BLACKLIST:
            return False

        if tl in MONTHS:
            return False

        if any(ch.isdigit() for ch in t):
            return False

        if any(re.match(p, t) for p in BAD_PATTERNS):
            return False

        return True

    def _meaningful_concept(self, chunk):
        """Extracting high-value noun-phrase concepts from text."""
        concepts = []

        for nc in chunk.noun_chunks:
            txt = nc.text.strip()
            txt_clean = self._clean(txt)
            tl = txt_clean.lower()

            if len(txt_clean) < 4:
                continue

            if tl in GENERIC_BLACKLIST or tl in MONTHS:
                continue

            first_word = txt_clean.split()[0].lower()
            if first_word in PRONOUN_PREFIXES:
                continue

            if any(tl.startswith(prefix) for prefix in GENERIC_PREFIXES):
                continue

            if txt_clean.split()[0].lower() in STOPWORD_PREFIXES:
                continue

            if not any(t.pos_ in {"NOUN", "PROPN"} for t in nc):
                continue

            words = txt_clean.split()
            if len(words) == 1 and tl not in VALID_SINGLE_WORDS:
                continue

            if any(ch.isdigit() for ch in txt_clean):
                continue

            concepts.append(txt_clean)

        return concepts

    def _canonical(self, text: str) -> str:
        """Creating canonical form by removing titles and unifying variations."""
        t = text.lower()

        t = re.sub(r"\b(dr|mr|sir|shri|prof|babasaheb)\.?\b", "", t)
        t = re.sub(r"[^\w\s]", "", t)
        t = " ".join(t.split())

        return t.strip()

    def extract_entities(self, chunks: List[Chunk]) -> List[Entity]:
        bucket = defaultdict(lambda: {"label": None, "chunk_ids": set()})

        print("Extracting high-value entities and concepts...")

        for i, chunk in enumerate(chunks):
            doc = self.nlp(chunk.text)

            # 1. NER Entities
            for ent in doc.ents:
                clean = self._clean(ent.text)
                if ent.label_ not in VALID_LABELS:
                    continue
                if not self._valid_candidate(clean):
                    continue

                bucket[clean]["label"] = ent.label_
                bucket[clean]["chunk_ids"].add(i)

            # 2. Meaningful Concepts
            for c in self._meaningful_concept(doc):
                if not self._valid_candidate(c):
                    continue

                bucket[c]["label"] = "CONCEPT"
                bucket[c]["chunk_ids"].add(i)

        final = []

        for text, data in bucket.items():
            freq = len(data["chunk_ids"])
            if freq < self.min_freq:
                continue

            ent = Entity(
                text=text,
                label=data["label"],
                chunk_ids=list(data["chunk_ids"]),
                frequency=freq,
                canonical=self._canonical(text)
            )

            final.append(ent)
            self.entities[text] = ent

        print(f"✔ Extracted {len(final)} high-quality entities.")
        return final
