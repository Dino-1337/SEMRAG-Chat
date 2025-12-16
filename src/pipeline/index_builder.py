# index_builder.py
# Clean, PyMuPDF-compatible index builder for SEMRAG

import json, pickle, hashlib, yaml
import numpy as np
from pathlib import Path
from datetime import datetime

from src.utils.data_loader import PDFLoader
from src.chunking.semantic_chunker import SemanticChunker, Chunk
from src.chunking.buffer_merger import BufferMerger

from src.graph.entity_extractor import EntityExtractor, Entity
from src.graph.relationship_extractor import RelationshipExtractor
from src.graph.graph_builder import GraphBuilder, canonicalize
from src.graph.community_detector import CommunityDetector
from src.graph.summarizer import CommunitySummarizer

from src.llm.llm_client import LLMClient
from sentence_transformers import SentenceTransformer
from src.utils.vector_store import ChromaVectorStore


class IndexBuilder:
    """Builds complete SEMRAG index from PDF + saves all artifacts."""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.out = Path("data/processed")
        self.out.mkdir(parents=True, exist_ok=True)

        emb_model = self.config["embedding"]["model_name"]
        cg = self.config["chunking"]
        kg = self.config["knowledge_graph"]
        llm_cfg = self.config["llm"]

        self.loader = PDFLoader()

        self.chunker = SemanticChunker(
            embedding_model_name=emb_model,
            similarity_threshold=cg["similarity_threshold"],
            max_chunk_tokens=cg["max_chunk_tokens"],
            sub_chunk_tokens=cg["sub_chunk_tokens"],
            chunk_overlap=cg["chunk_overlap"],
        )
        self.buffer_merger = BufferMerger(buffer_size=cg["buffer_size"],
                                          max_chunk_tokens=cg["max_chunk_tokens"])

        self.entity_extractor = EntityExtractor(
            min_entity_frequency=kg.get("min_entity_frequency", 1)
        )
        self.relationship_extractor = RelationshipExtractor(
            nlp=self.entity_extractor.nlp
        )
        self.graph_builder = GraphBuilder()
        self.community_detector = CommunityDetector(
            community_resolution=kg["community_resolution"]
        )

        self.llm = LLMClient(
            model=llm_cfg["model"],
            base_url=llm_cfg["base_url"],
            temperature=llm_cfg["temperature"],
        )
        self.summarizer = CommunitySummarizer(self.llm)
        self.summary_embedder = SentenceTransformer(emb_model)
        
        # Initialize ChromaDB vector store
        self.vector_store = ChromaVectorStore(
            persist_directory=str(self.out / "chroma_db")
        )

        self.chunks: list[Chunk] = []
        self.entities: dict[str, Entity] = {}
        self.relationships = []
        self.graph = None
        self.communities = {}
        self.community_chunks = {}
        self.community_summaries = {}
        self.summary_embeddings = {}

    # ======================================================
    # BUILD INDEX
    # ======================================================
    def build_index(self, pdf_path: str):
        print("\n=== BUILDING INDEX ===")

        print("\n[1] Loading PDF with PyMuPDF...")
        text = self.loader.load_pdf(pdf_path)
        sentences = self.loader.split_into_sentences(text)
        print(f"✓ Sentences: {len(sentences)}")

        print("\n[2] Semantic Chunking...")
        embeddings = self.chunker.compute_sentence_embeddings(sentences)
        chunks = self.chunker.semantic_chunking(sentences, embeddings)
        chunks = self.chunker.enforce_token_limits(chunks)
        chunks = self.buffer_merger.apply_buffer_merging(chunks)
        
        # CRITICAL: Assign unique chunk IDs after all processing
        for idx, chunk in enumerate(chunks):
            chunk.chunk_id = idx
        
        self.chunks = chunks
        print(f"✓ Final chunks: {len(chunks)}")

        print("\n[3] Extracting Entities...")
        ents = self.entity_extractor.extract_entities(chunks)
        # CRITICAL: Store entities using canonical names
        self.entities = {canonicalize(e.text): e for e in ents}
        print(f"✓ Entities: {len(ents)}")

        print("\n[4] Extracting Relationships + Building Graph...")
        rels = self.relationship_extractor.extract_relationships(chunks, ents)
        # CRITICAL: Canonicalize relationship sources/targets
        rel_dicts = [
            {
                "source": canonicalize(r.source),
                "target": canonicalize(r.target),
                "relation": r.relation,
                "weight": r.weight,
            }
            for r in rels
        ]
        self.relationships = rels
        self.graph, self.alias_map = self.graph_builder.build_graph(ents, rel_dicts)

        print(f"✓ Graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

        print("\n[5] Detecting Communities...")
        self.communities = self.community_detector.detect_communities(self.graph)
        self.community_chunks = self.community_detector.get_community_chunks(
            self.communities, self.entities, self.chunks
        )
        print(f"✓ Communities: {len(self.communities)}")

        print("\n[6] Summarizing Communities...")
        self.community_summaries = self.summarizer.generate_summaries(
            self.communities, self.community_chunks, self.chunks, self.entities
        )
        for cid, summ in self.community_summaries.items():
            emb = self.summary_embedder.encode([summ], convert_to_numpy=True)[0]
            self.summary_embeddings[cid] = emb
        print(f"✓ Summaries: {len(self.community_summaries)}")

        print("\n=== SAVING ARTIFACTS ===")
        self._save_all()
        print("\n✓ Index Build Complete.")

        return self._stats()

    # ======================================================
    # SAVE ARTIFACTS
    # ======================================================
    def _save_all(self):
        # Reset ChromaDB collections for clean rebuild
        print("\n[Saving] Resetting ChromaDB collections...")
        self.vector_store.reset()
        
        # Save chunk metadata as JSON
        chunk_data = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "sentences": c.sentences,
                "sentence_indices": c.sentence_indices,
                "token_count": c.token_count,
            }
            for c in self.chunks
        ]
        self._save_json("chunks.json", chunk_data)

        # Save chunk embeddings to ChromaDB
        print("[Saving] Adding chunk embeddings to ChromaDB...")
        chunk_embeddings = np.array([c.embedding for c in self.chunks])
        self.vector_store.add_chunk_embeddings(chunk_data, chunk_embeddings)

        # Save entity metadata
        self._save_json("entities.json", {
            name: {
                "text": e.text,
                "label": e.label,
                "chunk_ids": e.chunk_ids,
                "frequency": e.frequency,
            }
            for name, e in self.entities.items()
        })

        # Save knowledge graph (NetworkX)
        with open(self.out / "knowledge_graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)

        # Save community data
        self._save_json("communities.json", self.communities)
        self._save_json("community_chunks.json", self.community_chunks)
        self._save_json("community_summaries.json", self.community_summaries)

        # Save community embeddings to ChromaDB
        print("[Saving] Adding community embeddings to ChromaDB...")
        sorted_ids = sorted(self.summary_embeddings.keys())
        community_embeddings = np.array([self.summary_embeddings[cid] for cid in sorted_ids])
        self.vector_store.add_community_embeddings(
            sorted_ids,
            community_embeddings,
            self.community_summaries
        )

        self._save_json("metadata.json", {
            "timestamp": datetime.now().isoformat(),
            "config_hash": self._config_hash(),
            "stats": self._stats(),
        })

    # ======================================================
    # HELPERS
    # ======================================================
    def _save_json(self, filename, data):
        with open(self.out / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _config_hash(self):
        raw = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def _stats(self):
        return {
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "graph_nodes": self.graph.number_of_nodes() if self.graph else 0,
            "graph_edges": self.graph.number_of_edges() if self.graph else 0,
            "communities": len(self.communities),
            "summaries": len(self.community_summaries),
        }
