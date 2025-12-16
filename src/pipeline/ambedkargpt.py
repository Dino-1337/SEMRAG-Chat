"""Main orchestrator for SEMRAG RAG System."""

from typing import Dict, List, Any, Optional
import yaml, json, pickle
import numpy as np
import networkx as nx
from pathlib import Path

from src.utils.data_loader import PDFLoader
from src.chunking.semantic_chunker import SemanticChunker, Chunk
from src.chunking.buffer_merger import BufferMerger

from src.graph.entity_extractor import EntityExtractor, Entity
from src.graph.relationship_extractor import RelationshipExtractor
from src.graph.graph_builder import GraphBuilder, canonicalize
from src.graph.community_detector import CommunityDetector
from src.graph.summarizer import CommunitySummarizer

from src.retrieval.local_search import LocalRAGSearch
from src.retrieval.global_search import GlobalRAGSearch
from src.retrieval.ranker import ResultRanker

from src.llm.llm_client import LLMClient
from src.llm.answer_generator import AnswerGenerator

from src.utils.query_expander import QueryExpander
from src.utils.vector_store import ChromaVectorStore


class AmbedkarGPT:
    def __init__(self, config_path: str = "config.yaml", mode: str = "index"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.mode = mode
        self.pdf_loader = PDFLoader()

        llm_cfg = self.config.get("llm", {})
        self.llm_client = LLMClient(
            model=llm_cfg.get("model", "mistral:7b"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            temperature=llm_cfg.get("temperature", 0.4),
            max_tokens=llm_cfg.get("max_tokens", 900),
        )
        self.answer_generator = AnswerGenerator(self.llm_client)

        emb_model = self.config["embedding"]["model_name"]
        ret_cfg = self.config.get("retrieval", {})
        local_cfg = ret_cfg.get("local_search", {})
        global_cfg = ret_cfg.get("global_search", {})

        self.vector_store = ChromaVectorStore(
            persist_directory="data/processed/chroma_db"
        )
        
        self.local_search = LocalRAGSearch(
            embedding_model_name=emb_model,
            top_k=local_cfg.get("top_k", 5),
            threshold=local_cfg.get("similarity_threshold", 0.2),
            vector_store=self.vector_store,
        )
        self.global_search = GlobalRAGSearch(
            embedding_model_name=emb_model,
            top_k=global_cfg.get("top_k", 3),
            threshold=global_cfg.get("similarity_threshold", 0.2),
            vector_store=self.vector_store,
        )
        self.ranker = ResultRanker()
        self.query_expander = QueryExpander()

        if mode == "index":
            c_cfg = self.config.get("chunking", {})
            self.chunker = SemanticChunker(
                embedding_model_name=emb_model,
                similarity_threshold=c_cfg.get("similarity_threshold", 0.72),
                max_chunk_tokens=c_cfg.get("max_chunk_tokens", 1024),
                sub_chunk_tokens=c_cfg.get("sub_chunk_tokens", 128),
                chunk_overlap=c_cfg.get("chunk_overlap", 20),
            )
            self.buffer_merger = BufferMerger(
                buffer_size=c_cfg.get("buffer_size", 2),
                max_chunk_tokens=c_cfg.get("max_chunk_tokens", 1024),
            )

            kg_cfg = self.config.get("knowledge_graph", {})
            self.entity_extractor = EntityExtractor(
                min_entity_frequency=kg_cfg.get("min_entity_frequency", 1)
            )
            self.relationship_extractor = RelationshipExtractor(
                self.entity_extractor.nlp
            )
            self.graph_builder = GraphBuilder()
            self.community_detector = CommunityDetector(
                community_resolution=kg_cfg.get("community_resolution", 1.0)
            )
            self.community_summarizer = CommunitySummarizer(self.llm_client)

        self.chunks: List[Chunk] = []
        self.graph: Optional[nx.Graph] = None
        self.entities: Dict[str, Entity] = {}
        self.relationships: List = []
        self.communities: Dict[int, List[str]] = {}
        self.community_chunks: Dict[int, List[int]] = {}
        self.community_summaries: Dict[int, str] = {}
        self.community_summary_embeddings: Dict[int, np.ndarray] = {}
        self.processed = False

    def load_and_process(self, pdf_path: str, save_processed=True):
        """Loading PDF and processing it through the complete SEMRAG pipeline."""
        print("\n=== AmbedkarGPT: Processing PDF ===")

        print("\n[1] Loading PDF with PyMuPDF...")
        text = self.pdf_loader.load_pdf(pdf_path)
        print(f"Text length: {len(text)} chars")

        print("\n[2] Splitting into sentences...")
        sentences = self.pdf_loader.split_into_sentences(text)
        print(f"Sentences: {len(sentences)}")

        print("\n[3] Semantic chunking...")
        embeddings = self.chunker.compute_sentence_embeddings(sentences)
        chunks = self.chunker.semantic_chunking(sentences, embeddings)
        print(f"Initial chunks: {len(chunks)}")

        chunks = self.buffer_merger.apply_buffer_merging(chunks)
        print(f"After merging: {len(chunks)}")

        chunks = self.chunker.enforce_token_limits(chunks)
        
        for idx, chunk in enumerate(chunks):
            chunk.chunk_id = idx
        
        print(f"Final chunks: {len(chunks)}")
        self.chunks = chunks

        print("\n[4] Building Knowledge Graph...")
        ents = self.entity_extractor.extract_entities(chunks)
        self.entities = {canonicalize(e.text): e for e in ents}

        rels = self.relationship_extractor.extract_relationships(chunks, ents)
        self.relationships = [
            {
                "source": canonicalize(r.source),
                "target": canonicalize(r.target),
                "relation": r.relation,
                "weight": r.weight,
            }
            for r in rels
        ]

        self.graph, self.alias_map = self.graph_builder.build_graph(ents, self.relationships)
        self.communities = self.community_detector.detect_communities(self.graph)
        self.community_chunks = self.community_detector.get_community_chunks(
            self.communities, self.entities, chunks
        )

        print("\n[5] Community Summaries...")
        self.community_summaries = self.community_summarizer.generate_summaries(
            self.communities, self.community_chunks, chunks
        )

        for cid, summary in self.community_summaries.items():
            emb = self.answer_generator.embedder.encode(
                [summary], convert_to_numpy=True
            )[0]
            self.community_summary_embeddings[cid] = emb

        print("\n=== Processing Complete ===")
        print(f"Chunks: {len(self.chunks)}")
        print(f"Entities: {len(self.entities)}")
        print(f"Graph Nodes: {self.graph.number_of_nodes()}")
        print(f"Graph Edges: {self.graph.number_of_edges()}")
        print(f"Communities: {len(self.communities)}")

        if save_processed:
            self._save_processed()

        self.processed = True

    def _save_processed(self):
        """Saving all processed data to disk for later querying."""
        out = Path("data/processed")
        out.mkdir(parents=True, exist_ok=True)

        # Chunks
        with open(out / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "text": c.text,
                        "sentences": c.sentences,
                        "sentence_indices": c.sentence_indices,
                        "token_count": c.token_count,
                        "chunk_id": c.chunk_id,
                    }
                    for c in self.chunks
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )


        ents = {
            name: {
                "text": e.text,
                "label": e.label,
                "chunk_ids": e.chunk_ids,
                "frequency": e.frequency,
            }
            for name, e in self.entities.items()
        }
        with open(out / "entities.json", "w", encoding="utf-8") as f:
            json.dump(ents, f, indent=2, ensure_ascii=False)


        with open(out / "knowledge_graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)

        with open(out / "communities.json", "w") as f:
            json.dump(self.communities, f, indent=2)

        with open(out / "community_chunks.json", "w") as f:
            json.dump(self.community_chunks, f, indent=2)

        with open(out / "community_summaries.json", "w") as f:
            json.dump(self.community_summaries, f, indent=2)

        meta = {
            "stats": {
                "chunks": len(self.chunks),
                "entities": len(self.entities),
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
                "communities": len(self.communities),
            }
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load_index(self):
        """Loading previously saved index data from disk."""
        base = Path("data/processed")

        with open(base / "chunks.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks = [
            Chunk(
                text=d["text"],
                sentences=d["sentences"],
                sentence_indices=d["sentence_indices"],
                token_count=d["token_count"],
                chunk_id=d["chunk_id"],
            )
            for d in raw
        ]

        chunk_embs = np.load(base / "chunk_embeddings.npy")
        for i, c in enumerate(self.chunks):
            c.embedding = chunk_embs[i]

        with open(base / "entities.json", "r", encoding="utf-8") as f:
            ents = json.load(f)
        self.entities = {
            canonicalize(name): Entity(
                text=info["text"],
                label=info["label"],
                chunk_ids=info["chunk_ids"],
                frequency=info["frequency"],
            )
            for name, info in ents.items()
        }

        with open(base / "knowledge_graph.pkl", "rb") as f:
            self.graph = pickle.load(f)

        with open(base / "communities.json", "r", encoding="utf-8") as f:
            self.communities = {int(k): v for k, v in json.load(f).items()}

        with open(base / "community_chunks.json", "r", encoding="utf-8") as f:
            self.community_chunks = {
                int(k): v for k, v in json.load(f).items()
            }

        with open(base / "community_summaries.json", "r", encoding="utf-8") as f:
            self.community_summaries = {
                int(k): v for k, v in json.load(f).items()
            }

        arr = np.load(base / "community_embeddings.npy")
        with open(base / "community_embeddings_map.json", "r", encoding="utf-8") as f:
            mp = json.load(f)
        self.community_summary_embeddings = {
            int(cid): arr[idx] for cid, idx in mp.items()
        }

        self.relationships = [
            {
                "source": u,
                "target": v,
                "relation": d.get("relation", "unknown"),
                "weight": d.get("weight", 1),
            }
            for u, v, d in self.graph.edges(data=True)
        ]

        self.processed = True

    def query(self, question: str) -> Dict[str, Any]:
        """Processing user query through local and global RAG search."""
        if not self.processed:
            raise RuntimeError("Run load_and_process() or load_index() first.")

        q_entities = []
        q_canonical = canonicalize(question)

        for cname in self.entities.keys():
            if cname in q_canonical:
                q_entities.append(cname)

        local = self.local_search.search(question, self.graph, self.chunks, self.entities)
        global_res = self.global_search.search(
            question,
            self.communities,
            self.community_summaries,
            self.community_summary_embeddings,
            self.chunks,
            self.community_chunks,
        )

        combined = self.ranker.combine_results(local, global_res)
        retrieved = [c for c, _ in combined]

        chunk_entities = []
        for c in retrieved[:3]:
            for name, e in self.entities.items():
                if c.chunk_id in e.chunk_ids:
                    chunk_entities.append(name)

        rel_summaries = {}
        for c in retrieved[:3]:
            for cid, ids in self.community_chunks.items():
                if c.chunk_id in ids and cid in self.community_summaries:
                    rel_summaries[cid] = self.community_summaries[cid]

        answer = self.answer_generator.generate_answer(
            question,
            retrieved,
            entities=chunk_entities[:10],
            community_summaries=rel_summaries,
        )

        cit = []
        for i, (c, score) in enumerate(combined[:3], 1):
            cit.append(
                {
                    "rank": i,
                    "chunk_id": c.chunk_id,
                    "text": c.text[:300] + "...",
                    "similarity_score": float(score),
                }
            )

        return {
            "answer": answer,
            "citations": cit,
            "metadata": {
                "local_results_count": len(local),
                "global_results_count": len(global_res),
                "combined_results_count": len(combined),
                "query_entities": q_entities[:10],
                "chunk_entities": chunk_entities[:10],
                "communities_used": list(rel_summaries.keys()),
            },
        }

    # ------------------------------------------------------------
    def get_statistics(self):
        if not self.processed:
            return {}
        return {
            "total_chunks": len(self.chunks),
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "communities": len(self.communities),
            "community_summaries": len(self.community_summaries),
        }
