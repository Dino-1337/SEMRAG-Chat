"""Main pipeline for AmbedkarGPT RAG System."""

from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path
import json
import pickle

from src.utils.data_loader import PDFLoader
from src.chunking.semantic_chunker import SemanticChunker, Chunk
from src.chunking.buffer_merger import BufferMerger
from src.graph.entity_extractor import EntityExtractor, Entity
from src.graph.relationship_extractor import RelationshipExtractor
from src.graph.graph_builder import GraphBuilder
from src.graph.community_detector import CommunityDetector
from src.graph.summarizer import CommunitySummarizer
from src.retrieval.local_search import LocalRAGSearch
from src.retrieval.global_search import GlobalRAGSearch
from src.retrieval.ranker import ResultRanker
from src.llm.llm_client import LLMClient
from src.llm.answer_generator import AnswerGenerator
from src.utils.query_expander import QueryExpander
import networkx as nx
import numpy as np


class AmbedkarGPT:
    """Main pipeline orchestrator for SEMRAG RAG system."""
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "index"):
        """
        Initialize AmbedkarGPT system.
        
        Args:
            config_path: Path to configuration file
            mode: "index" for building index, "query" for querying
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mode = mode
        
        # Always initialize these (needed for both modes)
        self.pdf_loader = PDFLoader()
        
        # LLM (needed for both modes)
        llm_config = self.config.get('llm', {})
        self.llm_client = LLMClient(
            model=llm_config.get('model', 'mistral:7b'),
            base_url=llm_config.get('base_url', 'http://localhost:11434'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 1000)
        )
        self.answer_generator = AnswerGenerator(self.llm_client)
        
        # Retrieval (needed for both modes)
        retrieval_config = self.config.get('retrieval', {})
        local_config = retrieval_config.get('local_search', {})
        global_config = retrieval_config.get('global_search', {})
        self.local_search = LocalRAGSearch(
            embedding_model_name=self.config['embedding']['model_name'],
            top_k=local_config.get('top_k', 5),
            threshold=local_config.get('similarity_threshold', 0.2)  # Lowered to 0.2
        )
        self.global_search = GlobalRAGSearch(
            embedding_model_name=self.config['embedding']['model_name'],
            top_k=global_config.get('top_k', 3),
            threshold=global_config.get('similarity_threshold', 0.2)  # Lowered to 0.2
        )
        self.ranker = ResultRanker()
        
        # Query expansion
        self.query_expander = QueryExpander()
        
        # Only initialize heavy components in index mode
        if mode == "index":
            # Semantic chunking
            chunking_config = self.config.get('chunking', {})
            self.chunker = SemanticChunker(
                embedding_model_name=self.config['embedding']['model_name'],
                similarity_threshold=chunking_config.get('similarity_threshold', 0.7),
                max_chunk_tokens=chunking_config.get('max_chunk_tokens', 1024),
                sub_chunk_tokens=chunking_config.get('sub_chunk_tokens', 128),
                chunk_overlap=chunking_config.get('chunk_overlap', 128)
            )
            self.buffer_merger = BufferMerger(
                buffer_size=chunking_config.get('buffer_size', 2),
                max_chunk_tokens=chunking_config.get('max_chunk_tokens', 1024)
            )
            
            # Knowledge graph
            kg_config = self.config.get('knowledge_graph', {})
            self.entity_extractor = EntityExtractor(
                entity_types=kg_config.get('entity_types', ["PERSON", "ORG", "GPE", "DATE", "EVENT", "WORK_OF_ART"]),
                min_entity_frequency=kg_config.get('min_entity_frequency', 1)
            )
            self.relationship_extractor = RelationshipExtractor(self.entity_extractor.nlp)
            self.graph_builder = GraphBuilder()
            self.community_detector = CommunityDetector(
                community_resolution=kg_config.get('community_resolution', 1.0)
            )
            self.community_summarizer = CommunitySummarizer(self.llm_client)
        
        # State
        self.chunks: List[Chunk] = []
        self.graph: Optional[nx.Graph] = None
        self.entities: Dict[str, Entity] = {}
        self.relationships: List = []
        self.communities: Dict[int, List[str]] = {}
        self.community_chunks: Dict[int, List[int]] = {}
        self.community_summaries: Dict[int, str] = {}
        self.community_summary_embeddings: Dict[int, np.ndarray] = {}
        self.processed = False
    
    def load_and_process(self, pdf_path: str, save_processed: bool = True):
        """
        Load PDF and process through the entire pipeline.
        
        Args:
            pdf_path: Path to PDF file
            save_processed: Whether to save processed data
        """
        print("=" * 60)
        print("AmbedkarGPT - Processing PDF")
        print("=" * 60)
        
        # Step 1: Load PDF
        print("\n[1/5] Loading PDF...")
        text = self.pdf_loader.load_pdf(pdf_path)
        print(f"Loaded {len(text)} characters")
        
        # Step 2: Split into sentences
        print("\n[2/5] Splitting into sentences...")
        sentences = self.pdf_loader.split_into_sentences(text)
        print(f"Extracted {len(sentences)} sentences")
        
        # Step 3: Semantic chunking
        print("\n[3/5] Performing semantic chunking...")
        embeddings = self.chunker.compute_sentence_embeddings(sentences)
        chunks = self.chunker.semantic_chunking(sentences, embeddings)
        print(f"Created {len(chunks)} initial chunks")
        
        # Apply buffer merging
        chunks = self.buffer_merger.apply_buffer_merging(chunks)
        print(f"After merging: {len(chunks)} chunks")
        
        # Enforce token limits
        chunks = self.chunker.enforce_token_limits(chunks)
        print(f"Final: {len(chunks)} chunks")
        self.chunks = chunks
        
        # Step 4: Build knowledge graph
        print("\n[4/5] Building knowledge graph...")
        entities = self.entity_extractor.extract_entities(chunks)
        self.entities = {e.text: e for e in entities}
        
        relationships = self.relationship_extractor.extract_relationships(chunks, entities)
        self.relationships = [{'source': r.source, 'target': r.target, 'relation': r.relation, 'weight': 1} 
                             for r in relationships]
        
        self.graph = self.graph_builder.build_graph(entities, self.relationships)
        self.communities = self.community_detector.detect_communities(self.graph)
        self.community_chunks = self.community_detector.get_community_chunks(
            self.communities, self.entities, chunks
        )
        
        # Step 5: Generate community summaries
        print("\n[5/5] Generating community summaries...")
        self.community_summaries = self.community_summarizer.generate_summaries(
            self.communities, self.community_chunks, chunks
        )
        
        # Generate embeddings for summaries
        for comm_id, summary in self.community_summaries.items():
            embedding = self.answer_generator.embedding_model.encode([summary], convert_to_numpy=True)[0]
            self.community_summary_embeddings[comm_id] = embedding
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print(f"Chunks: {len(self.chunks)}")
        print(f"Graph nodes: {self.graph.number_of_nodes()}")
        print(f"Graph edges: {self.graph.number_of_edges()}")
        print(f"Communities: {len(self.communities)}")
        print("=" * 60)
        
        # Save processed data
        if save_processed:
            self._save_processed_data()
        
        self.processed = True
    
    def _save_processed_data(self):
        """Save all processed data to disk."""
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data...")
        
        # 1. Save chunks
        chunks_data = [
            {
                'text': chunk.text,
                'sentences': chunk.sentences,
                'sentence_indices': chunk.sentence_indices,
                'token_count': chunk.token_count,
                'chunk_id': chunk.chunk_id
            }
            for chunk in self.chunks
        ]
        chunks_path = processed_dir / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved chunks: {chunks_path}")
        
        # 2. Save chunk embeddings
        chunk_embeddings = self.local_search.embedding_model.encode(
            [chunk.text for chunk in self.chunks],
            convert_to_numpy=True
        )
        embeddings_path = processed_dir / "chunk_embeddings.npy"
        np.save(embeddings_path, chunk_embeddings)
        print(f"  ✓ Saved chunk embeddings: {embeddings_path}")
        
        # 3. Save entities
        entities_data = {
            name: {
                'text': entity.text,
                'type': entity.type,
                'chunk_ids': entity.chunk_ids,
                'frequency': entity.frequency
            }
            for name, entity in self.entities.items()
        }
        entities_path = processed_dir / "entities.json"
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved entities: {entities_path}")
        
        # 4. Save knowledge graph
        graph_path = processed_dir / "knowledge_graph.pkl"
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"  ✓ Saved knowledge graph: {graph_path}")
        
        # 5. Save communities
        communities_data = {
            str(comm_id): nodes for comm_id, nodes in self.communities.items()
        }
        communities_path = processed_dir / "communities.json"
        with open(communities_path, 'w', encoding='utf-8') as f:
            json.dump(communities_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved communities: {communities_path}")
        
        # 6. Save community chunks
        community_chunks_data = {
            str(comm_id): chunk_ids for comm_id, chunk_ids in self.community_chunks.items()
        }
        community_chunks_path = processed_dir / "community_chunks.json"
        with open(community_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(community_chunks_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved community chunks: {community_chunks_path}")
        
        # 7. Save community summaries
        summaries_data = {
            str(comm_id): summary for comm_id, summary in self.community_summaries.items()
        }
        summaries_path = processed_dir / "community_summaries.json"
        with open(summaries_path, 'w', encoding='utf-8') as f:
            json.dump(summaries_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved community summaries: {summaries_path}")
        
        # 8. Save community summary embeddings
        summary_embeddings_data = {
            comm_id: embedding for comm_id, embedding in self.community_summary_embeddings.items()
        }
        embeddings_array = np.array([summary_embeddings_data[k] for k in sorted(summary_embeddings_data.keys())])
        summary_embeddings_path = processed_dir / "community_embeddings.npy"
        np.save(summary_embeddings_path, embeddings_array)
        
        # Also save the mapping
        embedding_map_path = processed_dir / "community_embeddings_map.json"
        with open(embedding_map_path, 'w') as f:
            json.dump({str(k): i for i, k in enumerate(sorted(summary_embeddings_data.keys()))}, f)
        print(f"  ✓ Saved community embeddings: {summary_embeddings_path}")
        
        # 9. Save metadata
        import hashlib
        config_hash = hashlib.md5(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        metadata = {
            'timestamp': str(Path(chunks_path).stat().st_mtime),
            'config_hash': config_hash,
            'stats': {
                'chunks': len(self.chunks),
                'entities': len(self.entities),
                'graph_nodes': self.graph.number_of_nodes(),
                'graph_edges': self.graph.number_of_edges(),
                'communities': len(self.communities)
            }
        }
        metadata_path = processed_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata: {metadata_path}")
        
        print(f"\n✓ All data saved to: {processed_dir.absolute()}")

    def _validate_index(self) -> bool:
        """
        Validate that all required index files exist.
        
        Returns:
            True if index is valid, False otherwise
        """
        processed_dir = Path("data/processed")
        
        required_files = [
            "chunks.json",
            "chunk_embeddings.npy",
            "entities.json",
            "knowledge_graph.pkl",
            "communities.json",
            "community_chunks.json",
            "community_summaries.json",
            "community_embeddings.npy",
            "community_embeddings_map.json",
            "metadata.json"
        ]
        
        missing_files = []
        for filename in required_files:
            filepath = processed_dir / filename
            if not filepath.exists():
                missing_files.append(filename)
        
        if missing_files:
            print(f"Missing index files: {', '.join(missing_files)}")
            return False
        
        return True
    
    def load_index(self):
        """Load pre-computed index from disk."""
        if not self._validate_index():
            raise FileNotFoundError(
                "Index files not found. Please run 'python build_index.py' first."
            )
        
        processed_dir = Path("data/processed")
        
        print("=" * 60)
        print("Loading Pre-computed Index")
        print("=" * 60)
        
        # 1. Load chunks
        print("\n[1/8] Loading chunks...")
        with open(processed_dir / "chunks.json", 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Reconstruct Chunk objects
        self.chunks = []
        for chunk_dict in chunks_data:
            # Load corresponding embedding
            chunk_embedding = None
            chunk = Chunk(
                text=chunk_dict['text'],
                sentences=chunk_dict['sentences'],
                sentence_indices=chunk_dict['sentence_indices'],
                embedding=chunk_embedding,  # Will load separately
                token_count=chunk_dict['token_count'],
                chunk_id=chunk_dict['chunk_id']
            )
            self.chunks.append(chunk)
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        # 2. Load chunk embeddings
        print("\n[2/8] Loading chunk embeddings...")
        chunk_embeddings = np.load(processed_dir / "chunk_embeddings.npy")
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = chunk_embeddings[i]
        print(f"✓ Loaded embeddings with shape {chunk_embeddings.shape}")
        
        # 3. Load entities
        print("\n[3/8] Loading entities...")
        with open(processed_dir / "entities.json", 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
        
        self.entities = {}
        for name, entity_dict in entities_data.items():
            entity = Entity(
                text=entity_dict['text'],
                label=entity_dict.get('type', entity_dict.get('label', 'UNKNOWN')),
                chunk_ids=entity_dict['chunk_ids'],
                frequency=entity_dict['frequency']
            )
            self.entities[name] = entity
        print(f"✓ Loaded {len(self.entities)} entities")
        
        # 4. Load knowledge graph
        print("\n[4/8] Loading knowledge graph...")
        with open(processed_dir / "knowledge_graph.pkl", 'rb') as f:
            self.graph = pickle.load(f)
        print(f"✓ Loaded graph with {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        # 5. Load communities
        print("\n[5/8] Loading communities...")
        with open(processed_dir / "communities.json", 'r', encoding='utf-8') as f:
            communities_data = json.load(f)
        
        self.communities = {int(k): v for k, v in communities_data.items()}
        print(f"✓ Loaded {len(self.communities)} communities")
        
        # 6. Load community chunks
        print("\n[6/8] Loading community chunks...")
        with open(processed_dir / "community_chunks.json", 'r', encoding='utf-8') as f:
            community_chunks_data = json.load(f)
        
        self.community_chunks = {int(k): v for k, v in community_chunks_data.items()}
        print(f"✓ Loaded community chunk mappings")
        
        # 7. Load community summaries
        print("\n[7/8] Loading community summaries...")
        with open(processed_dir / "community_summaries.json", 'r', encoding='utf-8') as f:
            summaries_data = json.load(f)
        
        self.community_summaries = {int(k): v for k, v in summaries_data.items()}
        print(f"✓ Loaded {len(self.community_summaries)} summaries")
        
        # 8. Load community embeddings
        print("\n[8/8] Loading community embeddings...")
        embeddings_array = np.load(processed_dir / "community_embeddings.npy")
        with open(processed_dir / "community_embeddings_map.json", 'r') as f:
            embedding_map = json.load(f)
        
        self.community_summary_embeddings = {}
        for comm_id_str, idx in embedding_map.items():
            comm_id = int(comm_id_str)
            self.community_summary_embeddings[comm_id] = embeddings_array[idx]
        print(f"✓ Loaded {len(self.community_summary_embeddings)} community embeddings")
        
        # Reconstruct relationships list from graph edges
        self.relationships = [
            {'source': u, 'target': v, 'relation': data.get('relation', 'unknown'), 'weight': data.get('weight', 1)}
            for u, v, data in self.graph.edges(data=True)
        ]
        
        # Load metadata
        with open(processed_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print("\n" + "=" * 60)
        print("Index Loaded Successfully!")
        print("=" * 60)
        print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
        if 'stats' in metadata:
            for key, value in metadata['stats'].items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 60)
        
        self.processed = True
    

    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        if not self.processed:
            raise RuntimeError("System not processed. Call load_and_process() first.")
        
        print(f"\nQuery: {question}")
        print("-" * 60)
        
        # Extract entities from query using query expansion
        expanded_terms = self.query_expander.expand_query(question)
        query_entities = []
        for term in expanded_terms:
            # Check if term matches any entity (case-insensitive)
            for entity_name in self.entities.keys():
                if term.lower() in entity_name.lower() or entity_name.lower() in term.lower():
                    if entity_name not in query_entities:
                        query_entities.append(entity_name)
        
        # Local RAG search
        print("Performing local RAG search...")
        local_results = self.local_search.search(
            question,
            self.graph,
            self.chunks,
            self.entities
        )
        print(f"Found {len(local_results)} chunks from local search")
        
        # Global RAG search
        print("Performing global RAG search...")
        global_results = self.global_search.search(
            question,
            self.communities,
            self.community_summaries,
            self.community_summary_embeddings,
            self.chunks,
            self.community_chunks
        )
        print(f"Found {len(global_results)} chunks from global search")
        
        # Combine results
        combine_strategy = self.config.get('retrieval', {}).get('combine_strategy', 'weighted')
        combined_results = self.ranker.combine_results(
            local_results,
            global_results,
            strategy=combine_strategy
        )
        
        # Get retrieved chunks
        retrieved_chunks = [chunk for chunk, score in combined_results]
        
        # Extract relevant entities from retrieved chunks (for context)
        chunk_entities = []
        for chunk in retrieved_chunks[:3]:
            for entity_name, entity in self.entities.items():
                if chunk.chunk_id in entity.chunk_ids:
                    if entity_name not in chunk_entities:
                        chunk_entities.append(entity_name)
        
        # Get relevant community summaries
        relevant_community_summaries = {}
        for chunk in retrieved_chunks[:3]:
            for comm_id, chunk_ids in self.community_chunks.items():
                if chunk.chunk_id in chunk_ids and comm_id in self.community_summaries:
                    relevant_community_summaries[comm_id] = self.community_summaries[comm_id]
        
        # Generate answer
        print("Generating answer...")
        answer = self.answer_generator.generate_answer(
            question,
            retrieved_chunks,
            entities=chunk_entities[:10],
            community_summaries=relevant_community_summaries
        )
        
        # Prepare citations (show top 3)
        citations = []
        for i, (chunk, score) in enumerate(combined_results[:3], 1):
            citations.append({
                'rank': i,
                'chunk_id': chunk.chunk_id,
                'text': chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                'similarity_score': float(score)
            })
        
        result = {
            'answer': answer,
            'citations': citations,
            'metadata': {
                'local_results_count': len(local_results),
                'global_results_count': len(global_results),
                'combined_results_count': len(combined_results),
                'query_entities': query_entities[:10],  # Entities extracted from query
                'chunk_entities': chunk_entities[:10],  # Entities from retrieved chunks
                'communities_used': list(relevant_community_summaries.keys())
            }
        }
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processed system."""
        if not self.processed:
            return {}
        
        return {
            'total_chunks': len(self.chunks),
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'communities': len(self.communities),
            'community_summaries': len(self.community_summaries)
        }
