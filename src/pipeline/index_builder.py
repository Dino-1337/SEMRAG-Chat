"""Index builder for SEMRAG - processes PDF and saves all artifacts."""

from typing import Dict, List, Any
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml

from src.utils.data_loader import PDFLoader
from src.chunking.semantic_chunker import SemanticChunker, Chunk
from src.chunking.buffer_merger import BufferMerger
from src.graph.entity_extractor import EntityExtractor, Entity
from src.graph.relationship_extractor import RelationshipExtractor, Relationship
from src.graph.graph_builder import GraphBuilder
from src.graph.community_detector import CommunityDetector
from src.graph.summarizer import CommunitySummarizer
from src.llm.llm_client import LLMClient
import networkx as nx


class IndexBuilder:
    """Builds and saves the complete index for SEMRAG."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize index builder.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("Initializing components...")
        
        # Chunking
        self.chunker = SemanticChunker(
            embedding_model_name=self.config['embedding']['model_name'],
            similarity_threshold=self.config['chunking']['similarity_threshold'],
            max_chunk_tokens=self.config['chunking']['max_chunk_tokens'],
            sub_chunk_tokens=self.config['chunking']['sub_chunk_tokens'],
            chunk_overlap=self.config['chunking']['chunk_overlap']
        )
        
        self.buffer_merger = BufferMerger(
            buffer_size=self.config['chunking']['buffer_size']
        )
        
        # Knowledge Graph
        self.entity_extractor = EntityExtractor(
            entity_types=self.config['knowledge_graph']['entity_types'],
            min_entity_frequency=self.config['knowledge_graph']['min_entity_frequency']
        )
        
        self.relationship_extractor = RelationshipExtractor(self.entity_extractor.nlp)
        self.graph_builder = GraphBuilder()
        self.community_detector = CommunityDetector(
            community_resolution=self.config['knowledge_graph']['community_resolution']
        )
        
        # LLM for community summaries
        self.llm_client = LLMClient(
            model=self.config['llm']['model'],
            base_url=self.config['llm']['base_url'],
            temperature=self.config['llm']['temperature']
        )
        
        self.summarizer = CommunitySummarizer(self.llm_client)
        
        # Storage for processed data
        self.chunks: List[Chunk] = []
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.graph: nx.Graph = None
        self.communities: Dict[int, List[str]] = {}
        self.community_summaries: Dict[int, str] = {}
        self.community_summary_embeddings: Dict[int, np.ndarray] = {}
        
        print("✓ Components initialized")
    
    def build_index(self, pdf_path: str) -> Dict[str, Any]:
        """
        Build complete index from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with statistics
        """
        print("\n" + "="*60)
        print("BUILDING INDEX")
        print("="*60)
        
        # Step 1: Load PDF
        print("\n[1/6] Loading PDF...")
        loader = PDFLoader()
        text = loader.load_pdf(pdf_path)
        sentences = loader.split_into_sentences(text)
        print(f"✓ Loaded {len(sentences)} sentences")
        
        # Step 2: Semantic Chunking
        print("\n[2/6] Performing semantic chunking...")
        embeddings = self.chunker.compute_sentence_embeddings(sentences)
        chunks = self.chunker.semantic_chunking(sentences, embeddings)
        chunks = self.chunker.enforce_token_limits(chunks)
        self.chunks = self.buffer_merger.apply_buffer_merging(chunks)
        print(f"✓ Created {len(self.chunks)} chunks")
        
        # Step 3: Entity Extraction
        print("\n[3/6] Extracting entities...")
        entities_list = self.entity_extractor.extract_entities(self.chunks)
        self.entities = {e.text: e for e in entities_list}
        print(f"✓ Extracted {len(self.entities)} entities")
        
        # Step 4: Build Knowledge Graph
        print("\n[4/6] Building knowledge graph...")
        self.relationships = self.relationship_extractor.extract_relationships(
            self.chunks, entities_list
        )
        
        # Convert Relationship objects to dictionaries for graph builder
        relationships_dict = [
            {
                'source': r.source,
                'target': r.target,
                'relation': r.relation,
                'weight': 1
            }
            for r in self.relationships
        ]
        
        self.graph = self.graph_builder.build_graph(
            entities_list, relationships_dict
        )
        print(f"✓ Built graph with {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        # Step 5: Community Detection
        print("\n[5/6] Detecting communities...")
        self.communities = self.community_detector.detect_communities(self.graph)
        self.community_chunks = self.community_detector.get_community_chunks(
            self.communities,
            self.entities,
            self.chunks
        )
        print(f"✓ Detected {len(self.communities)} communities")
        
        # Step 6: Generate Community Summaries
        print("\n[6/6] Generating community summaries...")
        self.community_summaries = self.summarizer.generate_summaries(
            self.communities,
            self.community_chunks,
            self.chunks
        )
        
        # Generate embeddings for summaries
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(self.config['embedding']['model_name'])
        
        for comm_id, summary in self.community_summaries.items():
            embedding = embedding_model.encode([summary], convert_to_numpy=True)[0]
            self.community_summary_embeddings[comm_id] = embedding
        
        print(f"✓ Generated {len(self.community_summaries)} summaries with embeddings")

        
        # Save all artifacts
        print("\n" + "="*60)
        print("SAVING ARTIFACTS")
        print("="*60)
        self._save_artifacts()
        
        # Return statistics
        stats = self._get_statistics()
        return stats
    
    def _save_artifacts(self):
        """Save all processed artifacts to disk."""
        
        # 1. Save chunks
        print("\n[1/8] Saving chunks...")
        chunks_data = [
            {
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                'sentences': chunk.sentences,
                'sentence_indices': chunk.sentence_indices,
                'token_count': chunk.token_count
            }
            for chunk in self.chunks
        ]
        with open(self.processed_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(chunks_data)} chunks")
        
        # 2. Save chunk embeddings
        print("\n[2/8] Saving chunk embeddings...")
        chunk_embeddings = np.array([chunk.embedding for chunk in self.chunks])
        np.save(self.processed_dir / "chunk_embeddings.npy", chunk_embeddings)
        print(f"✓ Saved embeddings with shape {chunk_embeddings.shape}")
        
        # 3. Save entities
        print("\n[3/8] Saving entities...")
        entities_data = {
            name: {
                'text': entity.text,
                'label': entity.label,
                'chunk_ids': entity.chunk_ids,
                'frequency': entity.frequency
            }
            for name, entity in self.entities.items()
        }
        with open(self.processed_dir / "entities.json", 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(entities_data)} entities")
        
        # 4. Save knowledge graph
        print("\n[4/8] Saving knowledge graph...")
        with open(self.processed_dir / "knowledge_graph.pkl", 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"✓ Saved graph with {self.graph.number_of_nodes()} nodes")
        
        # 5. Save communities
        print("\n[5/8] Saving communities...")
        communities_data = {
            str(comm_id): entity_names
            for comm_id, entity_names in self.communities.items()
        }
        with open(self.processed_dir / "communities.json", 'w', encoding='utf-8') as f:
            json.dump(communities_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(communities_data)} communities")
        
        # 6. Save community chunks
        print("\n[6/10] Saving community chunks...")
        community_chunks_data = {
            str(comm_id): chunk_ids
            for comm_id, chunk_ids in self.community_chunks.items()
        }
        with open(self.processed_dir / "community_chunks.json", 'w', encoding='utf-8') as f:
            json.dump(community_chunks_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(community_chunks_data)} community chunk mappings")
        
        # 7. Save community summaries
        print("\n[7/10] Saving community summaries...")
        summaries_data = {
            str(comm_id): summary
            for comm_id, summary in self.community_summaries.items()
        }
        with open(self.processed_dir / "community_summaries.json", 'w', encoding='utf-8') as f:
            json.dump(summaries_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(summaries_data)} summaries")
        
        # 8. Save community embeddings (NPY format)
        print("\n[8/10] Saving community embeddings...")
        # Save as numpy array with separate mapping file
        sorted_comm_ids = sorted(self.community_summary_embeddings.keys())
        embeddings_array = np.array([self.community_summary_embeddings[cid] for cid in sorted_comm_ids])
        np.save(self.processed_dir / "community_embeddings.npy", embeddings_array)
        
        # Save the mapping
        embedding_map = {str(cid): idx for idx, cid in enumerate(sorted_comm_ids)}
        with open(self.processed_dir / "community_embeddings_map.json", 'w') as f:
            json.dump(embedding_map, f, indent=2)
        print(f"✓ Saved {len(sorted_comm_ids)} community embeddings")
        
        # 9. Save metadata
        print("\n[9/10] Saving metadata...")
        config_hash = self._compute_config_hash()
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config_hash': config_hash,
            'statistics': self._get_statistics()
        }
        with open(self.processed_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata")
        
        # 10. Cleanup - remove old JSON format if exists
        print("\n[10/10] Cleanup...")
        old_json = self.processed_dir / "community_embeddings.json"
        if old_json.exists():
            old_json.unlink()
            print("✓ Removed old community_embeddings.json")
        else:
            print("✓ Cleanup complete")
        
        print("\n" + "="*60)
        print("✓ ALL ARTIFACTS SAVED SUCCESSFULLY")
        print("="*60)
    
    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for cache validation."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processed data."""
        return {
            'num_chunks': len(self.chunks),
            'num_entities': len(self.entities),
            'num_relationships': len(self.relationships),
            'num_graph_nodes': self.graph.number_of_nodes() if self.graph else 0,
            'num_graph_edges': self.graph.number_of_edges() if self.graph else 0,
            'num_communities': len(self.communities),
            'num_summaries': len(self.community_summaries),
            'total_tokens': sum(chunk.token_count for chunk in self.chunks),
            'avg_chunk_size': np.mean([chunk.token_count for chunk in self.chunks]) if self.chunks else 0
        }
