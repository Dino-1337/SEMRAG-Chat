"""Interactive chat interface for AmbedkarGPT."""

import sys
import os
from pathlib import Path

# Disable TensorFlow
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

from src.pipeline.ambedkargpt import AmbedkarGPT


def print_separator(char="=", length=60):
    """Print a separator line."""
    print(char * length)


def print_answer(result):
    """Print formatted answer with citations."""
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result['answer'])
    
    # Print citations
    if result['citations']:
        print(f"\n{'='*60}")
        print(f"CITATIONS ({len(result['citations'])})")
        print("=" * 60)
        for citation in result['citations']:
            print(f"\n[{citation['rank']}] Relevance Score: {citation['similarity_score']:.3f}")
            print(f"Chunk ID: {citation.get('chunk_id', 'N/A')}")
            print(f"Text: {citation['text'][:300]}...")
            print("-" * 60)
    else:
        print("\n⚠️  No citations found - answer may not be based on the document")
    
    # Print metadata
    print(f"\n{'='*60}")
    print("SEARCH METADATA")
    print("=" * 60)
    print(f"Local Search Results: {result['metadata']['local_results_count']}")
    print(f"Global Search Results: {result['metadata']['global_results_count']}")
    print(f"Combined Results: {result['metadata']['combined_results_count']}")
    
    if result['metadata'].get('query_entities'):
        print(f"\nEntities Detected in Your Question:")
        for entity in result['metadata']['query_entities']:
            print(f"  • {entity}")
    
    if result['metadata'].get('chunk_entities'):
        print(f"\nEntities Found in Retrieved Chunks:")
        for entity in result['metadata']['chunk_entities'][:10]:
            print(f"  • {entity}")
    
    print("=" * 60)


def main():
    """Run interactive chat interface."""
    
    # Check if index exists
    metadata_path = Path("data/processed/metadata.json")
    if not metadata_path.exists():
        print("=" * 60)
        print("ERROR: Pre-built index not found!")
        print("=" * 60)
        print("The SEMRAG system requires a pre-built index to run.")
        print("\nPlease build the index first by running:")
        print("  python build_index.py")
        print("\nThis is a one-time process that will:")
        print("  - Process the PDF document")
        print("  - Build the knowledge graph")
        print("  - Generate community summaries")
        print("  - Save all artifacts for fast loading")
        print("\nAfter building the index, you can run this script again.")
        print("=" * 60)
        sys.exit(1)
    
    # Initialize system in query mode
    print("\n" + "=" * 60)
    print("AMBEDKARGPT - Interactive RAG System")
    print("=" * 60)
    print("Loading pre-built index...")
    
    try:
        gpt = AmbedkarGPT("config.yaml", mode="query")
        gpt.load_index()
    except Exception as e:
        print(f"Error loading index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display statistics
    stats = gpt.get_statistics()
    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
    print(f"Chunks: {stats['total_chunks']}")
    print(f"Entities: {stats['total_entities']}")
    print(f"Relationships: {stats['total_relationships']}")
    print(f"Communities: {stats['communities']}")
    print("=" * 60)
    
    print("\n💡 Tips:")
    print("  • Ask questions about Dr. Ambedkar's work")
    print("  • Type 'quit' or 'exit' to end the session")
    print("  • Type 'stats' to see system statistics")
    print("  • Type 'help' for more commands")
    print()
    
    # Interactive loop
    while True:
        try:
            # Get user input
            print("\n" + "=" * 60)
            question = input("Your Question: ").strip()
            print("=" * 60)
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'stats':
                print("\nSystem Statistics:")
                for key, value in stats.items():
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                continue
            
            if question.lower() == 'help':
                print("\nAvailable Commands:")
                print("  quit/exit - Exit the program")
                print("  stats     - Show system statistics")
                print("  help      - Show this help message")
                print("\nJust type your question to get an answer!")
                continue
            
            # Process query
            print("\nProcessing your question...")
            print("-" * 60)
            
            try:
                result = gpt.query(question)
                print_answer(result)
                
            except Exception as e:
                print(f"\n❌ Error processing question: {e}")
                import traceback
                traceback.print_exc()
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
