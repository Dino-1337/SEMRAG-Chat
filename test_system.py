#!/usr/bin/env python
"""Quick test script for SEMRAG system"""

from src.pipeline.ambedkargpt import AmbedkarGPT

def main():
    print("\\n" + "="*60)
    print("SEMRAG SYSTEM TEST")
    print("="*60)
    
    # Load system
    print("\\nLoading index...")
    gpt = AmbedkarGPT("config.yaml", mode="query")
    gpt.load_index()
    
    print(f"✓ Loaded: {len(gpt.chunks)} chunks, {len(gpt.entities)} entities")
    
    # Test query
    question = "What is Ambedkar's explanation for the origin of the caste system?"
    print(f"\\nQuestion: {question}")
    print("\\nProcessing...")
    
    result = gpt.query(question)
    
    # Display results
    print("\\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(result["answer"])
    
    print("\\n" + "="*60)
    print("TOP 3 CITATIONS:")
    print("="*60)
    for i, cit in enumerate(result["citations"][:3], 1):
        print(f"\\n[{i}] Chunk ID: {cit['chunk_id']} | Score: {cit['similarity_score']:.3f}")
        print(f"Text: {cit['text'][:200]}...")
    
    print("\\n" + "="*60)
    print("METADATA:")
    print("="*60)
    meta = result["metadata"]
    print(f"Local results: {meta['local_results_count']}")
    print(f"Global results: {meta['global_results_count']}")
    print(f"Combined results: {meta['combined_results_count']}")
    print(f"Query entities: {meta['query_entities'][:5]}")
    print(f"Communities used: {meta['communities_used']}")
    
    print("\\n" + "="*60)
    print("✓ TEST COMPLETE")
    print("="*60 + "\\n")

if __name__ == "__main__":
    main()
