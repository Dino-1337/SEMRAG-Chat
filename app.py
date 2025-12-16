# app.py
import os
import sys
from pathlib import Path

# Disable TensorFlow everywhere
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from src.pipeline.ambedkargpt import AmbedkarGPT


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def show_answer(result):
    section("ANSWER")
    print(result["answer"])

    if result["citations"]:
        section(f"CITATIONS ({len(result['citations'])})")
        for c in result["citations"]:
            print(f"[{c['rank']}] Score: {c['similarity_score']:.3f}")
            print(f"Chunk {c['chunk_id']}: {c['text'][:280]}...\n")
    else:
        print("\n⚠ No citations found — the answer may not be grounded.")

    section("SEARCH METADATA")
    md = result["metadata"]
    print(f"Local Matches:   {md['local_results_count']}")
    print(f"Global Matches:  {md['global_results_count']}")
    print(f"Combined Matches:{md['combined_results_count']}")

    if md.get("query_entities"):
        print("\nEntities in Query:")
        for e in md["query_entities"]:
            print(" •", e)

    if md.get("chunk_entities"):
        print("\nEntities in Retrieved Chunks:")
        for e in md["chunk_entities"][:10]:
            print(" •", e)


def main():
    """Main CLI for querying the SEMRAG system."""
    metadata_path = Path("data/processed/metadata.json")

    if not metadata_path.exists():
        section("ERROR: INDEX NOT FOUND")
        print("You must build the index first:")
        print("  python build_index.py\n")
        sys.exit(1)

    # Load GPT system
    section("AmbedkarGPT — Loading Index")
    gpt = AmbedkarGPT("config.yaml", mode="query")
    gpt.load_index()

    stats = gpt.get_statistics()
    section("SYSTEM READY")
    print(f"Chunks:       {stats['total_chunks']}")
    print(f"Entities:     {stats['total_entities']}")
    print(f"Relations:    {stats['total_relationships']}")
    print(f"Communities:  {stats['communities']}")

    print("\nCommands:")
    print("  /help   Show commands")
    print("  /stats  Show statistics")
    print("  /exit   Quit")
    print()

    # Interactive loop
    while True:
        try:
            print("\n" + "-" * 60)
            query = input("Your Question: ").strip()
            print("-" * 60)

            if not query:
                continue

            # Commands
            if query.lower() in ["/exit", "exit", "quit"]:
                print("\nGoodbye 👋")
                break

            if query.lower() == "/stats":
                section("SYSTEM STATISTICS")
                for k, v in stats.items():
                    print(f"{k.replace('_',' ').title()}: {v}")
                continue

            if query.lower() == "/help":
                print("\nCommands:")
                print("  /help   Show commands")
                print("  /stats  Show statistics")
                print("  /exit   Quit")
                continue

            # RAG query
            print("\nProcessing...")
            result = gpt.query(query)
            show_answer(result)

        except KeyboardInterrupt:
            print("\n\nGoodbye 👋")
            break
        except EOFError:
            print("\n\nGoodbye 👋")
            break


if __name__ == "__main__":
    main()
