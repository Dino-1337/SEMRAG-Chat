"""Build index once - processes PDF and saves all artifacts."""

import sys
from pathlib import Path
from src.pipeline.index_builder import IndexBuilder


def main():
    """Building and saving the index for SEMRAG system."""
    
    pdf_path = Path("data/Ambedkar_works.pdf")
    if not pdf_path.exists():
        print("=" * 60)
        print("ERROR: Ambedkar_works.pdf not found!")
        print("=" * 60)
        print(f"Please place the PDF file at: {pdf_path.absolute()}")
        print("=" * 60)
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Building Index for AmbedkarGPT")
    print("=" * 60)
    print("This will process the PDF and save all artifacts.")
    print("This only needs to be done ONCE.")
    print("=" * 60)
    
    try:
        builder = IndexBuilder("config.yaml")
        stats = builder.build_index(str(pdf_path))
        
        print("\n" + "=" * 60)
        print("INDEX STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\n" + "=" * 60)
        print("✓ Index built successfully!")
        print("=" * 60)
        print("You can now run: python run_app.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error building index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
