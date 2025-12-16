"""Build index once - processes PDF and saves all artifacts."""

import sys
from pathlib import Path
from src.pipeline.index_builder import IndexBuilder


def main():
    """Building and saving the index for SEMRAG system."""
    
    pdf_path = Path("data/Ambedkar_works.pdf")
    if not pdf_path.exists():
        print("\nERROR: Ambedkar_works.pdf not found!")
        print(f"Please place the PDF file at: {pdf_path.absolute()}\n")
        sys.exit(1)
    
    print("\nBuilding Index for SEMRAG")
    print("─" * 25)
    print("This will process the PDF and save all artifacts.")
    print("This only needs to be done ONCE.\n")
    
    try:
        builder = IndexBuilder("config.yaml")
        stats = builder.build_index(str(pdf_path))
        
        print("\nIndex Statistics")
        print("─" * 16)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\n✓ Index built successfully!")
        print("You can now run: python app.py\n")
        
    except Exception as e:
        print(f"\n✗ Error building index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
