"""PDF loading and text extraction module using PyMuPDF."""

from typing import List
import fitz  # PyMuPDF


class PDFLoader:
    """Load and extract text from PDF files using PyMuPDF (fitz)."""
    
    def __init__(self):
        """Initialize PDF loader."""
        pass
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load PDF and extract all text using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
        """
        text_parts = []
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")  # Extract plain text
                
                if page_text.strip():
                    text_parts.append(page_text)
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            return full_text
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF with PyMuPDF: {e}")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        import re
        
        # Split on sentence endings, but preserve them
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences and clean them
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def get_page_texts(self, pdf_path: str) -> List[dict]:
        """
        Get text from each page separately for citation purposes.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dicts with 'page_num' and 'text' keys
        """
        page_texts = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                
                if page_text.strip():
                    page_texts.append({
                        'page_num': page_num + 1,  # 1-indexed for user display
                        'text': page_text
                    })
            
            doc.close()
            
        except Exception as e:
            print(f"Warning: Could not extract page texts: {e}")
            # Fallback to full text
            full_text = self.load_pdf(pdf_path)
            if full_text:
                page_texts.append({
                    'page_num': 1,
                    'text': full_text
                })
        
        return page_texts

