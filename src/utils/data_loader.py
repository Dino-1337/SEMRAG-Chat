"""PDF loading and text extraction module using LangChain."""

from typing import List
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber


class PDFLoader:
    """Load and extract text from PDF files."""
    
    def __init__(self):
        """Initialize PDF loader."""
        pass
    
    def load_pdf(self, pdf_path: str) -> str:
        """
        Load PDF and extract all text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
        """
        try:
            # Try using pdfplumber for better text extraction
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            full_text = "\n\n".join(text_parts)
            return full_text
        except Exception as e:
            # Fallback to LangChain's PyPDFLoader
            print(f"Warning: pdfplumber failed, using PyPDFLoader: {e}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            full_text = "\n\n".join([doc.page_content for doc in documents])
            return full_text
    
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
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        page_texts.append({
                            'page_num': page_num,
                            'text': page_text
                        })
        except Exception as e:
            print(f"Warning: Could not extract page texts: {e}")
            full_text = self.load_pdf(pdf_path)
            if full_text:
                page_texts.append({
                    'page_num': 1,
                    'text': full_text
                })
        
        return page_texts

