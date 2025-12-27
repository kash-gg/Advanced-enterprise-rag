"""
PDF Document Loader
Loads and extracts text from PDF files with metadata extraction.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from llama_index.core import Document
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import PDFReader
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loader for PDF documents with metadata extraction.
    
    Features:
    - Extract text from PDF files
    - Preserve page numbers and structure
    - Extract metadata (title, author, creation date)
    - Support batch processing
    - Handle multi-page documents
    """
    
    def __init__(self):
        """Initialize the PDF loader."""
        self.pdf_reader = PDFReader()
        logger.info("PDFLoader initialized")
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a single PDF file and extract text with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects, one per page
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            if not file_path.suffix.lower() == '.pdf':
                raise ValueError(f"File is not a PDF: {file_path}")
            
            logger.info(f"Loading PDF: {file_path}")
            
            # Extract metadata using pypdf
            metadata = self._extract_metadata(file_path)
            
            # Load documents using LlamaIndex PDFReader
            documents = self.pdf_reader.load_data(file=file_path)
            
            # Enrich documents with metadata
            for idx, doc in enumerate(documents):
                doc.metadata.update(metadata)
                doc.metadata['page_number'] = idx + 1
                doc.metadata['source_type'] = 'pdf'
                doc.metadata['file_path'] = str(file_path)
                doc.metadata['file_name'] = file_path.name
                doc.metadata['loaded_at'] = datetime.now().isoformat()
            
            logger.info(f"Successfully loaded {len(documents)} pages from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def load_batch(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple PDF files in batch.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            Combined list of Document objects from all PDFs
        """
        all_documents = []
        successful = 0
        failed = 0
        
        for file_path in file_paths:
            try:
                documents = self.load_pdf(file_path)
                all_documents.extend(documents)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                failed += 1
        
        logger.info(f"Batch loading complete: {successful} successful, {failed} failed")
        return all_documents
    
    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {}
        
        try:
            with open(file_path, 'rb') as f:
                pdf = PdfReader(f)
                pdf_metadata = pdf.metadata
                
                if pdf_metadata:
                    # Extract common metadata fields
                    metadata['title'] = pdf_metadata.get('/Title', file_path.stem)
                    metadata['author'] = pdf_metadata.get('/Author', 'Unknown')
                    metadata['subject'] = pdf_metadata.get('/Subject', '')
                    metadata['creator'] = pdf_metadata.get('/Creator', '')
                    metadata['producer'] = pdf_metadata.get('/Producer', '')
                    
                    # Handle creation date
                    creation_date = pdf_metadata.get('/CreationDate')
                    if creation_date:
                        metadata['creation_date'] = str(creation_date)
                
                # Add page count
                metadata['total_pages'] = len(pdf.pages)
                
        except Exception as e:
            logger.warning(f"Could not extract metadata from {file_path}: {str(e)}")
            metadata['title'] = file_path.stem
            metadata['author'] = 'Unknown'
        
        return metadata
    
    def load_from_directory(self, directory_path: str, recursive: bool = False) -> List[Document]:
        """
        Load all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            recursive: Whether to search subdirectories
            
        Returns:
            List of Document objects from all PDFs found
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find PDF files
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        # Load all PDFs
        file_paths = [str(f) for f in pdf_files]
        return self.load_batch(file_paths)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = PDFLoader()
    
    # Example: Load a single PDF
    # documents = loader.load_pdf("path/to/document.pdf")
    
    # Example: Load from directory
    # documents = loader.load_from_directory("path/to/pdfs", recursive=True)
    
    print("PDFLoader initialized and ready to use!")
