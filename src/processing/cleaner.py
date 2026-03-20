"""
Document text cleaning and normalization.
"""

import re
import unicodedata
import logging
from typing import List

from llama_index.core import Document

logger = logging.getLogger(__name__)


class DocumentCleaner:
    """
    Cleans and normalizes document text.
    """

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Apply cleaning operations to a single text string.
        """
        if not text:
            return ""

        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with a double newline to preserve paragraph structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove control characters except tab and newline
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        return text.strip()

    def clean_document(self, document: Document) -> Document:
        """
        Clean the text of a LlamaIndex Document.
        """
        document.text = self.clean_text(document.text)
        return document

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean a list of documents.
        """
        cleaned_docs = []
        for doc in documents:
            try:
                cleaned_docs.append(self.clean_document(doc))
            except Exception as e:
                logger.error(f"Error cleaning document {doc.doc_id}: {e}")
                cleaned_docs.append(doc)  # Keep original if cleaning fails
        
        logger.info(f"Cleaned {len(cleaned_docs)} documents.")
        return cleaned_docs
