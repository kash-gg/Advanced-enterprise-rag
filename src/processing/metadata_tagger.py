"""
Metadata extraction and enrichment.
"""

import logging
import re
from typing import List
from collections import Counter

from llama_index.core import Document

logger = logging.getLogger(__name__)


class MetadataTagger:
    """
    Extracts and adds metadata to documents.
    """

    def __init__(self):
        pass

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Simple keyword extraction using word frequencies."""
        # Lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        # Filter out common stop words (a minimal list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'it', 'this', 'that'}
        words = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Get most common
        counter = Counter(words)
        return [word for word, count in counter.most_common(top_n)]

    def tag_document(self, document: Document) -> Document:
        """
        Enrich a document's metadata.
        """
        # Ensure metadata exists
        if document.metadata is None:
            document.metadata = {}
            
        text = document.text
        
        # Add basic document length
        document.metadata['char_count'] = len(text)
        document.metadata['word_count'] = len(text.split())
        
        # Extract keywords if not present
        if 'keywords' not in document.metadata:
            document.metadata['keywords'] = self.extract_keywords(text)
            
        # Try to infer title from first line if not present
        if 'title' not in document.metadata:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                document.metadata['title'] = lines[0][:100]  # First line, max 100 chars
            else:
                document.metadata['title'] = "Untitled Document"
                
        return document

    def tag_documents(self, documents: List[Document]) -> List[Document]:
        """
        Tag a list of documents.
        """
        tagged_docs = []
        for doc in documents:
            try:
                tagged_docs.append(self.tag_document(doc))
            except Exception as e:
                logger.error(f"Error tagging document {doc.doc_id}: {e}")
                tagged_docs.append(doc)
                
        logger.info(f"Tagged {len(tagged_docs)} documents.")
        return tagged_docs
