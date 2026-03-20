"""
Document quality checking.
"""

import logging
from typing import List

from llama_index.core import Document

logger = logging.getLogger(__name__)


class DocumentQualityChecker:
    """
    Checks documents for quality issues before indexing.
    """

    def __init__(self, min_length: int = 50, require_title: bool = False):
        self.min_length = min_length
        self.require_title = require_title

    def check_document(self, document: Document) -> bool:
        """
        Check if a single document meets quality standards.
        Returns True if it passes, False otherwise.
        """
        if not document.text or len(document.text.strip()) < self.min_length:
            logger.warning(f"Document {document.doc_id} failed length check ({len(document.text)} chars).")
            return False
            
        # Check alphanumeric ratio to avoid gibberish
        text = document.text
        alnum_count = sum(c.isalnum() for c in text)
        if alnum_count / len(text) < 0.5:
            logger.warning(f"Document {document.doc_id} failed alphanumeric ratio check.")
            return False

        if self.require_title:
            if not document.metadata or 'title' not in document.metadata:
                logger.warning(f"Document {document.doc_id} is missing a required title.")
                return False

        return True

    def filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        Filter a list of documents, keeping only those that pass quality checks.
        """
        passed_docs = []
        for doc in documents:
            if self.check_document(doc):
                passed_docs.append(doc)
                
        logger.info(f"{len(passed_docs)}/{len(documents)} documents passed quality checks.")
        return passed_docs
