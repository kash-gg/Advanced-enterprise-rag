"""
Document chunking strategies.
"""

import logging
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Chunks documents using various strategies.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_sentence_splitter(self):
        """Sentence-aware chunking."""
        return SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
    def get_token_splitter(self):
        """Fixed-size token chunking."""
        return TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk_documents(self, documents: List[Document], strategy: str = "sentence") -> List[Any]:
        """
        Split a list of documents into chunks (nodes).
        """
        if strategy == "sentence":
            parser = self.get_sentence_splitter()
        elif strategy == "token":
            parser = self.get_token_splitter()
        else:
            logger.warning(f"Unknown strategy {strategy}, falling back to sentence.")
            parser = self.get_sentence_splitter()

        nodes = parser.get_nodes_from_documents(documents)
        logger.info(f"Chunked {len(documents)} documents into {len(nodes)} nodes using {strategy} strategy.")
        
        return nodes
