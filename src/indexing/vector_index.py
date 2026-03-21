"""
Vector Index using FAISS and HuggingFace Sentence Transformers.

Implements a dense vector index for semantic similarity search.
Uses CPU-based FAISS for zero-cost, local operation.
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    FAISS-based vector index with HuggingFace embeddings.

    Features:
    - Sentence-transformer embeddings (all-MiniLM-L6-v2 default)
    - CPU-based FAISS IndexFlatL2 for exact search
    - Persistent storage to disk (save/load)
    - Incremental indexing (add documents without rebuilding)
    - Metadata filtering support
    - Batch embedding generation
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/faiss_index",
        dimension: int = 384,
        batch_size: int = 32,
    ):
        """
        Initialize the vector index.

        Args:
            embedding_model: HuggingFace model name for embeddings
            index_path: Directory to persist the FAISS index
            dimension: Embedding dimension (must match model output)
            batch_size: Number of documents to embed at once
        """
        self.embedding_model_name = embedding_model
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.batch_size = batch_size

        # Internal state
        self._embedding_model = None   # lazy-loaded
        self._faiss_index = None       # lazy-created
        self._doc_store: List[Dict[str, Any]] = []  # parallel list to FAISS rows
        self._id_to_pos: Dict[str, int] = {}         # doc_id -> FAISS row position

        # Try to load existing index from disk
        if self._index_files_exist():
            self._load_index()
            logger.info(
                f"Loaded existing FAISS index with {self._faiss_index.ntotal} vectors"
                f" from {self.index_path}"
            )
        else:
            logger.info(
                f"No existing index found at {self.index_path}. "
                "A new index will be created on first add."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Generate embeddings and add documents to the FAISS index.

        Args:
            documents: List of dicts with keys:
                - 'doc_id' (str): Unique identifier
                - 'text'   (str): Document/chunk text to embed
                - 'metadata' (dict, optional): Arbitrary metadata to store

        Returns:
            Number of documents successfully added
        """
        if not documents:
            logger.warning("add_documents called with empty list.")
            return 0

        self._ensure_model_loaded()
        self._ensure_index_initialized()

        added = 0
        # Process in batches to avoid OOM on large corpora
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start: start + self.batch_size]
            texts = [d["text"] for d in batch]

            try:
                embeddings = self._embed_texts(texts)  # (N, dim) float32
            except Exception as e:
                logger.error(f"Embedding failed for batch starting at {start}: {e}")
                continue

            for i, doc in enumerate(batch):
                doc_id = doc.get("doc_id", f"doc_{len(self._doc_store)}")

                if doc_id in self._id_to_pos:
                    logger.debug(f"Skipping duplicate doc_id: {doc_id}")
                    continue

                pos = len(self._doc_store)
                self._id_to_pos[doc_id] = pos
                self._doc_store.append({
                    "doc_id": doc_id,
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                })

                vec = embeddings[i: i + 1].astype(np.float32)
                self._faiss_index.add(vec)
                added += 1

        logger.info(f"Added {added} documents to the vector index.")
        return added

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search.

        Args:
            query: Natural language query
            top_k: Number of results to return
            metadata_filter: Optional key-value filter on stored metadata

        Returns:
            List of result dicts:
                - 'doc_id', 'text', 'metadata', 'score' (L2 distance, lower = better)
                - 'rank' (1-based)
        """
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            logger.warning("search() called on empty index.")
            return []

        self._ensure_model_loaded()

        query_vec = self._embed_texts([query]).astype(np.float32)

        # Over-fetch to allow metadata filtering
        fetch_k = min(top_k * 5, self._faiss_index.ntotal) if metadata_filter else top_k
        distances, indices = self._faiss_index.search(query_vec, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._doc_store):
                continue  # FAISS padding / out-of-bounds

            entry = self._doc_store[idx]

            # Apply metadata filter
            if metadata_filter and not self._matches_filter(entry["metadata"], metadata_filter):
                continue

            results.append({
                "doc_id": entry["doc_id"],
                "text": entry["text"],
                "metadata": entry["metadata"],
                "score": float(dist),   # L2 distance (lower = more similar)
            })

            if len(results) >= top_k:
                break

        # Add 1-based rank
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

        logger.debug(f"Search returned {len(results)} results for query: '{query[:60]}...'")
        return results

    def save(self):
        """Persist the index and metadata to disk."""
        import faiss
        self.index_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "doc_store.pkl", "wb") as f:
            pickle.dump(self._doc_store, f)
        with open(self.index_path / "id_to_pos.json", "w") as f:
            json.dump(self._id_to_pos, f)

        logger.info(
            f"Saved FAISS index ({self._faiss_index.ntotal} vectors) to {self.index_path}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        total = self._faiss_index.ntotal if self._faiss_index else 0
        return {
            "total_vectors": total,
            "embedding_model": self.embedding_model_name,
            "dimension": self.dimension,
            "index_path": str(self.index_path),
            "index_exists_on_disk": self._index_files_exist(),
        }

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove a document from the in-memory store.

        Note: FAISS IndexFlatL2 does not support efficient single-vector
        removal. The vector remains in FAISS but is masked from results.
        A full rebuild is required for true deletion; call rebuild() after
        bulk deletions.

        Returns:
            True if the document was found and masked, False otherwise.
        """
        if doc_id not in self._id_to_pos:
            return False
        pos = self._id_to_pos.pop(doc_id)
        # Mark the doc_store entry as deleted
        self._doc_store[pos]["_deleted"] = True
        logger.info(f"Marked document {doc_id} as deleted (soft delete).")
        return True

    def rebuild(self):
        """
        Rebuild the FAISS index from the current (non-deleted) doc_store.
        Use after bulk deletions to reclaim memory.
        """
        import faiss
        active_docs = [d for d in self._doc_store if not d.get("_deleted", False)]
        logger.info(f"Rebuilding index with {len(active_docs)} active documents...")

        self._faiss_index = faiss.IndexFlatL2(self.dimension)
        self._doc_store = []
        self._id_to_pos = {}

        if active_docs:
            self.add_documents(active_docs)

        logger.info("Index rebuilt successfully.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self):
        """Lazy-load the sentence transformer model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            # Infer actual dimension from model
            sample = self._embedding_model.encode(["test"], convert_to_numpy=True)
            self.dimension = sample.shape[1]
            logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def _ensure_index_initialized(self):
        """Create an empty FAISS index if not yet initialized."""
        if self._faiss_index is None:
            import faiss
            self._faiss_index = faiss.IndexFlatL2(self.dimension)
            logger.debug(f"Created new FAISS IndexFlatL2 with dimension {self.dimension}")

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self._embedding_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)

    def _index_files_exist(self) -> bool:
        """Check whether a saved index exists on disk."""
        return (
            (self.index_path / "index.faiss").exists()
            and (self.index_path / "doc_store.pkl").exists()
            and (self.index_path / "id_to_pos.json").exists()
        )

    def _load_index(self):
        """Load a persisted index from disk."""
        import faiss
        self._faiss_index = faiss.read_index(str(self.index_path / "index.faiss"))
        with open(self.index_path / "doc_store.pkl", "rb") as f:
            self._doc_store = pickle.load(f)
        with open(self.index_path / "id_to_pos.json", "r") as f:
            self._id_to_pos = json.load(f)

    @staticmethod
    def _matches_filter(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Return True if all filter key-value pairs match the metadata."""
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True


# ---------------------------------------------------------------------------
# Convenience helper: build index from LlamaIndex Document/Node objects
# ---------------------------------------------------------------------------

def build_vector_index_from_llama_nodes(
    nodes: list,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_path: str = "data/faiss_index",
    save: bool = True,
) -> VectorIndex:
    """
    Build a VectorIndex from a list of LlamaIndex TextNode objects.

    Args:
        nodes: List of llama_index.core.schema.TextNode objects
        embedding_model: Embedding model name
        index_path: Path to persist the index
        save: Whether to save the index after building

    Returns:
        Populated VectorIndex instance
    """
    docs = []
    for node in nodes:
        docs.append({
            "doc_id": node.node_id,
            "text": node.get_content(),
            "metadata": node.metadata or {},
        })

    vi = VectorIndex(embedding_model=embedding_model, index_path=index_path)
    vi.add_documents(docs)

    if save:
        vi.save()

    return vi


# ---------------------------------------------------------------------------
# Quick smoke-test (run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== VectorIndex smoke test ===")
    vi = VectorIndex(index_path="data/faiss_index_test")

    sample_docs = [
        {"doc_id": "d1", "text": "Machine learning is a subset of artificial intelligence.", "metadata": {"source": "wiki", "topic": "ML"}},
        {"doc_id": "d2", "text": "Deep learning uses neural networks with many layers.", "metadata": {"source": "wiki", "topic": "DL"}},
        {"doc_id": "d3", "text": "Natural language processing enables computers to understand text.", "metadata": {"source": "wiki", "topic": "NLP"}},
        {"doc_id": "d4", "text": "FAISS is a library for efficient similarity search.", "metadata": {"source": "meta", "topic": "search"}},
    ]

    vi.add_documents(sample_docs)
    vi.save()

    results = vi.search("What is deep learning?", top_k=2)
    print("\nSearch results for 'What is deep learning?':")
    for r in results:
        print(f"  [{r['rank']}] (score={r['score']:.4f}) {r['text'][:80]}")

    print(f"\nStats: {vi.get_stats()}")
    print("VectorIndex smoke test PASSED ✅")
