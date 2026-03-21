"""
Sentence-Window Index for fine-grained retrieval with contextual expansion.

Strategy:
  - Each sentence is indexed individually (fine-grained retrieval unit).
  - Each sentence entry stores a surrounding context window of neighboring
    sentences so that when a sentence is retrieved its broader context can
    be returned to the LLM.

This avoids the precision/recall trade-off of pure chunk-based retrieval:
retrieval happens at sentence precision, but generation receives rich context.
"""

import json
import pickle
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class SentenceWindowIndex:
    """
    Sentence-window retrieval index.

    Each document is split into sentences; each sentence is stored with a
    configurable window of surrounding sentences.  Similarity search is
    performed at the sentence level via a lightweight vector index, but
    results are returned with the full context window.

    The index uses the VectorIndex internally for the similarity search step
    so that all embedding/FAISS logic lives in one place.
    """

    def __init__(
        self,
        window_size: int = 3,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "data/sentence_window_index",
        batch_size: int = 32,
    ):
        """
        Args:
            window_size: Number of sentences before AND after the target sentence
                         to include in the context window.
            embedding_model: HuggingFace model used for sentence embeddings.
            index_path: Directory to persist the index.
            batch_size: Embedding batch size.
        """
        self.window_size = window_size
        self.index_path = Path(index_path)
        self.batch_size = batch_size
        self.embedding_model_name = embedding_model

        # Sentence store: parallel to VectorIndex rows
        # Each entry: {sentence_id, sentence_text, context_window, doc_id, metadata}
        self._sentence_store: List[Dict[str, Any]] = []
        self._id_to_pos: Dict[str, int] = {}

        # Lazy-loaded vector index for embedding search
        self._vector_index = None

        # Try to load from disk
        if self._files_exist():
            self._load()
            logger.info(
                f"Loaded SentenceWindowIndex with {len(self._sentence_store)} sentences"
            )
        else:
            logger.info("No existing SentenceWindowIndex found; will create on first add.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Split documents into sentences and add them to the index.

        Args:
            documents: List of dicts with keys:
                - 'doc_id' (str)
                - 'text'   (str)  - full document or chunk text
                - 'metadata' (dict, optional)

        Returns:
            Number of sentences indexed.
        """
        self._ensure_vector_index()
        indexed = 0
        new_entries: List[Dict[str, Any]] = []

        for doc in documents:
            doc_id = doc.get("doc_id", f"doc_{len(self._sentence_store)}")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            sentences = self._split_into_sentences(text)

            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Build context window
                start = max(0, sent_idx - self.window_size)
                end = min(len(sentences), sent_idx + self.window_size + 1)
                context_window = " ".join(sentences[start:end]).strip()

                sentence_id = f"{doc_id}__sent_{sent_idx}"
                if sentence_id in self._id_to_pos:
                    continue  # already indexed

                entry = {
                    "sentence_id": sentence_id,
                    "sentence_text": sentence,
                    "context_window": context_window,
                    "doc_id": doc_id,
                    "sentence_idx": sent_idx,
                    "window_start": start,
                    "window_end": end - 1,
                    "metadata": metadata,
                }

                pos = len(self._sentence_store) + len(new_entries)
                self._id_to_pos[sentence_id] = pos
                new_entries.append(entry)
                indexed += 1

        # Batch-add to underlying vector index (embed sentence_text)
        if new_entries:
            self._sentence_store.extend(new_entries)
            vec_docs = [
                {
                    "doc_id": e["sentence_id"],
                    "text": e["sentence_text"],
                    "metadata": e["metadata"],
                }
                for e in new_entries
            ]
            self._vector_index.add_documents(vec_docs)

        logger.info(f"SentenceWindowIndex: added {indexed} sentences.")
        return indexed

    def search(
        self,
        query: str,
        top_k: int = 5,
        return_context: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find sentences most similar to the query and return with context.

        Args:
            query: Natural language query.
            top_k: Number of sentence-level results to return.
            return_context: If True, returned text is the full context window;
                            if False, only the matched sentence.
            metadata_filter: Optional key-value metadata filter.

        Returns:
            List of result dicts:
                - sentence_id, sentence_text, context_window,
                  doc_id, metadata, score, rank
        """
        if not self._sentence_store:
            logger.warning("search() called on empty SentenceWindowIndex.")
            return []

        self._ensure_vector_index()
        raw_results = self._vector_index.search(
            query=query, top_k=top_k, metadata_filter=metadata_filter
        )

        results = []
        for rank, raw in enumerate(raw_results, start=1):
            sentence_id = raw["doc_id"]
            pos = self._id_to_pos.get(sentence_id)
            if pos is None or pos >= len(self._sentence_store):
                continue

            entry = self._sentence_store[pos]
            results.append({
                "sentence_id": sentence_id,
                "sentence_text": entry["sentence_text"],
                "context_window": entry["context_window"],
                "doc_id": entry["doc_id"],
                "sentence_idx": entry["sentence_idx"],
                "metadata": entry["metadata"],
                "score": raw["score"],
                "rank": rank,
                # Convenience: 'text' field contains the appropriate content
                "text": entry["context_window"] if return_context else entry["sentence_text"],
            })

        return results

    def save(self):
        """Persist the index to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        with open(self.index_path / "sentence_store.pkl", "wb") as f:
            pickle.dump(self._sentence_store, f)
        with open(self.index_path / "id_to_pos.json", "w") as f:
            json.dump(self._id_to_pos, f)

        # Also persist the underlying vector index
        if self._vector_index:
            self._vector_index.save()

        logger.info(
            f"Saved SentenceWindowIndex ({len(self._sentence_store)} sentences)"
            f" to {self.index_path}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "total_sentences": len(self._sentence_store),
            "window_size": self.window_size,
            "embedding_model": self.embedding_model_name,
            "index_path": str(self.index_path),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple regex heuristics.

        Falls back to spaCy sentencizer if available for better accuracy.
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            nlp.add_pipe("sentencizer")
            doc = nlp(text[:100_000])  # guard against huge inputs
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception:
            pass  # spaCy not available or model not downloaded – use regex

        # Regex sentence splitter: split on '. ', '! ', '? ' and '...'
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _ensure_vector_index(self):
        """Lazy-initialize the VectorIndex backend."""
        if self._vector_index is None:
            from src.indexing.vector_index import VectorIndex
            vi_path = str(self.index_path / "vector_backend")
            self._vector_index = VectorIndex(
                embedding_model=self.embedding_model_name,
                index_path=vi_path,
                batch_size=self.batch_size,
            )

    def _files_exist(self) -> bool:
        return (
            (self.index_path / "sentence_store.pkl").exists()
            and (self.index_path / "id_to_pos.json").exists()
        )

    def _load(self):
        with open(self.index_path / "sentence_store.pkl", "rb") as f:
            self._sentence_store = pickle.load(f)
        with open(self.index_path / "id_to_pos.json", "r") as f:
            self._id_to_pos = json.load(f)
        self._ensure_vector_index()


# ---------------------------------------------------------------------------
# Convenience helper: build index from LlamaIndex TextNode objects
# ---------------------------------------------------------------------------

def build_sentence_window_index_from_llama_nodes(
    nodes: list,
    window_size: int = 3,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_path: str = "data/sentence_window_index",
    save: bool = True,
) -> SentenceWindowIndex:
    """
    Build a SentenceWindowIndex from a list of LlamaIndex TextNode objects.
    """
    docs = [
        {
            "doc_id": node.node_id,
            "text": node.get_content(),
            "metadata": node.metadata or {},
        }
        for node in nodes
    ]

    swi = SentenceWindowIndex(
        window_size=window_size,
        embedding_model=embedding_model,
        index_path=index_path,
    )
    swi.add_documents(docs)

    if save:
        swi.save()

    return swi


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== SentenceWindowIndex smoke test ===")
    swi = SentenceWindowIndex(window_size=2, index_path="data/swi_test")

    sample_docs = [
        {
            "doc_id": "doc_ml",
            "text": (
                "Artificial intelligence is the simulation of human intelligence. "
                "Machine learning is a subset of AI that uses statistical methods. "
                "Deep learning uses multi-layer neural networks. "
                "These networks learn representations automatically from raw data. "
                "Transfer learning allows models to apply knowledge across domains."
            ),
            "metadata": {"source": "textbook", "chapter": 1},
        },
        {
            "doc_id": "doc_nlp",
            "text": (
                "Natural language processing enables machines to understand text. "
                "Tokenisation splits text into words or subwords. "
                "Named-entity recognition identifies proper nouns in documents. "
                "Transformers have revolutionised NLP benchmarks since 2017."
            ),
            "metadata": {"source": "textbook", "chapter": 2},
        },
    ]

    swi.add_documents(sample_docs)
    swi.save()

    results = swi.search("What is deep learning?", top_k=2)
    print("\nResults for 'What is deep learning?':")
    for r in results:
        print(f"  [{r['rank']}] sentence: {r['sentence_text'][:80]}")
        print(f"       context: {r['context_window'][:120]}")

    print(f"\nStats: {swi.get_stats()}")
    print("SentenceWindowIndex smoke test PASSED ✅")
