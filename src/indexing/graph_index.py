"""
Graph-Based Knowledge Index using NetworkX and spaCy NER.

Builds a knowledge graph from documents by:
  1. Extracting named entities (spaCy NER)
  2. Inferring co-occurrence relationships between entities within a sentence
  3. Storing the graph with NetworkX
  4. Supporting entity-centric retrieval (find docs containing related entities)

All processing is in-memory / on-disk — no external graph DB required.
"""

import json
import pickle
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Entity types we consider interesting for the knowledge graph
_DEFAULT_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "PRODUCT", "EVENT",
    "LAW", "WORK_OF_ART", "NORP", "FAC", "LANGUAGE",
    "TECH",   # custom label added by some pipelines
}


class GraphIndex:
    """
    Knowledge graph index over a document corpus.

    Graph structure:
    - Nodes: Named entities (normalised text + type)
    - Edges: Co-occurrence within the same sentence, with weight (frequency)
    - Node attributes: entity type, label, mention_count, doc_ids
    - Edge attributes: weight (co-occurrence count), doc_ids, sentences

    Retrieval:
    - Given a query, extract its entities → expand via graph neighbours →
      return documents that contain the expanded entity set.
    """

    def __init__(
        self,
        ner_model: str = "en_core_web_sm",
        entity_types: Optional[Set[str]] = None,
        index_path: str = "data/graph_index",
        max_nodes: int = 10_000,
        max_edges: int = 50_000,
    ):
        """
        Args:
            ner_model: spaCy model for NER (e.g. 'en_core_web_sm').
            entity_types: Set of spaCy entity labels to include.
            index_path: Directory to persist the graph.
            max_nodes: Safety cap on graph nodes.
            max_edges: Safety cap on graph edges.
        """
        self.ner_model_name = ner_model
        self.entity_types = entity_types or _DEFAULT_ENTITY_TYPES
        self.index_path = Path(index_path)
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self._nlp = None               # lazy-loaded spaCy pipeline
        self._graph = None             # lazy-created NetworkX graph

        # Mapping: normalised entity label -> list of doc_ids containing it
        self._entity_to_docs: Dict[str, List[str]] = defaultdict(list)

        # Mapping: doc_id -> list of entity labels found in that document
        self._doc_to_entities: Dict[str, List[str]] = defaultdict(list)

        if self._files_exist():
            self._load()
            logger.info(
                f"Loaded GraphIndex: {self._graph.number_of_nodes()} nodes, "
                f"{self._graph.number_of_edges()} edges"
            )
        else:
            import networkx as nx
            self._graph = nx.Graph()
            logger.info("Initialised empty GraphIndex.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Extract entities and relationships from documents and update the graph.

        Args:
            documents: List of dicts with keys:
                - 'doc_id' (str)
                - 'text'   (str)
                - 'metadata' (dict, optional)

        Returns:
            Dict with 'entities_added', 'relations_added'
        """
        self._ensure_nlp_loaded()
        entities_added = 0
        relations_added = 0

        for doc in documents:
            doc_id = doc.get("doc_id", "")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            if not text.strip():
                continue

            # Extract entities and co-occurrence relations per sentence
            sentences = self._split_sentences(text)
            for sentence in sentences:
                entities_in_sentence = self._extract_entities(sentence)
                if not entities_in_sentence:
                    continue

                # Add / update entity nodes
                for entity_label, entity_type in entities_in_sentence:
                    e_added = self._add_entity_node(
                        entity_label, entity_type, doc_id, sentence
                    )
                    if e_added:
                        entities_added += 1

                    # Track entity <-> document mapping
                    if doc_id not in self._entity_to_docs[entity_label]:
                        self._entity_to_docs[entity_label].append(doc_id)
                    if entity_label not in self._doc_to_entities[doc_id]:
                        self._doc_to_entities[doc_id].append(entity_label)

                # Add co-occurrence edges between entities within the sentence
                entity_labels = [e[0] for e in entities_in_sentence]
                for i in range(len(entity_labels)):
                    for j in range(i + 1, len(entity_labels)):
                        e_added = self._add_relation_edge(
                            entity_labels[i],
                            entity_labels[j],
                            doc_id,
                            sentence,
                        )
                        if e_added:
                            relations_added += 1

            # Safety cap
            if self._graph.number_of_nodes() >= self.max_nodes:
                logger.warning(
                    f"Max nodes ({self.max_nodes}) reached; stopping entity extraction."
                )
                break
            if self._graph.number_of_edges() >= self.max_edges:
                logger.warning(
                    f"Max edges ({self.max_edges}) reached; stopping relation extraction."
                )
                break

        logger.info(
            f"GraphIndex.add_documents complete: "
            f"{entities_added} new entity nodes, {relations_added} new relation edges"
        )
        return {"entities_added": entities_added, "relations_added": relations_added}

    def search_by_query(
        self,
        query: str,
        top_k: int = 5,
        hop_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Entity-centric retrieval.

        Steps:
        1. Extract entities from query text.
        2. Expand entity set via graph neighbours (up to hop_depth hops).
        3. Collect all doc_ids associated with the expanded entities.
        4. Return top_k doc_ids ranked by number of matching entities.

        Args:
            query: Natural language query.
            top_k: Maximum number of document IDs to return.
            hop_depth: Number of graph hops to expand from seed entities.

        Returns:
            List of result dicts:
                - 'doc_id', 'matched_entities', 'entity_count', 'rank'
        """
        self._ensure_nlp_loaded()
        seed_entities = {e[0] for e in self._extract_entities(query)}

        if not seed_entities:
            logger.debug("No entities found in query, returning empty results.")
            return []

        # Expand via graph traversal
        expanded_entities = self._expand_entities(seed_entities, hop_depth)

        # Collect matching documents
        doc_entity_counts: Dict[str, Set[str]] = defaultdict(set)
        for entity in expanded_entities:
            for doc_id in self._entity_to_docs.get(entity, []):
                doc_entity_counts[doc_id].add(entity)

        # Sort by number of matched entities (descending)
        ranked = sorted(
            doc_entity_counts.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:top_k]

        return [
            {
                "doc_id": doc_id,
                "matched_entities": list(entities),
                "entity_count": len(entities),
                "rank": rank,
            }
            for rank, (doc_id, entities) in enumerate(ranked, start=1)
        ]

    def search_entities(
        self,
        entity_name: str,
        hop_depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Find graph neighbours of a specific entity.

        Args:
            entity_name: Normalised entity label (lowercase).
            hop_depth: Number of hops to traverse.

        Returns:
            List of neighbour dicts: {entity, type, weight, doc_ids}
        """
        normalised = entity_name.lower().strip()
        if normalised not in self._graph:
            return []

        neighbours = self._expand_entities({normalised}, hop_depth) - {normalised}
        results = []
        for neighbour in neighbours:
            node_data = self._graph.nodes.get(neighbour, {})
            edge_data = self._graph.edges.get((normalised, neighbour), {})
            results.append({
                "entity": neighbour,
                "type": node_data.get("type", "UNKNOWN"),
                "weight": edge_data.get("weight", 1),
                "doc_ids": self._entity_to_docs.get(neighbour, []),
            })

        results.sort(key=lambda x: x["weight"], reverse=True)
        return results

    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get node attributes for a specific entity."""
        key = entity_name.lower().strip()
        if key not in self._graph:
            return None
        data = dict(self._graph.nodes[key])
        data["doc_ids"] = self._entity_to_docs.get(key, [])
        return data

    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "documents_indexed": len(self._doc_to_entities),
            "ner_model": self.ner_model_name,
            "entity_types": list(self.entity_types),
            "index_path": str(self.index_path),
        }

    def save(self):
        """Persist the graph and mappings to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)

        with open(self.index_path / "graph.pkl", "wb") as f:
            pickle.dump(self._graph, f)
        with open(self.index_path / "entity_to_docs.json", "w") as f:
            json.dump(dict(self._entity_to_docs), f)
        with open(self.index_path / "doc_to_entities.json", "w") as f:
            json.dump(dict(self._doc_to_entities), f)

        logger.info(
            f"Saved GraphIndex ({self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges) to {self.index_path}"
        )

    def export_graph_json(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export the graph as a JSON-serialisable dict (nodes + edges).

        Useful for debugging or visualisation tools like Gephi / D3.js.
        """
        import networkx as nx
        data = nx.node_link_data(self._graph)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        return data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_nlp_loaded(self):
        """Lazy-load the spaCy NLP pipeline."""
        if self._nlp is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load(self.ner_model_name, disable=["parser"])
            logger.info(f"Loaded spaCy model: {self.ner_model_name}")
        except OSError:
            logger.warning(
                f"spaCy model '{self.ner_model_name}' not found. "
                "Falling back to regex-based entity detection."
            )
            self._nlp = None  # will use regex fallback

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text.

        Returns list of (normalised_label, entity_type) tuples.
        """
        if self._nlp is not None:
            try:
                doc = self._nlp(text[:50_000])
                entities = []
                seen = set()
                for ent in doc.ents:
                    if ent.label_ in self.entity_types:
                        key = ent.text.lower().strip()
                        if key and key not in seen and len(key) > 1:
                            entities.append((key, ent.label_))
                            seen.add(key)
                return entities
            except Exception as e:
                logger.debug(f"spaCy NER error: {e}, using regex fallback.")

        # --- Regex fallback: detect capitalised phrases as ENTITY candidates ---
        return self._regex_entity_fallback(text)

    def _regex_entity_fallback(self, text: str) -> List[Tuple[str, str]]:
        """
        Regex entity extractor: captures both multi-word and single-word proper nouns.

        - Multi-word proper nouns (e.g. "Steve Jobs", "New York") are highest confidence.
        - Single-word proper nouns mid-sentence (e.g. "Apple", "Google") are also captured.
        """
        found: List[Tuple[str, str]] = []
        seen: Set[str] = set()

        # Multi-word capitalised phrases
        multi_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        for m in re.findall(multi_pattern, text):
            key = m.lower().strip()
            if key not in seen and len(key) > 3:
                found.append((key, "ENTITY"))
                seen.add(key)

        # Single-word capitalised words that appear mid-sentence (after a space or comma)
        # Uses a lookbehind to exclude sentence-starting words where possible
        single_pattern = r'(?<=[,;:\s])([A-Z][a-z]{2,})'
        for m in re.finditer(single_pattern, text):
            word = m.group(1)
            key = word.lower().strip()
            if key not in seen and len(key) >= 3:
                found.append((key, "ENTITY"))
                seen.add(key)

        return found[:50]  # cap per call

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self._nlp is not None:
            try:
                # Use sentencizer for efficiency
                sentencizer_nlp = self._nlp
                doc = sentencizer_nlp(text[:100_000])
                sents = [s.text.strip() for s in doc.sents if s.text.strip()]
                if sents:
                    return sents
            except Exception:
                pass

        # Regex fallback
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _add_entity_node(
        self,
        label: str,
        entity_type: str,
        doc_id: str,
        sentence: str,
    ) -> bool:
        """Add or update an entity node. Returns True if newly added."""
        if label in self._graph:
            node = self._graph.nodes[label]
            node["mention_count"] = node.get("mention_count", 1) + 1
            if doc_id not in node.get("doc_ids", []):
                node.setdefault("doc_ids", []).append(doc_id)
            return False
        else:
            if self._graph.number_of_nodes() >= self.max_nodes:
                return False
            self._graph.add_node(
                label,
                label=label,
                type=entity_type,
                mention_count=1,
                doc_ids=[doc_id],
            )
            return True

    def _add_relation_edge(
        self,
        entity_a: str,
        entity_b: str,
        doc_id: str,
        sentence: str,
    ) -> bool:
        """Add or increment a co-occurrence edge. Returns True if newly added."""
        if entity_a == entity_b:
            return False

        if self._graph.has_edge(entity_a, entity_b):
            self._graph[entity_a][entity_b]["weight"] += 1
            if doc_id not in self._graph[entity_a][entity_b].get("doc_ids", []):
                self._graph[entity_a][entity_b].setdefault("doc_ids", []).append(doc_id)
            return False
        else:
            if self._graph.number_of_edges() >= self.max_edges:
                return False
            # Ensure both nodes exist before adding edge
            if entity_a not in self._graph or entity_b not in self._graph:
                return False
            self._graph.add_edge(
                entity_a,
                entity_b,
                weight=1,
                relation="co-occurrence",
                doc_ids=[doc_id],
            )
            return True

    def _expand_entities(
        self,
        seed_entities: Set[str],
        hop_depth: int,
    ) -> Set[str]:
        """
        BFS expansion of entities in the graph up to hop_depth hops.
        """
        expanded: Set[str] = set()
        frontier = {e for e in seed_entities if e in self._graph}
        expanded.update(frontier)

        for _ in range(hop_depth):
            next_frontier: Set[str] = set()
            for entity in frontier:
                for neighbour in self._graph.neighbors(entity):
                    if neighbour not in expanded:
                        next_frontier.add(neighbour)
                        expanded.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break

        return expanded

    def _files_exist(self) -> bool:
        return (self.index_path / "graph.pkl").exists()

    def _load(self):
        with open(self.index_path / "graph.pkl", "rb") as f:
            self._graph = pickle.load(f)
        with open(self.index_path / "entity_to_docs.json", "r") as f:
            raw = json.load(f)
            self._entity_to_docs = defaultdict(list, raw)
        with open(self.index_path / "doc_to_entities.json", "r") as f:
            raw = json.load(f)
            self._doc_to_entities = defaultdict(list, raw)


# ---------------------------------------------------------------------------
# Convenience helper: build from LlamaIndex nodes
# ---------------------------------------------------------------------------

def build_graph_index_from_llama_nodes(
    nodes: list,
    ner_model: str = "en_core_web_sm",
    index_path: str = "data/graph_index",
    save: bool = True,
) -> "GraphIndex":
    """Build a GraphIndex from a list of LlamaIndex TextNode objects."""
    docs = [
        {
            "doc_id": node.node_id,
            "text": node.get_content(),
            "metadata": node.metadata or {},
        }
        for node in nodes
    ]
    gi = GraphIndex(ner_model=ner_model, index_path=index_path)
    gi.add_documents(docs)
    if save:
        gi.save()
    return gi


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== GraphIndex smoke test ===")
    gi = GraphIndex(index_path="data/graph_index_test")

    sample_docs = [
        {
            "doc_id": "d1",
            "text": (
                "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino. "
                "Apple launched the iPhone in 2007. "
                "Steve Jobs returned to Apple in 1997 after NeXT was acquired."
            ),
            "metadata": {"source": "wiki"},
        },
        {
            "doc_id": "d2",
            "text": (
                "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque. "
                "Microsoft released Windows in 1985. "
                "Bill Gates left Microsoft to focus on the Bill & Melinda Gates Foundation."
            ),
            "metadata": {"source": "wiki"},
        },
        {
            "doc_id": "d3",
            "text": (
                "Google was founded by Larry Page and Sergey Brin at Stanford University. "
                "Google Search became the world's most popular search engine. "
                "Alphabet was created as the parent company of Google in 2015."
            ),
            "metadata": {"source": "wiki"},
        },
    ]

    stats = gi.add_documents(sample_docs)
    print(f"\nExtraction stats: {stats}")

    results = gi.search_by_query("Who founded Apple?", top_k=3)
    print("\nGraph search results for 'Who founded Apple?':")
    for r in results:
        print(f"  [{r['rank']}] doc_id={r['doc_id']}, matched_entities={r['matched_entities']}")

    gi.save()
    print(f"\nFinal graph stats: {gi.get_stats()}")
    print("GraphIndex smoke test PASSED ✅")
