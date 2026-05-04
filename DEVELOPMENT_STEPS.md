# Development Steps & Journey

This document details the exact, step-by-step process followed to build the Advanced Multi-Source RAG application up to its current state.

## Phase 1: Project Setup & Infrastructure
1. **Repository Initialization**: Initialized the core project directory structure to follow modular Python architecture (`src/ingestion`, `src/processing`, `src/indexing`, etc.).
2. **Environment Configuration**: Created `.env.example` to define environment variables for Vector Store parameters, LLM model selection, document chunk sizes, and graph configurations.
3. **Dependency Management**: Drafted `requirements.txt` incorporating strictly free and open-source libraries: `llama-index`, `langchain`, `faiss-cpu`, `sentence-transformers`, `networkx`, `spacy`, `pypdf`, `pandas`, and `streamlit`.

## Phase 2: Data Ingestion Pipeline (`src/ingestion/`)
The goal of this phase was to build robust loaders capable of unifying disparate data formats into a single `Document` schema.
1. **Unified Schema Creation**: Developed `document_store.py` to establish a base `Document` class holding text and metadata, and to track document origins and IDs.
2. **PDF Loader**: Implemented `pdf_loader.py` utilizing `pypdf` to parse text and extract critical metadata like page numbers from uploaded PDF files.
3. **Web Loader**: Implemented `web_loader.py` leveraging `BeautifulSoup4` to scrape HTML content, aggressively strip out headers, footers, and scripts, and extract the main article text.
4. **CSV/Database Loader**: Implemented `csv_loader.py` using `pandas` to read tabular data, mapping each row into an independent text representation along with its schema metadata.

## Phase 3: Document Processing (`src/processing/`)
To optimize text for LLM retrieval, raw ingested documents required rigorous cleaning, chunking, and enrichment.
1. **Text Cleaning**: Implemented `cleaner.py` to strip excessive whitespace, normalize unicode characters, and handle encoding artifacts to prevent garbage data from entering the indices.
2. **Text Chunking**: Developed `chunker.py` to systematically slice large documents into smaller semantic units. Supported strategies include fixed-size chunking with overlaps, sentence-aware chunking, and semantic topic-based chunking.
3. **Metadata Tagging**: Implemented `metadata_tagger.py` to enrich chunks by extracting document summaries, keywords, and document-type classifications prior to indexing.
4. **Quality Checks**: Created `quality_checker.py` to proactively discard chunks with excessive special characters or insufficient word counts, ensuring only high-quality context is passed to the LLM.

## Phase 4: Multi-Index System (`src/indexing/`)
The core reasoning capability of the system requires specialized indices to handle broad semantic searches, fine-grained details, and interconnected entities concurrently.
1. **Vector Index (FAISS)**: Developed `vector_index.py` integrating `faiss-cpu` and `sentence-transformers` to generate dense vector embeddings for chunked text, supporting high-speed semantic similarity search on local CPU hardware.
2. **Sentence-Window Index**: Built `sentence_window_index.py` using LlamaIndex constructs to store highly granular single sentences while maintaining links to their surrounding context window (e.g., 3 sentences before and after). This provides highly specific retrieval with robust context.
3. **Knowledge Graph Index**: Created `graph_index.py` using `spaCy` for Named Entity Recognition (NER) and `NetworkX` for graph construction. This module extracts entities, maps their relationships, and plots them in a graph, paving the way for advanced multi-hop reasoning.

## Future Development (Pending)
- **Phase 5**: Building the Multi-Retriever and Fusion systems to query all three indices simultaneously and merge results.
- **Phase 6**: Integrating Cross-Encoder Semantic Re-Ranking and the local HuggingFace LLM generation component.
- **Phase 7**: Constructing the Streamlit Web UI.
- **Phase 8 & 9**: End-to-end testing, optimization, and local deployment via Docker.
