# Advanced Multi-Source RAG for Enterprise Knowledge Base

## Project Overview

This implementation plan details the development of a sophisticated Retrieval-Augmented Generation (RAG) system designed for enterprise knowledge bases. The system ingests data from three distinct sources (PDFs, websites, and structured CSV/databases), employs multiple retrieval strategies, fuses and re-ranks results based on semantic relevance, and generates precise answers with source citations using a Large Language Model.

### Architecture Highlights

The system follows a 10-stage pipeline as shown in the architecture diagram:

1. **Streamlit UI** - User interface for data upload and queries
2. **Data Ingestion Pipeline** - Multi-format document loaders
3. **Unified Document DB** - Centralized document storage
4. **Processing Layer** - Chunking, cleaning, metadata tagging
5. **Multi-Index Layer** - Vector, sentence-window, and graph indices
6. **Multi-Retriever System** - Three parallel retrieval strategies
7. **Re-Ranking Engine** - Semantic relevance scoring
8. **Query Construction** - Context-aware query optimization
9. **LLM Integration** - Answer generation with citations
10. **Response Formatting** - Structured output with sources

### Technology Stack

- **Framework**: Python 3.9+
- **RAG Orchestration**: LlamaIndex + LangChain
- **Vector Database**: FAISS (local) or Pinecone (cloud)
- **Embeddings**: HuggingFace models (sentence-transformers)
- **Graph Database**: NetworkX (lightweight) or Neo4j (production)
- **LLM**: OpenAI GPT-4 / GPT-3.5-turbo or HuggingFace models
- **Web Framework**: Streamlit
- **Deployment**: AWS EC2/Lambda or Azure App Service
- **Storage**: Local file system / S3 / Azure Blob Storage

---

## User Review Required

> [!IMPORTANT]
> **LLM Provider Selection**: The implementation assumes OpenAI API access. If you prefer open-source models (e.g., Llama 2, Mistral) via HuggingFace, this will require additional GPU infrastructure and model hosting setup. Please confirm your preference.

> [!IMPORTANT]
> **Vector Database Choice**: FAISS is recommended for local development and smaller datasets. For production with large-scale data, Pinecone offers managed hosting but requires a subscription. Please confirm which you'd like to start with.

> [!WARNING]
> **Cloud Deployment Platform**: The plan assumes AWS deployment. If you prefer Azure, configuration files and deployment scripts will need adjustments. Please specify your target platform.

> [!IMPORTANT]
> **Graph Database Scope**: The plan uses NetworkX for a lightweight graph implementation. For production-scale entity relationship management, Neo4j is recommended but adds infrastructure complexity. Please confirm your preference.

> [!CAUTION]
> **Data Privacy & Security**: If handling sensitive enterprise data, ensure compliance with data protection regulations (GDPR, HIPAA, etc.). The implementation will include basic security measures, but production deployment may require additional hardening, encryption, and access controls.

---

## Proposed Changes

### Core Infrastructure

#### [NEW] [requirements.txt](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/requirements.txt)

Python dependencies including:
- `llama-index>=0.9.0` - Core RAG orchestration
- `langchain>=0.1.0` - Additional LLM utilities
- `faiss-cpu>=1.7.4` - Vector similarity search
- `sentence-transformers>=2.2.0` - Embeddings generation
- `openai>=1.0.0` - LLM API integration
- `streamlit>=1.28.0` - Web interface
- `beautifulsoup4>=4.12.0` - Web scraping
- `pypdf>=3.17.0` - PDF parsing
- `pandas>=2.0.0` - CSV/data handling
- `networkx>=3.1` - Graph operations
- `chromadb>=0.4.0` - Alternative vector store
- `python-dotenv>=1.0.0` - Environment management

#### [NEW] [.env.example](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/.env.example)

Environment configuration template:
```env
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_environment
VECTOR_STORE=faiss  # or pinecone
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

#### [NEW] [config.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/config.py)

Central configuration management for all system parameters including model selection, chunk sizes, retrieval settings, and deployment options.

---

### Data Ingestion Pipeline

#### [NEW] [src/ingestion/pdf_loader.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ingestion/pdf_loader.py)

PDF document loader using `pypdf` and LlamaIndex's `PDFReader`:
- Extract text with metadata (page numbers, titles)
- Handle multi-page documents
- Support batch processing
- Extract embedded images and tables (optional)

#### [NEW] [src/ingestion/web_loader.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ingestion/web_loader.py)

Web content scraper:
- Accept URLs or sitemap
- Use BeautifulSoup4 for HTML parsing
- Extract main content (remove headers/footers/ads)
- Handle pagination and multi-page articles
- Respect robots.txt and rate limiting

#### [NEW] [src/ingestion/csv_loader.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ingestion/csv_loader.py)

Structured data loader:
- Parse CSV files with pandas
- Support database connections (SQLite, PostgreSQL, MySQL)
- Convert tabular data to document format
- Preserve schema information as metadata
- Handle data type inference

#### [NEW] [src/ingestion/document_store.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ingestion/document_store.py)

Unified document storage:
- Define `Document` schema with metadata fields
- Implement document ID generation
- Store raw documents and processed versions
- Maintain source tracking
- Support CRUD operations

---

### Document Processing

#### [NEW] [src/processing/chunker.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/processing/chunker.py)

Text chunking with multiple strategies:
- **Fixed-size chunking**: Split by character/token count with overlap
- **Sentence-aware chunking**: Preserve sentence boundaries
- **Semantic chunking**: Split based on topic shifts
- **Recursive chunking**: Hierarchical splitting for large documents
- Implement LlamaIndex's `SentenceSplitter` and custom chunkers

#### [NEW] [src/processing/cleaner.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/processing/cleaner.py)

Text cleaning and normalization:
- Remove excessive whitespace, special characters
- Normalize unicode characters
- Fix encoding issues
- Remove boilerplate content
- Language detection and filtering

#### [NEW] [src/processing/metadata_tagger.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/processing/metadata_tagger.py)

Metadata extraction and enrichment:
- Extract document titles, authors, dates
- Classify document types
- Extract keywords using TF-IDF or KeyBERT
- Generate document summaries
- Add custom tags based on content

---

### Multi-Index System

#### [NEW] [src/indexing/vector_index.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/indexing/vector_index.py)

Vector database implementation:
- Initialize FAISS index (or Pinecone client)
- Generate embeddings using HuggingFace sentence-transformers
- Build dense vector index for semantic search
- Support incremental indexing
- Implement similarity search with configurable top-k

#### [NEW] [src/indexing/sentence_window_index.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/indexing/sentence_window_index.py)

Sentence-window retrieval index:
- Implement LlamaIndex's `SentenceWindowNodeParser`
- Store sentences with surrounding context windows
- Enable fine-grained retrieval with context expansion
- Configure window sizes (e.g., 3 sentences before/after)

#### [NEW] [src/indexing/graph_index.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/indexing/graph_index.py)

Knowledge graph construction:
- Extract entities using NER (spaCy or Transformers)
- Identify relationships between entities
- Build graph structure with NetworkX
- Support entity resolution and linking
- Enable graph traversal queries
- Store entity attributes and metadata

---

### Multi-Retriever System

#### [NEW] [src/retrieval/vector_retriever.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/vector_retriever.py)

Vector similarity search retriever:
- Query vector index with user question embeddings
- Return top-k most similar chunks
- Include relevance scores
- Support filtering by metadata

#### [NEW] [src/retrieval/sentence_window_retriever.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/sentence_window_retriever.py)

Sentence-window retriever:
- Retrieve precise sentences matching query
- Expand to include context windows
- Return structured results with sentence + context

#### [NEW] [src/retrieval/graph_retriever.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/graph_retriever.py)

Graph-based retriever:
- Extract entities from query
- Traverse knowledge graph to find related entities
- Retrieve documents containing matching entities/relationships
- Support multi-hop reasoning

#### [NEW] [src/retrieval/query_constructor.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/query_constructor.py)

Query optimization:
- Analyze user query intent
- Generate query variations
- Extract key phrases and entities
- Support query expansion

#### [NEW] [src/retrieval/fusion.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/fusion.py)

Context fusion mechanism:
- Merge results from multiple retrievers
- Deduplicate documents
- Combine relevance scores
- Preserve source attribution

---

### Re-Ranking & LLM Integration

#### [NEW] [src/reranking/reranker.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/reranking/reranker.py)

Semantic re-ranking:
- Implement cross-encoder re-ranking (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Score query-document pairs
- Re-order results by semantic relevance
- Support configurable re-ranking models

#### [NEW] [src/llm/llm_client.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/llm/llm_client.py)

LLM integration:
- Initialize OpenAI client (or HuggingFace)
- Implement retry logic and error handling
- Support streaming responses
- Track token usage and costs

#### [NEW] [src/llm/prompt_templates.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/llm/prompt_templates.py)

Prompt engineering:
- Create system prompts for RAG tasks
- Design templates for answer generation with citations
- Implement few-shot examples
- Support custom prompt modifications

#### [NEW] [src/llm/citation_extractor.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/llm/citation_extractor.py)

Source citation extraction:
- Parse LLM responses for citation markers
- Map citations to source documents
- Format citations with document metadata
- Validate citation accuracy

#### [NEW] [src/llm/response_formatter.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/llm/response_formatter.py)

Response formatting:
- Structure final output with answer + sources
- Generate citation links
- Format latency/accuracy metrics
- Create human-readable responses

---

### Streamlit Web Interface

#### [NEW] [src/app.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/app.py)

Main Streamlit application:
- Design multi-section layout (upload, query, results)
- Implement session state management
- Coordinate backend pipeline execution
- Display real-time processing status

#### [NEW] [src/ui/upload_interface.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ui/upload_interface.py)

File upload interface:
- Support PDF file uploads
- Accept CSV file uploads
- Provide URL input field
- Display upload progress
- Validate file types and sizes

#### [NEW] [src/ui/query_interface.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ui/query_interface.py)

Query input interface:
- Text input for questions
- Query history tracking
- Example queries for guidance
- Search button with loading state

#### [NEW] [src/ui/results_display.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ui/results_display.py)

Results presentation:
- Display answer prominently
- Show source citations with clickable links
- Present latency metrics (retrieval time, LLM time, total time)
- Display accuracy/confidence scores
- Show retrieved documents in expandable sections

---

### Testing & Quality Assurance

#### [NEW] [tests/test_ingestion.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/test_ingestion.py)

Unit tests for data ingestion:
- Test PDF loader with sample documents
- Test web loader with mock HTML
- Test CSV loader with sample data
- Validate document schema

#### [NEW] [tests/test_processing.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/test_processing.py)

Unit tests for processing:
- Test chunking strategies
- Test text cleaning functions
- Test metadata extraction
- Validate chunk boundaries

#### [NEW] [tests/test_retrieval.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/test_retrieval.py)

Unit tests for retrieval:
- Test vector search
- Test sentence-window retrieval
- Test graph retrieval
- Test fusion logic

#### [NEW] [tests/test_integration.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/test_integration.py)

End-to-end integration tests:
- Test full pipeline from ingestion to response
- Use sample knowledge base
- Validate query accuracy
- Test multiple query types

#### [NEW] [tests/sample_data/](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/sample_data/)

Test data directory:
- Sample PDF documents
- Sample CSV files
- Mock web content
- Expected query-answer pairs

---

### Deployment Configuration

#### [NEW] [Dockerfile](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/Dockerfile)

Container configuration:
- Base image: `python:3.9-slim`
- Install system dependencies
- Copy application code
- Install Python packages
- Expose Streamlit port (8501)
- Define entrypoint

#### [NEW] [docker-compose.yml](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/docker-compose.yml)

Multi-container orchestration:
- Streamlit app service
- Vector database service (if using Pinecone: external)
- Optional: Neo4j service for graph database
- Volume mounts for data persistence

#### [NEW] [deploy/aws/](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/deploy/aws/)

AWS deployment scripts:
- EC2 instance configuration
- Lambda function setup (for serverless components)
- S3 bucket creation for document storage
- IAM roles and policies

#### [NEW] [deploy/azure/](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/deploy/azure/)

Azure deployment scripts:
- App Service configuration
- Azure Blob Storage setup
- Azure Functions (optional)
- Resource group templates

---

### Documentation

#### [MODIFY] [README.md](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/README.md)

Comprehensive project documentation:
- Architecture overview with embedded diagram
- Setup instructions
- Usage examples
- Configuration guide
- API documentation
- Deployment guide
- Troubleshooting

#### [NEW] [docs/architecture.md](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/docs/architecture.md)

Detailed architecture documentation:
- System design decisions
- Component interactions
- Data flow diagrams
- Technology choices rationale

#### [NEW] [docs/api.md](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/docs/api.md)

API reference:
- Core functions and classes
- Interface contracts
- Usage examples
- Return value specifications

---

## Verification Plan

### Automated Tests

**Unit Tests**:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all unit tests with coverage
pytest tests/test_ingestion.py tests/test_processing.py tests/test_retrieval.py -v --cov=src --cov-report=html

# Expected: 80%+ code coverage, all tests passing
```

**Integration Tests**:
```bash
# Run end-to-end integration tests
pytest tests/test_integration.py -v

# Expected: Complete pipeline execution with sample data, accurate responses with citations
```

**Component Tests**:
```bash
# Test vector index creation and search
python -c "from src.indexing.vector_index import VectorIndex; idx = VectorIndex(); print('Vector index OK')"

# Test LLM integration (requires API key)
python -c "from src.llm.llm_client import LLMClient; client = LLMClient(); print('LLM client OK')"

# Test graph construction
python -c "from src.indexing.graph_index import GraphIndex; graph = GraphIndex(); print('Graph index OK')"
```

### Manual Verification

**Local Development Testing**:
1. Set up virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the Streamlit app locally:
   ```bash
   streamlit run src/app.py
   ```

4. Test data ingestion:
   - Upload a sample PDF (e.g., research paper, technical documentation)
   - Provide a website URL (e.g., documentation site)
   - Upload a CSV file (e.g., product catalog, FAQ data)
   - Verify documents appear in the system

5. Test query functionality:
   - Ask a question that requires information from PDFs: "What is the main conclusion of the research paper?"
   - Ask a question requiring web data: "What are the installation steps?"
   - Ask a question requiring structured data: "What products are available in category X?"
   - **Expected**: Accurate answers with source citations from correct document types

6. Validate multi-retriever system:
   - Check that responses include citations from multiple sources
   - Verify latency metrics are displayed (retrieval time, LLM time, total time)
   - Confirm source documents are clickable/expandable
   - **Expected**: Answers synthesize information from 2+ sources when relevant

7. Performance validation:
   - Query response time < 10 seconds for typical queries
   - System handles 10+ concurrent users without degradation
   - Re-ranking improves answer relevance (compare with/without re-ranking)

**Production Deployment Testing** (requires user action):
1. Deploy to AWS/Azure using provided scripts
2. Access deployed Streamlit app via public URL
3. Upload production-scale data (100+ documents)
4. Test multiple concurrent users
5. Monitor system logs and metrics
6. **Expected**: Stable operation, response times < 15 seconds, accurate citations

### Browser-Based UI Testing

**Streamlit Interface Verification**:
1. Use browser subagent to navigate to `http://localhost:8501`
2. Verify UI elements render correctly:
   - Upload buttons for PDF/CSV
   - URL input field
   - Query text area
   - Results display section
3. Test file upload flow
4. Submit sample query and verify results display
5. Check citation links are functional
6. Validate metrics display (latency, accuracy scores)

### Performance Benchmarking

**Retrieval Accuracy**:
```bash
# Run accuracy benchmark with test dataset
python tests/benchmark_accuracy.py

# Expected: Retrieval recall@10 > 0.85, answer accuracy > 0.80
```

**Latency Profiling**:
```bash
# Profile query latency
python tests/benchmark_latency.py

# Expected: Vector retrieval < 1s, re-ranking < 2s, LLM generation < 5s
```

### User Acceptance Criteria

- [ ] System ingests PDFs, web pages, and CSV files successfully
- [ ] Query interface is intuitive and responsive
- [ ] Answers are accurate and cite sources correctly
- [ ] Latency metrics are displayed for transparency
- [ ] System handles edge cases (empty queries, large files, network errors)
- [ ] Documentation is clear and comprehensive
- [ ] Deployment scripts work on target platform (AWS/Azure)

---

## Implementation Timeline

**Week 1-2**: Project setup, data ingestion, and processing (Phases 1-3)
**Week 3**: Multi-index system implementation (Phase 4)
**Week 4**: Multi-retriever system and fusion (Phase 5)
**Week 5**: Re-ranking and LLM integration (Phase 6)
**Week 6**: Streamlit UI development (Phase 7)
**Week 7**: Testing, optimization, and documentation (Phase 8)
**Week 8**: Deployment and final validation (Phase 9)

**Total estimated timeline**: 4-6 weeks for a working demo, 8+ weeks for production-ready system
