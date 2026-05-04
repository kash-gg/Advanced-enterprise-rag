# Advanced Multi-Source RAG for Enterprise Knowledge Base

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

A sophisticated Retrieval-Augmented Generation (RAG) system designed for enterprise knowledge bases. This system ingests data from multiple sources (PDFs, websites, structured data), employs advanced retrieval strategies, and generates precise answers with source citations using state-of-the-art LLMs.

## 🌟 Key Features

- **Multi-Source Data Ingestion**: PDFs, web pages, CSV/database
- **Advanced Retrieval**: Vector search, sentence-window retrieval, knowledge graph traversal
- **Intelligent Re-ranking**: Semantic re-ranking for optimal relevance
- **LLM Integration**: HuggingFace open-source models (Mistral, Zephyr, Llama 2)
- **Source Attribution**: Accurate citations with document metadata
- **Interactive UI**: Streamlit-based web interface

## 💰 100% Free & Open-Source

All components are completely free with no subscription costs:
- **Vector DB**: FAISS (CPU-based, no GPU required)
- **Graph DB**: NetworkX (in-memory)
- **LLM**: HuggingFace models (local inference)
- **Embeddings**: Sentence-Transformers
- **Total Cost**: 0

## 🏗️ Architecture

```
User Query → Multi-Retriever System → Re-Ranking → LLM → Answer + Citations
              ↓                        ↓
         [Vector Index]          [Graph Index]
         [Sentence Window]       [Metadata Filter]
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9, 3.10, 3.11, or 3.12** (Python 3.13+ not yet supported due to spaCy compatibility)
- 8GB+ RAM (16GB recommended for larger models)
- Optional: GPU for faster LLM inference

> [!IMPORTANT]
> **Python Version Requirement**: This project requires Python 3.9-3.12. Python 3.13 and 3.14 are not yet supported because spaCy (a core dependency) relies on Pydantic v1, which doesn't support Python 3.13+. If you have Python 3.13/3.14, please install Python 3.12 alongside it.

### Installation

**Windows (PowerShell):**
```powershell
# Clone the repository
git clone <repository-url>
cd Advanced-enterprise-rag

# Run setup script
.\scripts\setup.ps1
```

**Linux/Mac:**
```bash
# Clone the repository
git clone <repository-url>
cd Advanced-enterprise-rag

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create .env file
cp .env.example .env
```

### Configuration

Edit `.env` file to customize:
- **LLM Model**: Choose from Mistral, Zephyr, Llama 2, or TinyLlama
- **Device**: `cpu` or `cuda` for GPU acceleration
- **Chunk Size**: Adjust for your document types
- **Retrieval Settings**: Top-K, similarity threshold, re-ranking

### Running the Application

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate    # Windows

# Run Streamlit app
streamlit run src/app.py
```

Access at: `http://localhost:8501`

## 📦 Technology Stack

| Component | Technology | License |
|-----------|-----------|---------|
| RAG Framework | LlamaIndex, LangChain | MIT |
| Vector DB | FAISS | MIT |
| Embeddings | Sentence-Transformers | Apache 2.0 |
| LLM | HuggingFace Transformers | Apache 2.0 |
| Graph DB | NetworkX | BSD |
| NER | spaCy | MIT |
| Web UI | Streamlit | Apache 2.0 |
| Document Processing | PyPDF, BeautifulSoup4, Pandas | BSD/MIT |

## 📚 Project Structure

```
Advanced-enterprise-rag/
├── src/
│   ├── ingestion/          # Data loaders (PDF, web, CSV)
│   ├── processing/         # Text chunking, cleaning, metadata
│   ├── indexing/           # Vector, graph, sentence-window indices
│   ├── retrieval/          # Multi-retriever system
│   ├── reranking/          # Semantic re-ranking
│   ├── llm/                # LLM integration & prompts
│   ├── ui/                 # Streamlit UI components
│   └── config.py           # Central configuration
├── data/                   # Document storage & indices
├── models/                 # Cached HuggingFace models
├── logs/                   # Application logs
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/                # Setup & utility scripts
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── README.md
```

## 🛠️ Development Status

## 🛠️ Work Completed & Data Flow

So far, we have successfully implemented the first 4 phases of our RAG architecture. Here is the step-by-step flow of how data moves through the completed components:

### 1. Data Ingestion Pipeline (`src/ingestion/`) ✅
Data enters the system through our ingestion module, which normalizes different formats into a unified `Document` schema.
- **PDF Loader**: Extracts text and metadata (page numbers, titles) from PDF documents using `pypdf`.
- **Web Loader**: Scrapes and parses web URLs, stripping out boilerplate HTML (headers, footers) to extract the main content using `BeautifulSoup4`.
- **CSV Loader**: Ingests structured tabular data using `pandas`, allowing row-by-row conversion into searchable text documents.
- **Document Store**: A centralized database (SQLite-based) that tracks document origins, metadata, and IDs.

### 2. Document Processing (`src/processing/`) ✅
Once ingested, raw documents are processed to optimize them for retrieval.
- **Text Cleaner**: Normalizes text, removes excessive whitespace, fixes encoding issues, and standardizes formats.
- **Chunker**: Splits large documents into smaller, semantically meaningful pieces. Supports fixed-size chunking with overlaps, sentence-aware chunking, and semantic chunking.
- **Metadata Tagger**: Enriches chunks by extracting keywords, generating summaries, and identifying document types.
- **Quality Checker**: Validates chunks to ensure they meet minimum quality thresholds (e.g., adequate length, low proportion of special characters) before indexing.

### 3. Multi-Index System (`src/indexing/`) ✅
Processed chunks are stored in three parallel, specialized indices to support diverse query types.
- **Vector Index (FAISS)**: Generates dense embeddings (using `sentence-transformers`) for chunks and indexes them for fast, scalable semantic similarity search. Runs efficiently on CPU.
- **Sentence-Window Index**: Stores individual sentences alongside their surrounding context (e.g., 3 sentences before/after). This allows the retriever to match highly specific sentences while passing the broader context to the LLM.
- **Graph Index (NetworkX)**: A knowledge graph built by extracting entities and relationships using NLP (`spaCy`). It supports complex, multi-hop reasoning and entity connection queries.

### Phase 5-9: Pending
- Phase 5: Multi-Retriever System & Fusion
- Phase 6: Re-Ranking & LLM Integration
- Phase 7: Streamlit Web Interface
- Phase 8: Testing & Optimization
- Phase 9: Deployment

See [task.md](task.md) for detailed roadmap.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test configuration
python -c "from src.config import config; print(config)"
```

## 📖 Documentation

- [Architecture Overview](docs/architecture.md) (Coming soon)
- [API Reference](docs/api.md) (Coming soon)
- [Configuration Guide](.env.example)
- [Implementation Plan](implementation_plan.md)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LlamaIndex & LangChain communities
- HuggingFace for open-source models
- Meta AI for FAISS

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is under active development. Features and documentation are continuously evolving.
