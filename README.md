# RAGfolio - Personal RAG System

A private Retrieval-Augmented Generation (RAG) system for semantic search and question-answering over your personal corpus of writing, transcripts, and notes.

## Features

-  **Multi-format Support**: Parse Markdown, TXT, DOCX, and PDF files
-  **Semantic Search**: Local embeddings using sentence-transformers
-  **Recency Weighting**: Prioritize newer material in search results
-  **Version Management**: Track document versions and changes
-  **LLM Integration**: Generate answers using OpenRouter models
-  **Metadata Filtering**: Search by topic, date, version, or priority
-  **Chat Interface**: Natural language querying via Streamlit

## Quick Start

1. **Clone and Setup**
   ```bash
   cd RAGfolio
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API Keys**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenRouter API key
   ```

3. **Add Your Documents**
   ```bash
   # Place documents in the documents/ folder
   # Supported formats: .md, .txt, .docx, .pdf
   ```

4. **Initialize and Index**
   ```bash
   python src/indexer.py --initialize
   python src/indexer.py --index-all
   ```

5. **Start the Interface**
   ```bash
   streamlit run src/app.py
   ```

## Project Structure

```
RAGfolio/
├── src/
│   ├── app.py              # Streamlit interface
│   ├── indexer.py          # Document processing and indexing
│   ├── retriever.py        # Semantic search and retrieval
│   ├── llm_integration.py  # OpenRouter LLM integration
│   ├── document_parser.py  # Multi-format document parsing
│   ├── version_manager.py  # Document version tracking
│   └── config.py           # Configuration settings
├── documents/              # Your source documents
├── data/
│   ├── embeddings/         # FAISS vector store
│   ├── metadata.db         # SQLite metadata database
│   └── versions/           # Document version history
├── requirements.txt
├── .env.example
└── README.md
```

## Usage Examples

### Adding Documents
```bash
# Add single document
python src/indexer.py --add documents/my_notes.md

# Bulk index all documents
python src/indexer.py --index-all

# Re-index with priority weighting
python src/indexer.py --reindex --priority-boost 1.5
```

### Searching
```bash
# Command line search
python src/retriever.py "meditation and mindfulness practices"

# Web interface
streamlit run src/app.py
```

## Configuration

Edit `src/config.py` or set environment variables:

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `CHUNK_SIZE`: Text chunk size for processing (default: 512)
- `OVERLAP_SIZE`: Chunk overlap size (default: 50)
- `MAX_RESULTS`: Maximum search results (default: 10)

## Document Metadata

The system automatically extracts and uses:
- **Date**: File modification time or embedded dates
- **Version**: Git commit hash or timestamp
- **Topic**: Extracted from content and file path
- **Priority**: Manual tags or recency-based scoring

## Version Management

- Automatic version tracking via Git integration
- Maintains links between chunks and source versions
- Supports rollback and historical search
- Version comparison and diff viewing

## Advanced Features

### Custom Metadata Tags
Add YAML frontmatter to your documents:
```yaml
---
title: "Meditation Notes"
priority: high
topics: ["mindfulness", "practice"]
date: 2024-11-08
---
```

### Search Filters
- Date range: `after:2024-01-01 before:2024-12-31`
- Topic: `topic:meditation`
- Priority: `priority:high`
- Version: `version:latest`

## API Integration

The system uses OpenRouter 


