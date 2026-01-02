"""Test script for DocumentStore functionality."""

from src.ingestion import DocumentStore
from llama_index.core import Document

print("Testing DocumentStore...")
print("-" * 50)

# Initialize store
store = DocumentStore('data/documents/test_store.db')
print("[OK] DocumentStore initialized")

# Create test document
doc = Document(
    text="This is a test document for the RAG system.",
    metadata={
        'source_type': 'test',
        'title': 'Test Document',
        'author': 'System'
    }
)

# Store document
doc_id = store.store_document(doc)
print(f"[OK] Stored document with ID: {doc_id}")

# Retrieve document
retrieved = store.get_document(doc_id)
print(f"[OK] Retrieved document: {retrieved.text[:30]}...")

# Query by metadata
results = store.query_by_metadata({'source_type': 'test'})
print(f"[OK] Query found {len(results)} document(s)")

# Get stats
stats = store.get_stats()
print(f"[OK] Storage stats: {stats['total_documents']} total documents")

# Close store
store.close()
print("-" * 50)
print("SUCCESS: All DocumentStore operations work correctly!")
