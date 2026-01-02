"""
Example: Document Storage System Usage
Demonstrates loading documents from multiple sources and storing them in unified storage.
"""

import logging
from pathlib import Path

from src.ingestion import PDFLoader, WebLoader, CSVDatabaseLoader, DocumentStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate document storage system."""
    
    print("=" * 70)
    print("Document Storage System - Example Usage")
    print("=" * 70)
    print()
    
    # Initialize document store
    store = DocumentStore("data/documents/example_store.db")
    
    # Initialize loaders
    pdf_loader = PDFLoader()
    web_loader = WebLoader()
    csv_loader = CSVDatabaseLoader()
    
    print("✓ Document store and loaders initialized")
    print()
    
    # Example 1: Load and store PDF documents
    print("Example 1: Loading PDF Documents")
    print("-" * 70)
    
    # Check if there are any PDFs in data directory
    pdf_dir = Path("data/documents")
    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files")
        for pdf_file in pdf_files[:2]:  # Process first 2 PDFs
            try:
                documents = pdf_loader.load_pdf(str(pdf_file))
                doc_ids = store.store_batch(documents)
                print(f"✓ Stored {len(doc_ids)} pages from {pdf_file.name}")
            except Exception as e:
                print(f"✗ Failed to load {pdf_file.name}: {str(e)}")
    else:
        print("ℹ No PDF files found in data/documents/")
        print("  (This is just an example - you can add PDFs later)")
    print()
    
    # Example 2: Load and store web content
    print("Example 2: Loading Web Content")
    print("-" * 70)
    
    # Example URLs (you can modify these)
    example_urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        # Add more URLs as needed
    ]
    
    print(f"Loading {len(example_urls)} web page(s)...")
    for url in example_urls:
        try:
            documents = web_loader.load_url(url)
            if documents:
                doc_ids = store.store_batch(documents)
                print(f"✓ Stored content from {url}")
            else:
                print(f"✗ No content extracted from {url}")
        except Exception as e:
            print(f"✗ Failed to load {url}: {str(e)}")
    print()
    
    # Example 3: Load and store CSV data
    print("Example 3: Loading CSV Data")
    print("-" * 70)
    
    # Check if there are any CSV files
    csv_files = list(pdf_dir.glob("*.csv")) if pdf_dir.exists() else []
    
    if csv_files:
        print(f"Found {len(csv_files)} CSV files")
        for csv_file in csv_files[:2]:  # Process first 2 CSVs
            try:
                documents = csv_loader.load_csv(str(csv_file))
                doc_ids = store.store_batch(documents)
                print(f"✓ Stored {len(doc_ids)} records from {csv_file.name}")
            except Exception as e:
                print(f"✗ Failed to load {csv_file.name}: {str(e)}")
    else:
        print("ℹ No CSV files found in data/documents/")
        print("  (This is just an example - you can add CSVs later)")
    print()
    
    # Example 4: Query documents
    print("Example 4: Querying Documents")
    print("-" * 70)
    
    # Get all documents
    all_docs = store.get_all_documents()
    print(f"Total documents in storage: {len(all_docs)}")
    
    # Get documents by source type
    pdf_docs = store.get_all_documents(source_type='pdf')
    web_docs = store.get_all_documents(source_type='web')
    csv_docs = store.get_all_documents(source_type='csv')
    
    print(f"  - PDF documents: {len(pdf_docs)}")
    print(f"  - Web documents: {len(web_docs)}")
    print(f"  - CSV documents: {len(csv_docs)}")
    print()
    
    # Query by metadata
    if all_docs:
        print("Querying documents with metadata filters...")
        
        # Example: Find all documents of a specific type
        results = store.query_by_metadata({'source_type': 'web'})
        print(f"  - Web documents: {len(results)}")
        
        # You can add more complex queries here
        # results = store.query_by_metadata({'author': 'John Doe', 'year': '2024'})
    print()
    
    # Example 5: Storage statistics
    print("Example 5: Storage Statistics")
    print("-" * 70)
    
    stats = store.get_stats()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Database size: {stats['database_size_mb']} MB")
    print("Documents by source type:")
    for source_type, count in stats['by_source_type'].items():
        print(f"  - {source_type}: {count}")
    print()
    
    # Example 6: Export/Import
    print("Example 6: Export Documents")
    print("-" * 70)
    
    if all_docs:
        export_path = "data/documents/exported_documents.json"
        store.export_to_json(export_path)
        print(f"✓ Exported all documents to {export_path}")
    else:
        print("ℹ No documents to export")
    print()
    
    # Close the store
    store.close()
    
    print("=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Add your PDF, CSV, or other documents to data/documents/")
    print("  2. Modify the example URLs to load web content you need")
    print("  3. Query and retrieve documents using metadata filters")
    print("  4. Use the stored documents for RAG pipeline processing")
    print()


if __name__ == "__main__":
    main()
