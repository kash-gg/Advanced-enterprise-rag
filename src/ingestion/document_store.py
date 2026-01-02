"""
Unified Document Storage System
Provides persistent storage and management for documents from all sources.
"""

import sqlite3
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from llama_index.core import Document

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    Unified document storage with SQLite backend.
    
    Features:
    - Store documents from any source (PDF, Web, CSV, Database)
    - CRUD operations for document management
    - Metadata indexing and querying
    - Export/import functionality
    - Document versioning support
    """
    
    def __init__(self, db_path: str = "data/documents/document_store.db"):
        """
        Initialize the document store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
        logger.info(f"DocumentStore initialized at {self.db_path}")
    
    def _create_schema(self):
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_path TEXT,
                ingested_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                metadata_json TEXT
            )
        """)
        
        # Metadata table for efficient querying
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)
        
        # Create indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_source 
            ON documents(source_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metadata_key 
            ON metadata(key)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metadata_value 
            ON metadata(value)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metadata_doc_id 
            ON metadata(doc_id)
        """)
        
        self.conn.commit()
        logger.info("Database schema created/verified")
    
    def store_document(self, document: Document, doc_id: Optional[str] = None) -> str:
        """
        Store a single document.
        
        Args:
            document: LlamaIndex Document object
            doc_id: Optional custom document ID (auto-generated if not provided)
            
        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        now = datetime.now().isoformat()
        
        # Extract metadata
        metadata = document.metadata or {}
        source_type = metadata.get('source_type', 'unknown')
        source_path = metadata.get('file_path') or metadata.get('url', '')
        
        cursor = self.conn.cursor()
        
        try:
            # Insert document
            cursor.execute("""
                INSERT INTO documents (doc_id, content, source_type, source_path, 
                                     ingested_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, document.text, source_type, source_path, 
                  now, now, json.dumps(metadata)))
            
            # Insert metadata entries for querying
            for key, value in metadata.items():
                cursor.execute("""
                    INSERT INTO metadata (doc_id, key, value)
                    VALUES (?, ?, ?)
                """, (doc_id, key, str(value)))
            
            self.conn.commit()
            logger.info(f"Stored document {doc_id} ({source_type})")
            return doc_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"Document {doc_id} already exists, use update_document() to modify")
            raise ValueError(f"Document {doc_id} already exists")
    
    def store_batch(self, documents: List[Document]) -> List[str]:
        """
        Store multiple documents in batch.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        successful = 0
        failed = 0
        
        for doc in documents:
            try:
                doc_id = self.store_document(doc)
                doc_ids.append(doc_id)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to store document: {str(e)}")
                failed += 1
        
        logger.info(f"Batch storage complete: {successful} successful, {failed} failed")
        return doc_ids
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document object or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT content, metadata_json 
            FROM documents 
            WHERE doc_id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        if row is None:
            logger.warning(f"Document {doc_id} not found")
            return None
        
        metadata = json.loads(row['metadata_json'])
        document = Document(text=row['content'], metadata=metadata)
        document.doc_id = doc_id
        
        return document
    
    def get_all_documents(self, source_type: Optional[str] = None) -> List[Document]:
        """
        Retrieve all documents, optionally filtered by source type.
        
        Args:
            source_type: Filter by source type (pdf, web, csv, etc.)
            
        Returns:
            List of Document objects
        """
        cursor = self.conn.cursor()
        
        if source_type:
            cursor.execute("""
                SELECT doc_id, content, metadata_json 
                FROM documents 
                WHERE source_type = ?
            """, (source_type,))
        else:
            cursor.execute("""
                SELECT doc_id, content, metadata_json 
                FROM documents
            """)
        
        documents = []
        for row in cursor.fetchall():
            metadata = json.loads(row['metadata_json'])
            doc = Document(text=row['content'], metadata=metadata)
            doc.doc_id = row['doc_id']
            documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} documents" + 
                   (f" (source_type={source_type})" if source_type else ""))
        return documents
    
    def query_by_metadata(self, filters: Dict[str, Any]) -> List[Document]:
        """
        Query documents by metadata filters.
        
        Args:
            filters: Dictionary of metadata key-value pairs to match
            
        Returns:
            List of matching Document objects
        """
        cursor = self.conn.cursor()
        
        # Build query with multiple metadata filters
        query = """
            SELECT DISTINCT d.doc_id, d.content, d.metadata_json
            FROM documents d
        """
        
        conditions = []
        params = []
        
        for idx, (key, value) in enumerate(filters.items()):
            alias = f"m{idx}"
            query += f" INNER JOIN metadata {alias} ON d.doc_id = {alias}.doc_id"
            conditions.append(f"{alias}.key = ? AND {alias}.value = ?")
            params.extend([key, str(value)])
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        
        documents = []
        for row in cursor.fetchall():
            metadata = json.loads(row['metadata_json'])
            doc = Document(text=row['content'], metadata=metadata)
            doc.doc_id = row['doc_id']
            documents.append(doc)
        
        logger.info(f"Query returned {len(documents)} documents matching {filters}")
        return documents
    
    def update_document(self, doc_id: str, document: Document):
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID to update
            document: New Document object with updated content/metadata
        """
        cursor = self.conn.cursor()
        
        # Check if document exists
        cursor.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"Document {doc_id} not found")
        
        now = datetime.now().isoformat()
        metadata = document.metadata or {}
        source_type = metadata.get('source_type', 'unknown')
        source_path = metadata.get('file_path') or metadata.get('url', '')
        
        # Update document
        cursor.execute("""
            UPDATE documents 
            SET content = ?, source_type = ?, source_path = ?, 
                updated_at = ?, metadata_json = ?
            WHERE doc_id = ?
        """, (document.text, source_type, source_path, now, 
              json.dumps(metadata), doc_id))
        
        # Delete old metadata entries
        cursor.execute("DELETE FROM metadata WHERE doc_id = ?", (doc_id,))
        
        # Insert new metadata entries
        for key, value in metadata.items():
            cursor.execute("""
                INSERT INTO metadata (doc_id, key, value)
                VALUES (?, ?, ?)
            """, (doc_id, key, str(value)))
        
        self.conn.commit()
        logger.info(f"Updated document {doc_id}")
    
    def delete_document(self, doc_id: str):
        """
        Delete a document.
        
        Args:
            doc_id: Document ID to delete
        """
        cursor = self.conn.cursor()
        
        # Delete metadata entries (will cascade if foreign key constraints enabled)
        cursor.execute("DELETE FROM metadata WHERE doc_id = ?", (doc_id,))
        
        # Delete document
        cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        
        if cursor.rowcount == 0:
            logger.warning(f"Document {doc_id} not found")
            raise ValueError(f"Document {doc_id} not found")
        
        self.conn.commit()
        logger.info(f"Deleted document {doc_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        cursor = self.conn.cursor()
        
        # Total documents
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        total = cursor.fetchone()['count']
        
        # Documents by source type
        cursor.execute("""
            SELECT source_type, COUNT(*) as count 
            FROM documents 
            GROUP BY source_type
        """)
        by_source = {row['source_type']: row['count'] for row in cursor.fetchall()}
        
        # Database size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return {
            'total_documents': total,
            'by_source_type': by_source,
            'database_size_bytes': db_size,
            'database_size_mb': round(db_size / (1024 * 1024), 2)
        }
    
    def export_to_json(self, output_path: str):
        """
        Export all documents to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        documents = self.get_all_documents()
        
        export_data = []
        for doc in documents:
            export_data.append({
                'doc_id': doc.doc_id,
                'content': doc.text,
                'metadata': doc.metadata
            })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(export_data)} documents to {output_path}")
    
    def import_from_json(self, input_path: str):
        """
        Import documents from JSON file.
        
        Args:
            input_path: Path to JSON file
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        imported = 0
        for item in import_data:
            doc = Document(
                text=item['content'],
                metadata=item['metadata']
            )
            try:
                self.store_document(doc, doc_id=item.get('doc_id'))
                imported += 1
            except Exception as e:
                logger.error(f"Failed to import document: {str(e)}")
        
        logger.info(f"Imported {imported}/{len(import_data)} documents from {input_path}")
    
    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("DocumentStore closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create document store
    with DocumentStore() as store:
        # Create sample document
        sample_doc = Document(
            text="This is a sample document.",
            metadata={
                'source_type': 'pdf',
                'title': 'Sample Document',
                'author': 'Test User'
            }
        )
        
        # Store document
        doc_id = store.store_document(sample_doc)
        print(f"Stored document: {doc_id}")
        
        # Retrieve document
        retrieved = store.get_document(doc_id)
        print(f"Retrieved: {retrieved.text}")
        
        # Get stats
        stats = store.get_stats()
        print(f"Storage stats: {stats}")
    
    print("DocumentStore initialized and ready to use!")
