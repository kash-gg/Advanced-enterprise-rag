"""
CSV and Database Loader
Loads and converts structured data from CSV files and databases.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime

from llama_index.core import Document
import pandas as pd

logger = logging.getLogger(__name__)


class CSVDatabaseLoader:
    """
    Loader for structured data from CSV files and databases.
    
    Features:
    - Parse CSV files with pandas
    - Support database connections (SQLite, PostgreSQL, MySQL)
    - Convert tabular data to document format
    - Preserve schema information as metadata
    - Handle data type inference
    - Support custom column-to-document mapping
    """
    
    def __init__(
        self,
        text_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None
    ):
        """
        Initialize the CSV/Database loader.
        
        Args:
            text_columns: Columns to combine as document text (None = all non-metadata columns)
            metadata_columns: Columns to store as metadata (None = auto-detect)
        """
        self.text_columns = text_columns
        self.metadata_columns = metadata_columns
        logger.info("CSVDatabaseLoader initialized")
    
    def load_csv(
        self,
        file_path: str,
        sep: str = ',',
        encoding: str = 'utf-8',
        combine_columns: bool = True
    ) -> List[Document]:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to CSV file
            sep: Delimiter character
            encoding: File encoding
            combine_columns: Whether to combine columns into single document per row
            
        Returns:
            List of Document objects
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            logger.info(f"Loading CSV: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path, sep=sep, encoding=encoding)
            
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Convert to documents
            documents = self._dataframe_to_documents(
                df,
                source_name=file_path.name,
                source_type='csv'
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {str(e)}")
            raise
    
    def load_from_database(
        self,
        connection_string: str,
        query: str,
        source_name: Optional[str] = "database"
    ) -> List[Document]:
        """
        Load data from a database using SQL query.
        
        Args:
            connection_string: Database connection string
                Examples:
                - SQLite: "sqlite:///path/to/database.db"
                - PostgreSQL: "postgresql://user:password@localhost:5432/dbname"
                - MySQL: "mysql+pymysql://user:password@localhost:3306/dbname"
            query: SQL query to execute
            source_name: Name to identify the source
            
        Returns:
            List of Document objects
        """
        try:
            logger.info(f"Loading data from database: {source_name}")
            logger.debug(f"Query: {query}")
            
            # Read from database
            df = pd.read_sql_query(query, connection_string)
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from database")
            
            # Convert to documents
            documents = self._dataframe_to_documents(
                df,
                source_name=source_name,
                source_type='database',
                extra_metadata={'connection': connection_string.split('://')[0], 'query': query}
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            raise
    
    def load_excel(
        self,
        file_path: str,
        sheet_name: Union[str, int] = 0
    ) -> List[Document]:
        """
        Load data from an Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index to load
            
        Returns:
            List of Document objects
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Excel file not found: {file_path}")
            
            logger.info(f"Loading Excel: {file_path}, sheet: {sheet_name}")
            
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            logger.info(f"Loaded Excel with {len(df)} rows and {len(df.columns)} columns")
            
            # Convert to documents
            documents = self._dataframe_to_documents(
                df,
                source_name=f"{file_path.name}:{sheet_name}",
                source_type='excel'
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading Excel {file_path}: {str(e)}")
            raise
    
    def _dataframe_to_documents(
        self,
        df: pd.DataFrame,
        source_name: str,
        source_type: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Convert pandas DataFrame to LlamaIndex Documents.
        
        Args:
            df: DataFrame to convert
            source_name: Name of the data source
            source_type: Type of source (csv, database, excel)
            extra_metadata: Additional metadata to include
            
        Returns:
            List of Document objects
        """
        documents = []
        
        # Determine text and metadata columns
        text_cols = self.text_columns if self.text_columns else df.columns.tolist()
        
        if self.metadata_columns:
            meta_cols = self.metadata_columns
            # Remove metadata columns from text columns
            text_cols = [col for col in text_cols if col not in meta_cols]
        else:
            # Auto-detect metadata columns (e.g., id, date, category)
            meta_cols = self._detect_metadata_columns(df)
            text_cols = [col for col in text_cols if col not in meta_cols]
        
        # Convert each row to a document
        for idx, row in df.iterrows():
            # Combine text columns
            text_parts = []
            for col in text_cols:
                value = row[col]
                if pd.notna(value):  # Skip NaN values
                    text_parts.append(f"{col}: {value}")
            
            text = "\n".join(text_parts)
            
            # Extract metadata
            metadata = {
                'source_type': source_type,
                'source_name': source_name,
                'row_index': int(idx),
                'loaded_at': datetime.now().isoformat(),
                'columns': df.columns.tolist(),
                'num_columns': len(df.columns)
            }
            
            # Add metadata from metadata columns
            for col in meta_cols:
                value = row[col]
                if pd.notna(value):
                    # Convert numpy types to Python types
                    if hasattr(value, 'item'):
                        value = value.item()
                    metadata[col] = value
            
            # Add extra metadata
            if extra_metadata:
                metadata.update(extra_metadata)
            
            # Create document
            doc = Document(
                text=text,
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Converted {len(documents)} rows to documents")
        return documents
    
    def _detect_metadata_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect columns that should be treated as metadata.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names to use as metadata
        """
        metadata_cols = []
        
        # Common metadata column names and patterns
        metadata_patterns = [
            'id', 'ID', 'Id',
            'date', 'Date', 'created', 'updated',
            'category', 'Category', 'type', 'Type',
            'author', 'Author', 'created_by', 'updated_by',
            'status', 'Status'
        ]
        
        for col in df.columns:
            # Check if column name matches metadata patterns
            if any(pattern in col for pattern in metadata_patterns):
                metadata_cols.append(col)
            # Check if column has limited unique values (categorical)
            elif df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.1:
                metadata_cols.append(col)
        
        return metadata_cols
    
    def load_batch_csv(self, file_paths: List[str], **kwargs) -> List[Document]:
        """
        Load multiple CSV files in batch.
        
        Args:
            file_paths: List of paths to CSV files
            **kwargs: Additional arguments to pass to load_csv
            
        Returns:
            Combined list of Document objects from all CSVs
        """
        all_documents = []
        successful = 0
        failed = 0
        
        for file_path in file_paths:
            try:
                documents = self.load_csv(file_path, **kwargs)
                all_documents.extend(documents)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                failed += 1
        
        logger.info(f"Batch loading complete: {successful} successful, {failed} failed")
        return all_documents


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loader = CSVDatabaseLoader()
    
    # Example: Load a CSV file
    # documents = loader.load_csv("path/to/data.csv")
    
    # Example: Load from database
    # connection_string = "sqlite:///path/to/database.db"
    # query = "SELECT * FROM table_name"
    # documents = loader.load_from_database(connection_string, query)
    
    # Example: Load Excel file
    # documents = loader.load_excel("path/to/data.xlsx", sheet_name="Sheet1")
    
    print("CSVDatabaseLoader initialized and ready to use!")
