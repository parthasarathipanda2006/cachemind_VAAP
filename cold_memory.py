import mysql.connector
import time
import logging
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from BM25 retrieval."""
    id: int
    text: str
    score: float


class ColdStorage:
    """
    A cold storage layer for documents with BM25-based retrieval.

    Features:
    - MySQL backend for persistent storage
    - In-memory BM25 index for fast retrieval
    - Automatic index rebuild on document additions/deletions
    - Access statistics tracking
    - Proper connection and resource management
    """

    def __init__(self, host: str, user: str, password: str, database: str):
        """
        Initialize the ColdStorage instance and establish database connection.

        Args:
            host: MySQL host address
            user: MySQL username
            password: MySQL password
            database: MySQL database name
        """
        self.host = "localhost"
        self.user = "root"
        self.password = "root"
        self.database = "cold_storage"
        self.connection = None
        self.cursor = None
        self.bm25_index = None
        self.documents = []  # List of (id, text) tuples
        self.id_to_index = {}  # Map id to index in documents list
        self.needs_index_rebuild = True
        self.nlp = spacy.load("en_core_web_sm")

        self._connect()
        self._create_table()

    def _connect(self):
        """Establish a connection to the MySQL database."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor()
            logger.info("Connected to MySQL database")
        except mysql.connector.Error as err:
            logger.error(f"Failed to connect to MySQL: {err}")
            raise

    def _create_table(self):
        """Create the documents table if it doesn't exist."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT NOT NULL,
                access_count INT DEFAULT 0,
                last_accessed DOUBLE
                );
                """)
            self.connection.commit()
            logger.info("Documents table ensured")
        except mysql.connector.Error as err:
            logger.error(f"Failed to create table: {err}")
            self.connection.rollback()
            raise

    def _tokenize(self, text: str) -> List[str]:
    
       
        """
        Simple tokenization: split by non-alphanumeric characters and convert to lowercase.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        
        docs=self.nlp(text.lower())
        tokens =[
                    token.lemma_
                    for token in docs
                    if not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                ]
        return  tokens
        # # Convert to lowercase and split by non-alphanumeric characters
        # tokens = re.findall(r'\b\w+\b', text.lower())
        # return tokens

    def _rebuild_index(self):
        """Rebuild the BM25 index from current documents in the database."""

        print("STARTING BM25 REBUILD")
        try:
            # Fetch all documents from database
            self.cursor.execute("SELECT id, text FROM documents")
            rows = self.cursor.fetchall()

            if not rows:
                self.documents = []
                self.bm25_index = None
                self.needs_index_rebuild = False
                logger.info("Index rebuilt: no documents found")
                return

            # Update internal document list and mapping
            self.documents = rows  # List of (id, text)
            self.id_to_index = {id: idx for idx, (id, _) in enumerate(rows)}

            # Tokenize all documents
            print("Tokenizing")
            tokenized_docs = [self._tokenize(text) for _, text in rows]

            # Build BM25 index
            print("BUILDING BM25...")
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.needs_index_rebuild = False
            logger.info(f"Index rebuilt with {len(rows)} documents")

        except mysql.connector.Error as err:
            logger.error(f"Failed to rebuild index: {err}")
            raise

    def add_document(self, text: str) -> int:
        """
        Add a new document to the storage.

        Args:
            text: Document text content

        Returns:
            int: The auto-generated id of the inserted document
        """
        try:
            self.cursor.execute(
                "INSERT INTO documents (text, access_count, last_accessed) VALUES (%s, %s, %s)",
                (text, 0, time.time())
            )
            doc_id = self.cursor.lastrowid
            self.connection.commit()
            self.needs_index_rebuild = True
            logger.info(f"Added document: {doc_id}")
            return doc_id
        except mysql.connector.Error as err:
            logger.error(f"Failed to add document: {err}")
            self.connection.rollback()
            raise

    def get_document(self, id: int) -> Optional[str]:
        """
        Retrieve a document by its ID.

        Args:
            id: Document identifier

        Returns:
            Document text if found, None otherwise
        """
        try:
            self.cursor.execute(
                "SELECT text FROM documents WHERE id = %s",
                (id,)
            )
            result = self.cursor.fetchone()
            if result:
                return result[0]
            return None
        except mysql.connector.Error as err:
            logger.error(f"Failed to get document {id}: {err}")
            return None

    def get_all_documents(self) -> List[Tuple[int, str]]:
        """
        Retrieve all documents from storage.

        Returns:
            List of (id, text) tuples
        """
        try:
            self.cursor.execute("SELECT id, text FROM documents")
            return self.cursor.fetchall()
        except mysql.connector.Error as err:
            logger.error(f"Failed to get all documents: {err}")
            return []

    def update_access_stats(self, id: int):
        """
        Update access statistics for a document.

        Args:
            id: Document identifier
        """
        try:
            self.cursor.execute(
                """UPDATE documents
                   SET access_count = access_count + 1,
                       last_accessed = %s
                   WHERE id = %s""",
                (time.time(), id)
            )
            self.connection.commit()
            logger.debug(f"Updated access stats for document: {id}")
        except mysql.connector.Error as err:
            logger.error(f"Failed to update access stats for {id}: {err}")
            self.connection.rollback()

    def delete_document(self, id: int):
        """
        Delete a document from storage.

        Args:
            id: Document identifier
        """
        try:
            self.cursor.execute(
                "DELETE FROM documents WHERE id = %s",
                (id,)
            )
            self.connection.commit()
            self.needs_index_rebuild = True
            logger.info(f"Deleted document: {id}")
        except mysql.connector.Error as err:
            logger.error(f"Failed to delete document {id}: {err}")
            self.connection.rollback()

    def search_bm25(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search documents using BM25 ranking.

        Args:
            query: Search query string
            top_k: Number of results to return (default: 5)

        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        # Rebuild index if needed
        if self.needs_index_rebuild:
            self._rebuild_index()

        # Handle empty database
        if not self.documents or self.bm25_index is None:
            logger.warning("No documents available for search")
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            logger.warning("Query tokenized to empty list")
            return []

        # Get BM25 scores
       
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices (highest scores first)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Build results and update access stats
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                doc_id, text = self.documents[idx]
                score = float(scores[idx])
                # Update access statistics for this document
                self.update_access_stats(doc_id)
                results.append(SearchResult(id=doc_id, text=text, score=score))

        logger.info(f"BM25 search returned {len(results)} results for query: '{query[:50]}...'")
        return results

    def delete_table(self):
        try:
            self.cursor.execute("DROP TABLE IF EXISTS documents")
            self.connection.commit()
            print("Table 'documents' deleted successfully.")
        except mysql.connector.Error as err:
            print(f"Error deleting table: {err}")
            self.connection.rollback()

    def close(self):
        """Close database connection and cursor."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            logger.info("MySQL connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()