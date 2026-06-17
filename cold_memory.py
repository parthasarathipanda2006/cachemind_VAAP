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
    doc_id: str
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
        self.host ="localhost"
        self.user ="root"
        self.password = "root"
        self.database ="cold_storage"
        self.connection = None
        self.cursor = None
        self.bm25_index = None
        self.documents = []  # List of (doc_id, text) tuples
        self.doc_id_to_index = {}  # Map doc_id to index in documents list
        self.needs_index_rebuild = True

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
                doc_id VARCHAR(255),
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
        nlp = spacy.load("en_core_web_sm")
        docs=nlp(text.lower())
        tokens=[token.lemma_ for token in docs]
        return  tokens
        # # Convert to lowercase and split by non-alphanumeric characters
        # tokens = re.findall(r'\b\w+\b', text.lower())
        # return tokens

    def _rebuild_index(self):
        """Rebuild the BM25 index from current documents in the database."""
        try:
            # Fetch all documents from database
            self.cursor.execute("SELECT doc_id, text FROM documents")
            rows = self.cursor.fetchall()

            if not rows:
                self.documents = []
                self.doc_id_to_index = {}
                self.bm25_index = None
                self.needs_index_rebuild = False
                logger.info("Index rebuilt: no documents found")
                return

            # Update internal document list and mapping
            self.documents = rows  # List of (doc_id, text)
            self.doc_id_to_index = {doc_id: idx for idx, (doc_id, _) in enumerate(rows)}

            # Tokenize all documents
            tokenized_docs = [self._tokenize(text) for _, text in rows]

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.needs_index_rebuild = False
            logger.info(f"Index rebuilt with {len(rows)} documents")

        except mysql.connector.Error as err:
            logger.error(f"Failed to rebuild index: {err}")
            raise

    def add_document(self, doc_id: str, text: str):
        """
        Add a new document to the storage.

        Args:
            doc_id: Unique identifier for the document
            text: Document text content
        """
        try:
            self.cursor.execute(
                """INSERT INTO documents (doc_id, text, access_count, last_accessed)
                   VALUES (%s, %s, %s, %s)""",
                (doc_id, text, 0, time.time())
            )
            self.connection.commit()
            self.needs_index_rebuild = True
            logger.info(f"Added document: {doc_id}")
        except mysql.connector.Error as err:
            logger.error(f"Failed to add document {doc_id}: {err}")
            self.connection.rollback()
            raise

    def get_document(self, doc_id: str) -> Optional[str]:
        """
        Retrieve a document by its ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document text if found, None otherwise
        """
        try:
            self.cursor.execute(
                "SELECT text FROM documents WHERE doc_id = %s",
                (doc_id,)
            )
            result = self.cursor.fetchone()
            if result:
                return result[0]
            return None
        except mysql.connector.Error as err:
            logger.error(f"Failed to get document {doc_id}: {err}")
            return None

    def get_all_documents(self) -> List[Tuple[str, str]]:
        """
        Retrieve all documents from storage.

        Returns:
            List of (doc_id, text) tuples
        """
        try:
            self.cursor.execute("SELECT doc_id, text FROM documents")
            return self.cursor.fetchall()
        except mysql.connector.Error as err:
            logger.error(f"Failed to get all documents: {err}")
            return []

    def update_access_stats(self, doc_id: str):
        """
        Update access statistics for a document.

        Args:
            doc_id: Document identifier
        """
        try:
            self.cursor.execute(
                """UPDATE documents
                   SET access_count = access_count + 1,
                       last_accessed = %s
                   WHERE doc_id = %s""",
                (time.time(), doc_id)
            )
            self.connection.commit()
            logger.debug(f"Updated access stats for document: {doc_id}")
        except mysql.connector.Error as err:
            logger.error(f"Failed to update access stats for {doc_id}: {err}")
            self.connection.rollback()

    def delete_document(self, doc_id: str):
        """
        Delete a document from storage.

        Args:
            doc_id: Document identifier
        """
        try:
            self.cursor.execute(
                "DELETE FROM documents WHERE doc_id = %s",
                (doc_id,)
            )
            self.connection.commit()
            self.needs_index_rebuild = True
            logger.info(f"Deleted document: {doc_id}")
        except mysql.connector.Error as err:
            logger.error(f"Failed to delete document {doc_id}: {err}")
            self.connection.rollback()
            raise

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
                results.append(SearchResult(doc_id=doc_id, text=text, score=score))

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