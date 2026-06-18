"""
LRU (Least Recently Used) cache policy implementation.

This policy tracks document IDs in the cache and their access order to
implement LRU eviction strategy.
"""

from collections import OrderedDict
from typing import Optional
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LRUPolicy:
    """
    Least Recently Used (LRU) cache policy.

    Maintains cache membership and access order to evict the least recently
    used item when the cache is full.

    Attributes:
        _max_size (int): Maximum number of items the cache can hold.
        _cache (OrderedDict): Stores document IDs in access order (LRU at front).
    """

    def __init__(self, max_cache_size: int):
        """
        Initialize the LRU policy.

        Args:
            max_cache_size: Maximum number of documents to cache.
        """
        if max_cache_size <= 0:
            raise ValueError("max_cache_size must be positive")
        self._max_size = max_cache_size
        self._cache = OrderedDict()
        logger.info(f"LRUPolicy initialized with max_cache_size={max_cache_size}")

    def contains(self, doc_id: str) -> bool:
        """
        Check if a document ID is in the cache.

        Args:
            doc_id: Document identifier to check.

        Returns:
            True if doc_id is in cache, False otherwise.
        """
        return doc_id in self._cache

    def admit(self, doc_id: str) -> None:
        """
        Admit a document to the cache as most recently used.

        If the document is already in the cache, it is moved to the most
        recently used position. Assumes cache has space (checked by caller).

        Args:
            doc_id: Document identifier to admit.
        """
        if doc_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(doc_id)
            logger.debug(f"Document {doc_id} already in cache, moved to MRU")
        else:
            # Add new document
            self._cache[doc_id] = None
            self._cache.move_to_end(doc_id)
            logger.debug(f"Admitted document {doc_id} to cache")

        # Log if cache becomes full after admission
        if self.is_full():
            logger.debug("Cache is now full")

    def record_access(self, doc_id: str) -> None:
        """
        Record an access to a document, marking it as most recently used.

        Does nothing if document is not in cache.

        Args:
            doc_id: Document identifier that was accessed.
        """
        if doc_id in self._cache:
            self._cache.move_to_end(doc_id)
            logger.debug(f"Recorded access for document {doc_id}")
        else:
            logger.debug(f"Attempted to record access for non-cached document {doc_id}")

    def evict_candidate(self) -> Optional[str]:
        """
        Get the least recently used document ID candidate for eviction.

        Does not remove the document from cache.

        Returns:
            The doc_id of the least recently used document, or None if cache is empty.
        """
        if not self._cache:
            logger.debug("Cache is empty, no eviction candidate")
            return None
        # The first item in OrderedDict is the least recently used
        lru_doc_id = next(iter(self._cache))
        logger.debug(f"Eviction candidate: {lru_doc_id}")
        return lru_doc_id

    def remove(self, doc_id: str) -> None:
        """
        Remove a document from the cache.

        Args:
            doc_id: Document identifier to remove.
        """
        if doc_id in self._cache:
            del self._cache[doc_id]
            logger.debug(f"Removed document {doc_id} from cache")
        else:
            logger.debug(f"Attempted to remove non-cached document {doc_id}")

    def is_full(self) -> bool:
        """
        Check if the cache is full.

        Returns:
            True if cache has reached max capacity, False otherwise.
        """
        return len(self._cache) >= self._max_size