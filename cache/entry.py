# src/cache/entry.py
"""
SCRLEntry: single object living in all 3 cache structures simultaneously.
Same reference in lru (DequeDict), lfu (HeapDict), embedding matrix.
Mutation to .freq visible everywhere instantly — no sync needed.
"""

import numpy as np


class SCRLEntry:
    """
    One entry = one cached document.

    Lives in:
      lru  : DequeDict  → ordered by recency  (SemanticLRU expert)
      lfu  : HeapDict   → ordered by frequency (LFU expert)
      emb  : np.ndarray → for cosine search    (RDGE expert + retrieval)

    Fields
    ------
    doc_id       : unique document identifier
    embedding    : L2-normalized embedding (dim,)
    freq         : access count (incremented on every hit)
    time         : last access time (query index)
    evicted_time : query index when evicted (set by _evict_one)
    """

    __slots__ = ['doc_id', 'embedding', 'freq', 'time', 'evicted_time']

    def __init__(
        self,
        doc_id    : str,
        embedding : np.ndarray,
        freq      : int = 1,
        time      : int = 0,
    ):
        self.doc_id       = doc_id
        self.embedding    = embedding
        self.freq         = freq
        self.time         = time
        self.evicted_time = None

    def __lt__(self, other):
        """
        HeapDict ordering for LFU expert.
        Lower freq = higher eviction priority.
        Tie-break: older time = higher eviction priority.
        Exact same logic as LeCaR's LeCaR_Entry.__lt__
        """
        if self.freq == other.freq:
            return self.time < other.time   # older = evict first
        return self.freq < other.freq

    def __repr__(self):
        return (f"SCRLEntry(id={self.doc_id}, "
                f"freq={self.freq}, "
                f"time={self.time}, "
                f"evicted={self.evicted_time})")