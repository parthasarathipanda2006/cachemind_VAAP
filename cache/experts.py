# src/cache/experts.py
"""
Three eviction experts for SCRL.

Each expert nominates one doc to evict.
All three run on every eviction — Hedge picks which one to use.

E0 SemanticLRU : evict doc least similar to recent queries
E1 LFU         : evict doc with lowest access frequency
E2 RDGE        : evict doc least similar to recent queries
                 AND recent cold-tier retrieved docs
"""

import numpy as np
from typing import Optional
from cache.structures import DequeDict, HeapDict
from cache.embed_index import EmbedIndex


def nominee_lfu(lfu: HeapDict) -> Optional[str]:
    """
    E1: LFU nominee.
    Simply return the doc_id at the top of the min-heap.
    O(1) — no computation needed.
    """
    entry = lfu.min()
    return entry.doc_id if entry else None


def nominee_semantic_lru(
    embed_index   : EmbedIndex,
    recent_queries: list,
    lfu           : HeapDict,
) -> Optional[str]:
    """
    E0: SemanticLRU nominee.
    Evict doc with lowest MAX cosine similarity to recent queries.

    If no recent queries → fall back to LFU nominee.
    Uses vectorized matmul — one call for all docs.

    Parameters
    ----------
    embed_index    : EmbedIndex holding current cached docs
    recent_queries : list of L2-normalized query embeddings (window)
    lfu            : HeapDict fallback when no queries available
    """
    if not recent_queries:
        return nominee_lfu(lfu)

    doc_ids, doc_mat = embed_index.get_doc_matrix()
    if doc_mat is None:
        return None

    # q_mat: (window_size, dim)
    q_mat = np.array(recent_queries, dtype=np.float32)

    # sims: (n_docs, window_size)
    sims = doc_mat @ q_mat.T

    # retention score = max similarity to any recent query
    # low score = doc is semantically cold = evict it
    scores = sims.max(axis=1)   # (n_docs,)

    return doc_ids[int(np.argmin(scores))]


def nominee_rdge(
    embed_index      : EmbedIndex,
    recent_queries   : list,
    recent_retrieved : list,
    lfu              : HeapDict,
) -> Optional[str]:
    """
    E2: RDGE (Retrieved-Document-Guided Eviction) nominee.
    Evict doc least similar to BOTH recent queries AND
    recently retrieved cold-tier docs.

    Retention score = max(sim_to_queries, sim_to_retrieved)
    Evict doc with lowest retention score.

    If neither window has entries → fall back to LFU.

    Parameters
    ----------
    embed_index       : EmbedIndex holding current cached docs
    recent_queries    : list of L2-normalized query embeddings
    recent_retrieved  : list of L2-normalized cold-tier doc embeddings
    lfu               : HeapDict fallback
    """
    if not recent_queries and not recent_retrieved:
        return nominee_lfu(lfu)

    doc_ids, doc_mat = embed_index.get_doc_matrix()
    if doc_mat is None:
        return None

    # start with zeros — will take max with each signal
    scores = np.zeros(len(doc_ids), dtype=np.float32)

    if recent_queries:
        q_mat  = np.array(recent_queries,   dtype=np.float32)
        scores = np.maximum(scores, (doc_mat @ q_mat.T).max(axis=1))

    if recent_retrieved:
        r_mat  = np.array(recent_retrieved, dtype=np.float32)
        scores = np.maximum(scores, (doc_mat @ r_mat.T).max(axis=1))

    return doc_ids[int(np.argmin(scores))]


def get_all_nominees(
    embed_index      : EmbedIndex,
    lfu              : HeapDict,
    recent_queries   : list,
    recent_retrieved : list,
) -> list[Optional[str]]:
    """
    Get one nominee from each expert.
    Returns [sem_lru_nominee, lfu_nominee, rdge_nominee].
    Called once per eviction.
    """
    return [
        nominee_semantic_lru(embed_index, recent_queries, lfu),
        nominee_lfu(lfu),
        nominee_rdge(embed_index, recent_queries, recent_retrieved, lfu),
    ]