# src/cache/scrl_cache.py
"""
SCRLCache: Full SCRL cache class.

Wires together all 7 components:
  SCRLEntry     : single entry living in all structures
  DequeDict     : LRU ordering
  HeapDict      : LFU ordering
  EmbedIndex    : embedding matrix + similarity search
  experts       : 3 nominee functions
  HedgeWeights  : weight update (LeCaR Algorithm 2 + CE)
  HistoryQueues : 3 FIFO ghost caches

Public interface (mirrors LeCaR's request/miss pattern):
  request(query_norm)              → list of (doc_id, sim)
  insert(doc_id, emb, ce_score)    → None
  update_retrieved(embeddings)     → None  [RDGE signal]
  get_stats()                      → dict
"""

import numpy as np
from cache.structures  import DequeDict, HeapDict
from cache.entry       import SCRLEntry
from cache.embed_index import EmbedIndex
from cache.experts     import get_all_nominees
from cache.hedge       import HedgeWeights
from cache.history     import HistoryQueues

E_SEM_LRU, E_LFU, E_RDGE = 0, 1, 2
N_EXPERTS = 3


class SCRLCache:

    def __init__(
        self,
        capacity      : int,
        dim           : int   = 384,
        learning_rate : float = 0.45,
        discount_rate : float = None,
        window_size   : int   = 50,
        threshold     : float = 0.68,
    ):
        self.capacity   = capacity
        self.dim        = dim
        self.threshold  = threshold
        self.time       = 0
        self.window_size = window_size

        # ── Core structures ──────────────────────────────────────
        self.lru         = DequeDict()
        self.lfu         = HeapDict()
        self.embed_index = EmbedIndex(capacity, dim)

        # ── Hedge + History ──────────────────────────────────────
        self.hedge   = HedgeWeights(
            n_experts     = N_EXPERTS,
            learning_rate = learning_rate,
            discount_rate = discount_rate,
            capacity      = capacity,
        )
        self.history = HistoryQueues(hist_size=capacity)

        # ── Semantic windows ─────────────────────────────────────
        self._q_window   : list = []   # recent query embeddings
        self._ret_window : list = []   # recent cold-tier doc embeddings

        # ── Stats ────────────────────────────────────────────────
        self.hits   = 0
        self.misses = 0
        self._expert_evictions = np.zeros(N_EXPERTS, dtype=np.int64)

    # ─────────────────────────────────────────────────────────────
    # Internal: add / remove from all structures
    # ─────────────────────────────────────────────────────────────

    def _add(self, doc_id: str, emb_norm: np.ndarray, freq: int = 1):
        """Add one Entry to LRU, LFU, EmbedIndex simultaneously."""
        entry = SCRLEntry(doc_id, emb_norm, freq=freq, time=self.time)
        self.lru[doc_id]         = entry
        self.lfu[doc_id]         = entry
        self.embed_index.add(doc_id, emb_norm)

    def _remove(self, doc_id: str):
        """Remove from all three structures."""
        del self.lru[doc_id]
        del self.lfu[doc_id]
        self.embed_index.remove(doc_id)

    # ─────────────────────────────────────────────────────────────
    # Internal: eviction
    # ─────────────────────────────────────────────────────────────

    def _evict_one(self):
        """
        LeCaR-faithful eviction:
        1. Get one nominee from each expert.
        2. Sample expert via Hedge weights.
        3. If all nominees agree → policy=-1, no ghost.
        4. Evict, stamp evicted_time, add to history.
        """
        nominees = get_all_nominees(
            self.embed_index,
            self.lfu,
            self._q_window,
            self._ret_window,
        )

        # sample expert
        expert_idx = self.hedge.sample()
        evict_id   = nominees[expert_idx]

        # fallback if nominee is None
        if evict_id is None:
            for i in range(N_EXPERTS):
                if nominees[i] is not None:
                    evict_id, expert_idx = nominees[i], i
                    break

        if evict_id is None:
            return   # cache is empty, nothing to evict

        # all-agree → policy=-1 (LeCaR: no ghost entry)
        unique = {n for n in nominees if n is not None}
        if len(unique) == 1 and len(self.lru) == 1:
            expert_idx = -1

        # evict
        entry              = self.lru[evict_id]
        entry.evicted_time = self.time
        self._remove(evict_id)

        # add to history (ghost cache)
        self.history.add(entry, expert_idx)

        if expert_idx >= 0:
            self._expert_evictions[expert_idx] += 1

    # ─────────────────────────────────────────────────────────────
    # Internal: hit update
    # ─────────────────────────────────────────────────────────────

    def _hit_update(self, doc_id: str):
        """
        On cache hit: update recency (LRU) and frequency (LFU).
        Same object reference — mutation visible in both structures.
        """
        entry      = self.lru[doc_id]
        entry.freq += 1
        entry.time  = self.time
        self.lru[doc_id] = entry   # moves to MRU end
        self.lfu[doc_id] = entry   # triggers heap reorder

    # ─────────────────────────────────────────────────────────────
    # Public: request
    # ─────────────────────────────────────────────────────────────

    def request(self, query_norm: np.ndarray) -> list:
        """
        LeCaR-faithful request().

        1. Increment time.
        2. Update query window.
        3. Search embed_index for top-10 similar docs.
        4. If ≥ 10 results above threshold → hit, update freq/recency.
        5. Otherwise → miss.

        Returns list of (doc_id, similarity) or [].
        """
        self.time += 1

        # update query window (FIFO, bounded to window_size)
        self._q_window.append(query_norm.astype(np.float32))
        if len(self._q_window) > self.window_size:
            self._q_window.pop(0)

        if self.embed_index.size == 0:
            self.misses += 1
            return []

        results = self.embed_index.search(
            query_norm, top_k=10, threshold=self.threshold
        )

        if len(results) >= 10:
            self.hits += 1
            for doc_id, _ in results:
                if doc_id in self.lru:
                    self._hit_update(doc_id)
        else:
            self.misses += 1
        
        return results

    # ─────────────────────────────────────────────────────────────
    # Public: insert
    # ─────────────────────────────────────────────────────────────

    def insert(self, doc_id: str, emb: np.ndarray, ce_score: float = 1.0):
        """
        LeCaR-faithful miss().

        1. If already cached → just update freq/recency, return.
        2. Check history → if found, penalize that expert via Hedge.
        3. Carry forward frequency from history (LeCaR principle).
        4. Evict if cache is full.
        5. Add to cache.
        """
        # already cached
        if doc_id in self.lru:
            self._hit_update(doc_id)
            return

        # check history for regrettable miss
        expert_idx, hist_entry = self.history.check(doc_id)

        if expert_idx >= 0:
            # regrettable miss → weight update
            self.hedge.update(
                expert_idx   = expert_idx,
                ce_score     = ce_score,
                evicted_time = hist_entry.evicted_time,
                current_time = self.time,
            )
            freq = hist_entry.freq + 1   # carry forward (LeCaR)
        else:
            freq = 1

        # evict if full
        if len(self.lru) >= self.capacity:
            self._evict_one()

        # normalize embedding
        n = np.linalg.norm(emb)
        emb_norm = (emb / n).astype(np.float32) if n > 1e-10 else emb.astype(np.float32)

        self._add(doc_id, emb_norm, freq=freq)

    # ─────────────────────────────────────────────────────────────
    # Public: RDGE signal
    # ─────────────────────────────────────────────────────────────

    def update_retrieved(self, embeddings: list):
        """
        Update RDGE window with cold-tier retrieved doc embeddings.
        Call this after every cold-tier retrieval (on miss).
        """
        for emb in embeddings:
            n = np.linalg.norm(emb)
            if n > 1e-10:
                self._ret_window.append(
                    (emb / n).astype(np.float32)
                )
        if len(self._ret_window) > self.window_size:
            self._ret_window = self._ret_window[-self.window_size:]

    # ─────────────────────────────────────────────────────────────
    # Public: stats
    # ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        w     = self.hedge.get_weights()
        return {
            "size"       : self.embed_index.size,
            "hit_rate"   : round(self.hits / total, 4) if total else 0.0,
            "hits"       : self.hits,
            "misses"     : self.misses,
            "weights"    : {
                "SemLRU" : round(float(w[E_SEM_LRU]), 4),
                "LFU"    : round(float(w[E_LFU]),     4),
                "RDGE"   : round(float(w[E_RDGE]),    4),
            },
            "expert_evictions" : {
                "SemLRU" : int(self._expert_evictions[E_SEM_LRU]),
                "LFU"    : int(self._expert_evictions[E_LFU]),
                "RDGE"   : int(self._expert_evictions[E_RDGE]),
            },
            "weight_updates" : self.hedge.n_updates,
            "history_hits"   : self.history.total_hits,
        }