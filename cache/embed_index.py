# src/cache/embed_index.py
"""
EmbedIndex: fixed-size numpy embedding matrix for the cache.

Pre-allocates (capacity × dim) matrix at init.
Tracks which slots are active via a free-slot list.
Similarity search = one batched numpy matmul over active slots only.

Why numpy over FAISS or TorchCacheIndex:
  - Cache is small (≤ 5000 docs typically)
  - No GPU launch overhead
  - O(1) slot insert/delete (just zero the row + return slot)
  - One matmul per query = ~0.1ms on CPU BLAS
"""

import numpy as np
from typing import Optional


class EmbedIndex:
    """
    Fixed-size embedding matrix with slot management.

    Slots are pre-allocated. Insert = assign free slot.
    Remove = zero the row + return slot to free list.
    Search = matmul over active slots only (masked by slot list).
    """

    def __init__(self, capacity: int, dim: int):
        self.capacity = capacity
        self.dim      = dim

        # pre-allocated matrix — all zeros initially
        self._matrix     = np.zeros((capacity, dim), dtype=np.float32)

        # slot management
        self._free_slots : list      = list(range(capacity))
        self._doc_to_slot: dict[str, int] = {}   # doc_id → slot index

    # ─────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._doc_to_slot)

    @property
    def is_full(self) -> bool:
        return len(self._free_slots) == 0

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._doc_to_slot

    # ─────────────────────────────────────────────────────────────
    # Insert / Remove
    # ─────────────────────────────────────────────────────────────

    def add(self, doc_id: str, embedding: np.ndarray):
        """
        Insert embedding into next free slot.
        embedding must be L2-normalized before calling.
        Raises if full or doc_id already exists.
        """
        if doc_id in self._doc_to_slot:
            raise ValueError(f"doc_id '{doc_id}' already in index.")
        if not self._free_slots:
            raise RuntimeError("EmbedIndex is full. Evict before inserting.")

        slot = self._free_slots.pop(0)
        self._matrix[slot]      = embedding.astype(np.float32)
        self._doc_to_slot[doc_id] = slot

    def remove(self, doc_id: str):
        """
        Free the slot for doc_id.
        Zeros the row so it doesn't affect future searches.
        """
        if doc_id not in self._doc_to_slot:
            raise KeyError(f"doc_id '{doc_id}' not in index.")

        slot = self._doc_to_slot.pop(doc_id)
        self._matrix[slot] = 0.0          # zero out — inactive
        self._free_slots.append(slot)

    # ─────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_norm : np.ndarray,
        top_k      : int   = 10,
        threshold  : float = 0.68,
    ) -> list[tuple[str, float]]:
        """
        Cosine similarity search over active docs.
        query_norm must be L2-normalized.

        Returns list of (doc_id, similarity) sorted descending,
        filtered by threshold.
        """
        if not self._doc_to_slot:
            return []

        # build active doc list and their slots
        doc_ids = list(self._doc_to_slot.keys())
        slots   = [self._doc_to_slot[d] for d in doc_ids]

        # one matmul: (n_active, dim) @ (dim,) → (n_active,)
        sims = self._matrix[slots] @ query_norm.astype(np.float32)

        # top-k selection
        k       = min(top_k, len(doc_ids))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        return [
            (doc_ids[i], float(sims[i]))
            for i in top_idx
            if float(sims[i]) >= threshold
        ]

    # ─────────────────────────────────────────────────────────────
    # Nominee helpers (for expert computations)
    # ─────────────────────────────────────────────────────────────

    def get_doc_matrix(self) -> tuple[list[str], Optional[np.ndarray]]:
        """
        Return (doc_ids, matrix) for all active docs.
        Used by SemanticLRU and RDGE experts.
        Returns ([], None) if empty.
        """
        if not self._doc_to_slot:
            return [], None

        doc_ids = list(self._doc_to_slot.keys())
        slots   = [self._doc_to_slot[d] for d in doc_ids]
        return doc_ids, self._matrix[slots]   # (n_active, dim)