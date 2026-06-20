# src/cache/history.py
"""
HistoryQueues: three separate FIFO ghost caches, one per expert.

Faithfully follows LeCaR:
  - History size = cache capacity (exactly N entries per queue)
  - Each queue is FIFO — oldest entry dropped when full
  - Each entry labeled by which expert evicted it
  - Weight update fires ONLY when missed doc found in history
  - policy=-1 means all experts agreed → no ghost entry created

On eviction:
    entry.evicted_time = current_time
    hist[expert_idx].add(entry)

On miss (doc requested that was evicted):
    expert_idx, entry = hist.check(doc_id)
    if expert_idx >= 0:
        hedge.update(expert_idx, ce_score, entry.evicted_time, current_time)
        freq = entry.freq + 1    # carry forward frequency (LeCaR)
"""

from cache.structures import DequeDict
from cache.entry      import SCRLEntry
from typing               import Optional

N_EXPERTS = 3


class HistoryQueues:
    """
    Three FIFO DequeDict queues, one per expert.
    Each bounded to hist_size entries.

    Key operations
    --------------
    add(entry, expert_idx)  : add evicted entry to expert's queue
    check(doc_id)           : find doc_id in any queue
                              returns (expert_idx, entry) or (-1, None)
    """

    def __init__(self, hist_size: int):
        """
        Parameters
        ----------
        hist_size : max entries per queue (= cache capacity, like LeCaR)
        """
        self.hist_size = hist_size
        self.queues    = [DequeDict() for _ in range(N_EXPERTS)]

        # stats
        self.total_added   = 0
        self.total_hits    = 0    # times a missed doc was found in history
        self.total_misses  = 0    # times a missed doc was NOT in history

    def add(self, entry: SCRLEntry, expert_idx: int):
        """
        Add evicted entry to expert_idx's FIFO queue.
        If queue is full → drop oldest entry (FIFO).
        policy=-1 → skip (all experts agreed, no attribution).
        """
        if expert_idx < 0:
            return                           # policy=-1, no ghost

        q = self.queues[expert_idx]

        # FIFO eviction from history when full
        if len(q) == self.hist_size:
            q.popFirst()                     # drop oldest

        q[entry.doc_id] = entry
        self.total_added += 1

    def check(self, doc_id: str) -> tuple[int, Optional[SCRLEntry]]:
        """
        Check all three queues for doc_id.
        If found → remove it, return (expert_idx, entry).
        If not found → return (-1, None).

        Checks queues in order 0, 1, 2.
        A doc can only be in one queue (evicted by one expert).
        """
        for i, q in enumerate(self.queues):
            if doc_id in q:
                entry = q[doc_id]
                del q[doc_id]
                self.total_hits += 1
                return i, entry

        self.total_misses += 1
        return -1, None

    def sizes(self) -> list[int]:
        """Return current size of each queue."""
        return [len(q) for q in self.queues]

    def total_size(self) -> int:
        return sum(len(q) for q in self.queues)

    def __repr__(self):
        return (f"HistoryQueues("
                f"sizes={self.sizes()}, "
                f"hits={self.total_hits}, "
                f"misses={self.total_misses})")