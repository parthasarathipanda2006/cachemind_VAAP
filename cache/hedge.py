# src/cache/hedge.py
"""
Hedge weight update for SCRL.

Faithfully follows LeCaR's Algorithm 2 (UPDATEWEIGHT),
extended with CE score as graded regret signal.

LeCaR formula:
    reward   = -(discount_rate ^ t)        # t = time in history
    W[expert] *= exp(learning_rate * reward)
    W = W / sum(W)
    W = clip(W, 0.01, 0.99)

SCRL extension:
    reward   = -(ce_score * discount_rate ^ t)
    # ce_score ∈ [0,1]: how relevant was the evicted doc?
    # high CE = expert made a bad mistake = larger penalty
    # when ce_score=1.0 → reduces exactly to LeCaR formula
"""

import numpy as np

N_EXPERTS = 3


class HedgeWeights:
    """
    Manages Hedge weights for N_EXPERTS experts.

    State
    -----
    W       : weight array (N_EXPERTS,), sums to 1.0
    regret  : cumulative regret per expert (for logging/plotting)

    Methods
    -------
    sample()        : sample expert index proportional to W
    update()        : penalize one expert (LeCaR Algorithm 2)
    get_weights()   : return copy of W
    """

    def __init__(
        self,
        n_experts     : int   = N_EXPERTS,
        learning_rate : float = 0.45,         # LeCaR default
        discount_rate : float = None,         # set per cache size
        capacity      : int   = 100,          # used to compute discount_rate
    ):
        self.n_experts     = n_experts
        self.learning_rate = learning_rate
        self.discount_rate = (discount_rate if discount_rate is not None
                              else 0.005 ** (1.0 / capacity))

        # weights start equal — same as LeCaR
        self.W      = np.ones(n_experts, dtype=np.float64) / n_experts

        # cumulative regret per expert (for plotting)
        self.regret = np.zeros(n_experts, dtype=np.float64)

        # stats
        self.n_updates = 0

    def sample(self) -> int:
        """
        Sample expert index proportional to current weights.
        If one expert dominates (W > 0.6) → deterministic argmax.
        Otherwise → stochastic sample.
        """
        if np.max(self.W) > 0.6:
            return int(np.argmax(self.W))
        return int(np.random.choice(self.n_experts, p=self.W))

    def update(
        self,
        expert_idx   : int,
        ce_score     : float,
        evicted_time : int,
        current_time : int,
    ):
        """
        Penalize expert_idx using LeCaR Algorithm 2 + CE score.

        Parameters
        ----------
        expert_idx   : which expert to penalize (0, 1, or 2)
        ce_score     : relevance of evicted doc ∈ [0, 1]
                       1.0 = highly relevant = bad eviction
                       0.0 = irrelevant = eviction was fine
        evicted_time : query index when doc was evicted
        current_time : current query index
        """
        # time doc spent in history (like LeCaR's t)
        t = max(0, current_time - evicted_time)

        # discount: older eviction = less blame (like LeCaR's d^t)
        decay = self.discount_rate ** t

        # graded reward: negative = penalty
        # ce_score scales the penalty by doc relevance
        reward = -(float(ce_score) * decay)

        # multiplicative weight update (LeCaR Algorithm 2)
        self.W[expert_idx] *= np.exp(self.learning_rate * reward)

        # normalize
        total = self.W.sum()
        if total > 0:
            self.W /= total

        # clamp to [0.01, 0.99] — LeCaR prevents silencing any expert
        self.W = np.clip(self.W, 0.01, 0.99)
        self.W /= self.W.sum()

        # track cumulative regret (absolute value, for plotting)
        self.regret[expert_idx] += abs(reward)

        self.n_updates += 1

    def get_weights(self) -> np.ndarray:
        return self.W.copy()

    def get_regret(self) -> np.ndarray:
        return self.regret.copy()

    def __repr__(self):
        return (f"HedgeWeights("
                f"W={np.round(self.W, 3)}, "
                f"regret={np.round(self.regret, 3)}, "
                f"updates={self.n_updates})")