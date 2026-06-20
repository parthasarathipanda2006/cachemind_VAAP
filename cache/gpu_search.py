# src/cache/gpu_search.py
"""
GPU-accelerated similarity search using PyTorch.
Replaces numpy matmul in run_trace.py hot path.
RTX A5000 can do 66k x 1024 matmul in ~0.5ms vs ~75ms on CPU.
"""

import numpy as np
import torch

# pin to GPU once at import
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[gpu_search] using device: {DEVICE}")


class GPUDocMatrix:
    """
    Holds doc embeddings on GPU.
    One matmul call per query = ~0.5ms on RTX A5000.
    """

    def __init__(self, doc_embs: np.ndarray, doc_id_list: list):
        self.doc_id_list = doc_id_list
        # move to GPU once — stays there for entire experiment
        self.matrix = torch.from_numpy(
            doc_embs.astype(np.float32)
        ).to(DEVICE)          # (D, dim)
        print(f"[gpu_search] doc matrix on GPU: {self.matrix.shape}")

    def top_k(
        self,
        query_norm : np.ndarray,
        k          : int   = 10,
        threshold  : float = 0.5,
    ) -> tuple[str, list]:
        """
        Returns (top1_doc_id, [(doc_id, sim), ...])
        top_k results above threshold, sorted descending.
        """
        q = torch.from_numpy(
            query_norm.astype(np.float32)
        ).to(DEVICE)                          # (dim,)

        sims = self.matrix @ q                # (D,)

        # top-1
        top1_idx = int(torch.argmax(sims).item())
        top1_doc = self.doc_id_list[top1_idx]

        # top-k
        k_actual  = min(k, len(self.doc_id_list))
        topk_vals, topk_idx = torch.topk(sims, k_actual)

        # back to CPU for downstream use
        topk_vals = topk_vals.cpu().numpy()
        topk_idx  = topk_idx.cpu().numpy()

        top_docs = [
            (self.doc_id_list[i], float(v))
            for i, v in zip(topk_idx, topk_vals)
            if float(v) >= threshold
        ]

        return top1_doc, top_docs