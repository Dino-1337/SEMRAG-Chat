# ranker.py
from typing import List, Tuple, Dict
import numpy as np
from src.chunking.semantic_chunker import Chunk


def _minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if len(x) == 0:
        return x
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + eps)


class ResultRanker:
    def __init__(
        self,
        local_weight: float = 0.6,
        global_weight: float = 0.4,
        intersection_boost: float = 0.15,
        strong_frac: float = 0.70     # intersection requires ≥70% of max scores
    ):
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.intersection_boost = intersection_boost
        self.strong_frac = strong_frac

    def _normalize_map(self, scores: Dict[int, Tuple[Chunk, float]]):
        if not scores:
            return {}
        vals = np.array([s for (_, s) in scores.values()])
        norm_vals = _minmax(vals)
        out = {}
        for (cid, (chunk, _)), nv in zip(scores.items(), norm_vals):
            out[cid] = (chunk, float(nv))
        return out

    def combine_results(self,
                        local_results: List[Tuple[Chunk, float]],
                        global_results: List[Tuple[Chunk, float]],
                        strategy: str = "weighted"):

        # -------- Convert result lists to maps --------
        local_map = {c.chunk_id: (c, s) for c, s in local_results}
        global_map = {c.chunk_id: (c, s) for c, s in global_results}

        # -------- Normalize both sources independently --------
        local_norm = self._normalize_map(local_map)
        global_norm = self._normalize_map(global_map)

        # Strategies
        if strategy == "union":
            merged = {}
            for cid, (c, s) in {**local_norm, **global_norm}.items():
                merged[cid] = (c, s)
            return sorted(merged.values(), key=lambda x: x[1], reverse=True)

        if strategy == "intersection":
            ids = set(local_norm.keys()) & set(global_norm.keys())
            results = []
            for cid in ids:
                c = local_norm[cid][0]
                score = max(local_norm[cid][1], global_norm[cid][1])
                results.append((c, float(score)))
            return sorted(results, key=lambda x: x[1], reverse=True)

        # -------- Default: weighted fusion --------
        combined = {}

        # Local contribution
        for cid, (chunk, score) in local_norm.items():
            combined[cid] = (chunk, score * self.local_weight)

        # Global contribution
        for cid, (chunk, score) in global_norm.items():
            if cid in combined:
                combined[cid] = (chunk, combined[cid][1] + score * self.global_weight)
            else:
                combined[cid] = (chunk, score * self.global_weight)

        # Intersection boost (only if BOTH signals are strong)
        local_max = max(local_norm.values(), key=lambda x: x[1])[1] if local_norm else 1
        global_max = max(global_norm.values(), key=lambda x: x[1])[1] if global_norm else 1

        for cid, (chunk, score) in list(combined.items()):
            if cid in local_norm and cid in global_norm:
                if (local_norm[cid][1] >= self.strong_frac * local_max and
                    global_norm[cid][1] >= self.strong_frac * global_max):
                    combined[cid] = (chunk, score + self.intersection_boost)

        # Final min-max normalization
        final_scores = np.array([s for (_, s) in combined.values()])
        final_norm = _minmax(final_scores)

        fused = []
        for (cid, (chunk, _)), nv in zip(combined.items(), final_norm):
            fused.append((chunk, float(nv)))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused
