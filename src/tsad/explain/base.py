"""Explanation object schema (JSON-serializable dict) + shared helpers."""
from __future__ import annotations
import numpy as np


def topk_channels(attr: np.ndarray, k: int = 3) -> list[dict]:
    """attr: (T, C) -> ranked channels by mean |attr|."""
    ch = np.abs(np.asarray(attr, dtype=float)).mean(axis=0)
    order = np.argsort(-ch)[:k]
    return [{"channel": int(c), "importance": float(ch[c])} for c in order]


def topk_timesteps(attr: np.ndarray, k: int = 5) -> list[int]:
    t = np.abs(np.asarray(attr, dtype=float)).sum(axis=1)
    return [int(i) for i in np.argsort(-t)[:k]]


def new_explanation(window_id: int, t_start: int, t_end: int, score: float,
                    threshold: float) -> dict:
    return {"window_id": int(window_id), "t_start": int(t_start), "t_end": int(t_end),
            "score": float(score), "threshold": float(threshold), "flagged": bool(score >= threshold),
            "parts": {}}
