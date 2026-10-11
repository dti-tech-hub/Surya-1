"""Core metrics: strict P/R/F1, AUC-ROC, AUC-PR, PA variants, event-wise + range-based F1.

Conventions: scores (T,) higher = more anomalous; labels (T,) binary.
PA metrics are computed via point_adjust and ALWAYS flagged as inflated (see report.py).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tsad.eval.point_adjust import point_adjust


def _safe_auc(fn, y: np.ndarray, s: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(fn(y, s))
    except Exception:
        return float("nan")


def prf_from_pred(y: np.ndarray, pred: np.ndarray, zero_division: int = 0) -> dict:
    y, pred = np.asarray(y).ravel(), np.asarray(pred).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=zero_division)),
        "recall": float(recall_score(y, pred, zero_division=zero_division)),
        "f1": float(f1_score(y, pred, zero_division=zero_division)),
    }


def strict_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Point-wise (strict) metrics at threshold + threshold-free AUCs."""
    pred = (np.asarray(scores) >= threshold).astype(np.int64)
    m = prf_from_pred(np.asarray(y), pred)
    m.update(
        {
            "auc_roc": _safe_auc(roc_auc_score, np.asarray(y), np.asarray(scores)),
            "auc_pr": _safe_auc(average_precision_score, np.asarray(y), np.asarray(scores)),
            "threshold": float(threshold),
        }
    )
    return m


def pa_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Point-adjusted metrics. INFLATED by design — never report without strict."""
    adj = point_adjust(np.asarray(scores), np.asarray(y), threshold)
    m = prf_from_pred(np.asarray(y), adj)
    return {
        "pa_precision": m["precision"],
        "pa_recall": m["recall"],
        "pa_f1": m["f1"],
        "threshold": float(threshold),
        "warning": "PA inflates scores; see strict_* too",
    }


def _events(y: np.ndarray) -> list[tuple[int, int]]:
    y = np.asarray(y).ravel()
    ev, i, T = [], 0, len(y)
    while i < T:
        if y[i] == 1:
            j = i
            while j < T and y[j] == 1:
                j += 1
            ev.append((i, j))
            i = j
        else:
            i += 1
    return ev


def event_wise_f1(y: np.ndarray, scores: np.ndarray, threshold: float, overlap: int = 1) -> dict:
    """Event-wise F1: each GT event = 1 unit; detected if >=overlap predicted points inside.

    pred_events counted from thresholded preds; precision over pred events, recall over GT events.
    """
    y = np.asarray(y).ravel()
    pred = (np.asarray(scores) >= threshold).astype(np.int64)
    gt_ev, pr_ev = _events(y), _events(pred)
    if not gt_ev and not pr_ev:
        return {"event_precision": 1.0, "event_recall": 1.0, "event_f1": 1.0}
    hit = sum(1 for (a, b) in gt_ev if pred[a:b].sum() >= overlap)
    rec = hit / max(1, len(gt_ev))
    prec = sum(1 for (a, b) in pr_ev if y[a:b].sum() >= overlap) / max(1, len(pr_ev))
    f1 = 2 * prec * rec / max(1e-12, prec + rec) if (prec + rec) > 0 else 0.0
    return {"event_precision": float(prec), "event_recall": float(rec), "event_f1": float(f1)}


def range_based_f1(y: np.ndarray, scores: np.ndarray, threshold: float, alpha: float = 0.5) -> dict:
    """Simplified Tatbul-style range F1: overlap reward * (1-existence) blended with existence.

    Full Tatbul weights positional bias; here: recall_R = mean over GT events of
    alpha*exist + (1-alpha)*overlap_ratio; precision_S symmetric. Documented simplification.
    """
    y = np.asarray(y).ravel()
    pred = (np.asarray(scores) >= threshold).astype(np.int64)
    gt_ev, pr_ev = _events(y), _events(pred)
    if not gt_ev and not pr_ev:
        return {"range_precision": 1.0, "range_recall": 1.0, "range_f1": 1.0}

    def _score(ev_list, other):
        if not ev_list:
            return 0.0
        s = 0.0
        for a, b in ev_list:
            ov = float(other[a:b].sum()) / max(1, b - a)
            s += alpha * (1.0 if ov > 0 else 0.0) + (1 - alpha) * ov
        return s / len(ev_list)

    rec = _score(gt_ev, pred)
    prec = _score(pr_ev, y)
    f1 = 2 * prec * rec / max(1e-12, prec + rec) if (prec + rec) > 0 else 0.0
    return {"range_precision": float(prec), "range_recall": float(rec), "range_f1": float(f1)}


def full_report(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Strict + PA (flagged) + event-wise + range-based in one dict."""
    out = {f"strict_{k}": v for k, v in strict_metrics(y, scores, threshold).items()}
    out.update(pa_metrics(y, scores, threshold))
    out.update(event_wise_f1(y, scores, threshold))
    out.update(range_based_f1(y, scores, threshold))
    return out
