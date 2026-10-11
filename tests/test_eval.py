"""Phase-2 tests: metrics vs hand-computed + sklearn, PA behaviour, thresholds, transfer flag."""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tsad.eval.metrics import strict_metrics, pa_metrics, event_wise_f1, range_based_f1
from tsad.eval.point_adjust import point_adjust
from tsad.eval.threshold import select_threshold, best_f1_threshold, percentile_threshold
from tsad.eval.bootstrap import bootstrap_ci, aggregate_seeds
from tsad.eval.transfer import transfer_report


def test_strict_matches_sklearn_and_handcomputed():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([0.1, 0.6, 0.4, 0.9, 0.3, 0.8])
    m = strict_metrics(y, s, 0.5)  # pred = 0,1,0,1,0,1
    pred = (s >= 0.5).astype(int)
    assert m["precision"] == precision_score(y, pred)
    assert m["recall"] == recall_score(y, pred)
    assert m["f1"] == f1_score(y, pred)
    assert abs(m["precision"] - 2 / 3) < 1e-9 and abs(m["recall"] - 2 / 3) < 1e-9  # TP=2 FP=1 FN=1


def test_pa_inflates_on_partial_event():
    y = np.zeros(20, dtype=int)
    y[5:15] = 1  # one 10-pt event, single detection at idx 5
    s = np.zeros(20)
    s[5] = 0.99
    strict = strict_metrics(y, s, 0.5)
    pa = pa_metrics(y, s, 0.5)
    assert strict["recall"] == 0.1  # 1/10 points
    assert pa["pa_recall"] == 1.0  # whole event counts -> inflation demonstrated
    assert pa["pa_f1"] > strict["f1"]
    assert "warning" in pa


def test_pa_no_false_inflation_on_clean():
    y = np.zeros(10, dtype=int)
    s = np.zeros(10)
    assert point_adjust(s, y, 0.5).sum() == 0


def test_event_and_range_metrics():
    y = np.zeros(30, dtype=int)
    y[5:10] = 1
    y[20:25] = 1
    s = np.zeros(30)
    s[5] = 0.9  # event 1 hit, event 2 missed
    e = event_wise_f1(y, s, 0.5)
    assert e["event_recall"] == 0.5 and e["event_precision"] == 1.0
    r = range_based_f1(y, s, 0.5)
    assert 0.0 < r["range_recall"] < 1.0


def test_threshold_val_only():
    va_s = np.array([0.1, 0.2, 0.15, 0.9, 0.85])
    va_y = np.array([0, 0, 0, 1, 1])
    t = best_f1_threshold(va_y, va_s)
    assert 0.2 < t <= 0.85  # separates normal from anomalous val
    assert select_threshold("percentile", va_s[:3], percentile=50) == percentile_threshold(va_s[:3], 50)
    assert select_threshold("fixed", va_s, fixed=0.7) == 0.7
    # POT returns finite threshold from normal scores only (no labels)
    assert np.isfinite(select_threshold("pot", np.random.RandomState(0).rand(500)))


def test_bootstrap_and_aggregate():
    y = np.array([0, 1, 0, 1] * 25)
    p = np.array([0, 1, 0, 0] * 25)
    ci = bootstrap_ci(y, p, n_boot=50, seed=0)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    agg = aggregate_seeds([{"f1": 0.5}, {"f1": 0.7}])
    assert abs(agg["f1"] - 0.6) < 1e-9 and agg["n_seeds"] == 2


def test_negative_transfer_indicator():
    assert transfer_report(0.6, 0.5)["negative_transfer"] is True
    assert transfer_report(0.6, 0.7)["negative_transfer"] is False
    r = transfer_report(0.6, 0.7, 0.8)
    assert abs(r["gap_to_target_only"] - 0.1) < 1e-9
