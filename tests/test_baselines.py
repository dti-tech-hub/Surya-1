"""Phase-2 baseline tests: interface contract + tiny synthetic run for every baseline."""
import numpy as np
from tsad.data.synthetic import SyntheticDataset
from tsad.data.normalize import PerDomainNormalizer
from tsad.baselines import all_baselines
from tsad.eval.metrics import strict_metrics


def _tiny():
    ds = SyntheticDataset(n_domains=1, length=256, n_channels=2, anomaly_ratio=0.05, seed=0).load()
    d = ds["domain_0"]
    n = PerDomainNormalizer().fit(d.train_series)
    return n.transform(d.train_series), n.transform(d.test_series), d.test_labels


def test_baseline_contract_and_finite_scores():
    tr, te, _ = _tiny()
    for bl in all_baselines(window=32, epochs=1, seed=0):
        bl.fit(tr)
        s = bl.score(te)
        assert s.shape == (len(te),), bl.name
        assert np.isfinite(s).all(), bl.name
        assert s.dtype == np.float64, bl.name


def test_baselines_end_to_end_metrics():
    tr, te, y = _tiny()
    for bl in all_baselines(window=32, epochs=1, seed=0):
        bl.fit(tr)
        m = strict_metrics(y, bl.score(te), float(np.quantile(bl.score(te), 0.9)))
        assert 0.0 <= m["f1"] <= 1.0, bl.name
        assert 0.0 <= m["auc_roc"] <= 1.0 or np.isnan(m["auc_roc"]), bl.name


def test_expected_baseline_count():
    assert len(all_baselines()) == 8  # zscore/iforest/ocsvm/pca/arima + 3 AEs
