"""Phase-1 tests: windows, leakage, splits, augmentation, synthetic masks, missing-data grace."""
import numpy as np
import pytest
from tsad.data.windows import sliding_windows, window_label
from tsad.data.splits import split_series, assert_window_disjointness
from tsad.data.normalize import PerDomainNormalizer
from tsad.data.augment import AugmentPipeline, AUGMENTS
from tsad.data.synthetic import SyntheticDataset
from tsad.data.base import DatasetMissingError


def test_window_correctness():
    x = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
    w = sliding_windows(x, window=4, stride=3)
    assert w.shape == (3, 4, 2)  # starts 0,3,6
    assert np.array_equal(w[1], x[3:7])
    with pytest.raises(ValueError):
        sliding_windows(np.ones((2, 2), np.float32), window=4, stride=1)


def test_window_labels():
    y = np.zeros(10, dtype=np.int64)
    y[5] = 1
    wl = window_label(y, window=4, stride=3)
    assert wl.tolist() == [0, 1, 0]  # windows [0:4],[3:7],[6:10]; label at idx 5


def test_split_disjointness_and_order():
    x = np.random.RandomState(0).randn(100, 2).astype(np.float32)
    y = np.zeros(100, np.int64)
    (tr_x, _), (va_x, _), (te_x, _) = split_series(x, y, 0.6, 0.2)
    assert len(tr_x) == 60 and len(va_x) == 20 and len(te_x) == 20
    assert tr_x[0, 0] == x[0, 0] and te_x[-1, 0] == x[-1, 0]  # order preserved
    assert_window_disjointness(100, 60, 80, window=16, stride=4)


def test_normalizer_fit_train_only_and_const_channel():
    tr = np.ones((50, 2), np.float32)
    tr[:, 0] = np.random.RandomState(1).randn(50)  # varying
    # ch1 constant -> std forced to 1, no div-by-zero; NaN handled
    tr[0, 0] = np.nan
    n = PerDomainNormalizer().fit(tr)
    assert n.std_[1] == 1.0
    te = np.full((10, 2), 5.0, np.float32)
    out = n.transform(te)
    assert np.isfinite(out).all()
    # leakage guard: transform before fit raises
    with pytest.raises(AssertionError):
        PerDomainNormalizer().transform(te)


def test_augment_shape_dtype():
    x = np.random.RandomState(2).randn(32, 3).astype(np.float32)
    for name in AUGMENTS:
        fn = AugmentPipeline([name], seed=0)
        y = fn(x)
        assert y.shape == x.shape and y.dtype == np.float32, name
    two = AugmentPipeline(["jitter", "scaling"], seed=0).two_views(x)
    assert len(two) == 2 and two[0].shape == x.shape


def test_synthetic_masks_and_types():
    ds = SyntheticDataset(n_domains=2, length=400, n_channels=3, anomaly_ratio=0.05, seed=0)
    domains = ds.load()
    assert len(domains) == 2
    for d in domains.values():
        assert d.test_channel_mask is not None
        assert d.test_channel_mask.shape == d.test_series.shape
        # every labelled point has >=1 masked channel; every masked channel is labelled
        lab = d.test_labels.astype(bool)
        m = d.test_channel_mask.astype(bool)
        assert (m[lab].sum(axis=1) >= 1).all()
        assert (~lab[m.any(axis=1)]).sum() == 0


def test_missing_datasets_skip_gracefully():
    from tsad.data.datasets import SWaTDataset, YahooDataset, SMDDataset

    for cls in (SWaTDataset, YahooDataset):
        with pytest.raises(DatasetMissingError, match="access-required|Missing"):
            cls().load()
    with pytest.raises(DatasetMissingError):
        SMDDataset(root="data/does-not-exist-xyz").load()
