"""Phase-5 tests: attribution shapes, fidelity on known-GT toy, localization, counterfactual."""

import numpy as np
from torch import nn

from tsad.explain import metrics as em
from tsad.explain.attribution import SoftProtoScore, integrated_gradients, saliency, smoothgrad
from tsad.explain.base import topk_channels
from tsad.explain.counterfactual import counterfactual
from tsad.models.encoders import make_encoder


class LinScore(nn.Module):
    """Toy score with KNOWN ground truth: score = sum over channels 0..1 only."""

    def forward(self, x):  # (B, T, C)
        return (x[:, :, :2].abs()).sum((1, 2))


def test_attribution_shapes():
    enc = make_encoder("tcn", 3, d=4)
    proto = np.random.default_rng(0).normal(size=(2, 4))
    fn = SoftProtoScore(enc, proto)
    x = np.random.default_rng(1).normal(size=(16, 3)).astype(np.float32)
    for f in (saliency, smoothgrad):
        a = f(fn, x)
        assert a.shape == x.shape and np.isfinite(a).all()
    a = integrated_gradients(fn, x, steps=4)
    assert a.shape == x.shape and np.isfinite(a).all()


def test_fidelity_identifies_ground_truth():
    fn = LinScore()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 4)).astype(np.float32)
    x[:, :2] *= 3  # channels 0,1 dominate the score by construction
    base = np.zeros_like(x)
    a_good = integrated_gradients(fn, x, baseline=base, steps=8)
    a_rand = rng.normal(size=x.shape)
    assert em.deletion_auc(fn, x, a_good, base) > em.deletion_auc(fn, x, a_rand, base)
    ch = topk_channels(a_good, k=2)
    assert {c["channel"] for c in ch} == {0, 1}  # exact GT channels recovered
    sc = em.sufficiency_comprehensiveness(fn, x, a_good, base)
    assert sc["comprehensiveness"] > 0.5  # removing top feats collapses score


def test_localization_and_stability():
    attr = np.zeros((10, 3))
    attr[4:6, 1] = 5.0
    mask = np.zeros((10, 3), dtype=int)
    mask[4:6, 1] = 1
    loc = em.localization(attr, mask)
    assert loc["precision_at_k"] == 1.0 and loc["auprc"] > 0.9
    # distinct background values so ranks are well-defined under tiny noise
    base = np.arange(30, dtype=float).reshape(10, 3) / 30.0
    base[4:6, 1] += 5.0
    rng = np.random.default_rng(0)
    noisy = [base + rng.normal(0, 0.001, base.shape) for _ in range(3)]
    st = em.stability(base, noisy, k=2)
    assert st["topk_jaccard_mean"] == 1.0 and st["rank_corr_mean"] > 0.9


def test_counterfactual_valid_sparse():
    fn = LinScore()
    x = np.ones((8, 3), dtype=np.float32) * 2.0  # score = 8*2*2 = 32
    cf = counterfactual(fn, x, threshold=5.0, steps=60, lo=-2.0, hi=2.0)
    assert cf["valid"] is True and cf["score_after"] < 5.0
    assert 0.0 < cf["frac_changed"] <= 1.0
    assert cf["top_channels"][:2] == [0, 1] or set(cf["top_channels"][:2]) <= {0, 1, 2}
