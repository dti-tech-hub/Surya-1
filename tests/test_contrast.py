"""Phase-3 tests: encoder shapes/attn, loss sanity + descent, prototype separation, mapping."""
import numpy as np
import torch
from tsad.models.encoders import make_encoder
from tsad.losses.contrastive import info_nce, ts2vec_loss, masked_mse
from tsad.models.prototype import PrototypeBank
from tsad.models.scoring import map_to_timestamps, smooth, hybrid


def test_encoder_shapes_and_attn():
    x = torch.randn(4, 16, 3)
    for kind in ("tcn", "transformer", "resnet"):
        o = make_encoder(kind, 3, d=8)(x)
        assert o["rep"].shape == (4, 8), kind
        assert o["tokens"].shape[0] == 4, kind
    o = make_encoder("transformer", 3, d=8)(x)
    assert o["attn"].shape == (4, 16)  # (B, T) exposed for explainability
    assert torch.allclose(o["attn"].sum(1), torch.ones(4), atol=1e-5)


def test_info_nce_sane_and_descends():
    torch.manual_seed(0)
    z1 = torch.randn(8, 16)
    z2 = z1 + 0.01 * torch.randn(8, 16)  # aligned views
    l_aligned = info_nce(z1, z2).item()
    l_random = info_nce(z1, torch.randn(8, 16)).item()
    assert np.isfinite(l_aligned) and l_aligned < l_random  # aligned views = lower loss
    # gradient descent reduces loss on fixed batch
    p = torch.nn.Parameter(torch.randn(8, 16))
    opt = torch.optim.SGD([p], lr=0.1)
    l0 = info_nce(p, z1).item()
    for _ in range(20):
        opt.zero_grad()
        info_nce(p, z1).backward()
        opt.step()
    assert info_nce(p, z1).item() < l0


def test_ts2vec_and_masked_finite():
    f1 = torch.randn(4, 16, 8)
    f2 = f1 + 0.05 * torch.randn(4, 16, 8)
    assert np.isfinite(ts2vec_loss(f1, f2).item())
    pr = torch.randn(2, 8, 3)
    assert np.isfinite(masked_mse(pr, torch.randn(2, 8, 3), torch.ones(2, 8)).item())


def test_prototype_separates_far_anomaly():
    rng = np.random.default_rng(0)
    Zn = rng.normal(0, 0.3, (50, 4))
    for method in ("kmeans", "knn", "mahalanobis"):
        b = PrototypeBank(k=4, method=method, seed=0).fit(Zn)
        sn = b.score(Zn).mean()
        sa = b.score(rng.normal(5, 0.3, (10, 4))).mean()
        assert sa > 3 * sn, (method, sn, sa)  # far anomalies score clearly higher


def test_mapping_and_smoothing():
    s = map_to_timestamps(np.array([1.0, 3.0]), [0, 2], T=6, window=4)
    assert s.shape == (6,)
    assert abs(s[2] - 2.0) < 1e-9  # overlap of windows 0 and 2 -> mean
    assert len(smooth(np.ones(10), 5)) == 10
    h = hybrid(np.array([0.0, 1.0, 2.0]), None)
    assert h[0] < h[2]  # fallback preserves order
