"""Phase-4 tests: gate range/sparsity, weights simplex, GRL sign, MMD/CORAL/DANN, preserve, monitor, TTA."""
import numpy as np
import torch
from tsad.losses.mmd import mmd2
from tsad.losses.coral import coral
from tsad.losses.dann import grad_reverse, dann_lambda, DomainDiscriminator
from tsad.alignment.gate import FeatureGate
from tsad.alignment.domain_weights import DomainWeighter
from tsad.alignment.preserve import compactness, pseudo_margin, no_collapse
from tsad.alignment.adaptive import AdaptiveAligner
from tsad.alignment.monitor import NegTransferMonitor
from tsad.alignment.tta import adabn_adapt
from tsad.models.encoders import make_encoder


def _gauss(n, shift, d=4, seed=0):
    return torch.tensor(np.random.default_rng(seed).normal(shift, 1.0, (n, d)), dtype=torch.float32)


def test_gate_range_and_sparsity():
    for kind in ("sigmoid", "hard_concrete"):
        g = FeatureGate(6, kind)
        z = torch.randn(10, 6)
        out = g(z)
        assert out.shape == (10, 6)
        assert ((g.probs >= 0) & (g.probs <= 1)).all()
        assert torch.isfinite(g.sparsity_loss())
    g = FeatureGate(6)
    with torch.no_grad():
        g.logits.fill_(5.0)  # saturated open
    assert float(g.sparsity_loss()) > 0.9  # ~mean 1.0 -> high (no sparsity)


def test_weights_simplex_downweights_far_source():
    w = DomainWeighter("mmd", tau=0.5)
    tgt = _gauss(64, 0.0)
    near, far = _gauss(64, 0.2, seed=1), _gauss(64, 5.0, seed=2)
    wt = w([near, far], tgt)
    assert abs(float(wt.sum()) - 1.0) < 1e-5
    assert wt[0] > wt[1]  # near source up-weighted
    assert w([near], tgt).item() == 1.0  # single source


def test_grl_sign_and_schedule():
    x = torch.ones(4, 3, requires_grad=True)
    grad_reverse(x, 2.0).sum().backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones_like(x))  # sign flip x lambda
    assert dann_lambda(0.0) == 0.0 and abs(dann_lambda(1.0) - 1.0) < 1e-3
    d = DomainDiscriminator(4)
    assert d(torch.randn(5, 4)).shape == (5,)


def test_mmd_coral_separate_and_alignable():
    A, B = _gauss(100, 0.0), _gauss(100, 2.0)
    m0, c0 = float(mmd2(A, B)), float(coral(A, B))
    assert m0 > 0.05 and c0 > 0.0
    B_aligned = B - 2.0  # oracle mean alignment
    assert float(mmd2(A, B_aligned)) < m0  # alignment reduces MMD
    assert float(coral(A, B_aligned)) < c0


def test_preserve_terms():
    Zn = torch.randn(32, 4) * 0.1
    assert float(compactness(Zn)) < float(compactness(torch.randn(32, 4) * 5))
    assert float(pseudo_margin(torch.tensor([0.0]), torch.tensor([2.0]))) == 0.0  # margin kept
    assert float(pseudo_margin(torch.tensor([0.0]), torch.tensor([0.2]))) > 0  # violated
    assert float(no_collapse(torch.zeros(16, 4))) > 0  # collapsed
    assert float(no_collapse(torch.randn(16, 4))) == 0.0  # healthy variance


def test_adaptive_forward_and_monitor_rollback():
    al = AdaptiveAligner(4, method="mmd", use_gate=True, use_weights=True)
    S = [_gauss(32, 0.0, seed=i) for i in range(2)]
    T = _gauss(32, 0.5)
    out = al(S, T, epoch=0, max_epochs=4)
    assert abs(float(out["weights"].sum()) - 1.0) < 1e-4
    assert 0.0 <= out["alpha_t"] <= al.alpha + 1e-6  # warm-up respected
    assert torch.isfinite(out["total_align"])
    al_none = AdaptiveAligner(4, method="none")
    assert al_none(S, T)["alpha_t"] == 0.0
    mon = NegTransferMonitor(patience=1)
    enc = make_encoder("tcn", 2, d=4)
    mon.update(enc, 1.0, 1.0)
    info = mon.update(enc, 2.0, 2.0)  # worse -> alpha cut (+rollback)
    assert info["alpha_mult"] == 0.5 and info["rolled_back"] is True


def test_tta_updates_bn_stats():
    enc = make_encoder("tcn", 2, d=4)
    enc.train()
    before = [b.running_mean.clone() for b in enc.modules()
              if isinstance(b, torch.nn.BatchNorm1d)]
    X = (np.random.default_rng(0).normal(5, 1, (64, 16, 2))).astype(np.float32)  # shifted target
    adabn_adapt(enc, X)
    after = [b.running_mean for b in enc.modules() if isinstance(b, torch.nn.BatchNorm1d)]
    assert any(not torch.allclose(a, b_) for a, b_ in zip(before, after))
    assert enc.training is False  # left in eval mode
