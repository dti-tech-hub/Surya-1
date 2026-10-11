"""Baselines package exports. Contrastive no-adapt model lands in Phase 3 (O1)."""
from tsad.baselines.base import Baseline
from tsad.baselines.classical import ZScoreBaseline, IsolationForestBaseline, OCSVMBaseline, PCABaseline
from tsad.baselines.arima import ARIMABaseline
from tsad.baselines.deep import AEModel

CLASSICAL = [ZScoreBaseline, IsolationForestBaseline, OCSVMBaseline, PCABaseline]


def all_baselines(window: int = 32, epochs: int = 3, seed: int = 0) -> list[Baseline]:
    """Every Phase-2 baseline, constructed with smoke-friendly defaults."""
    return [* [c() for c in CLASSICAL], ARIMABaseline(),
            AEModel("lstm", window=window, epochs=epochs, seed=seed),
            AEModel("conv", window=window, epochs=epochs, seed=seed),
            AEModel("trans", window=window, epochs=epochs, seed=seed)]


__all__ = ["Baseline", "ZScoreBaseline", "IsolationForestBaseline", "OCSVMBaseline",
           "PCABaseline", "ARIMABaseline", "AEModel", "all_baselines"]
