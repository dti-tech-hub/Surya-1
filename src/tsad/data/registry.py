"""Domain registry: stats + transfer protocols (unlabeled-target discipline)."""
from __future__ import annotations
import numpy as np
from tsad.data.base import DomainData


def domain_stats(d: DomainData) -> dict:
    def _s(x: np.ndarray) -> dict:
        return {"mean": float(np.mean(x)), "std": float(np.std(x)),
                "len": int(len(x)), "channels": int(x.shape[1])}

    return {
        "domain": d.domain_id,
        "train": _s(d.train_series),
        "val": _s(d.val_series),
        "test": _s(d.test_series),
        "anomaly_ratio_test": float(d.test_labels.mean()) if len(d.test_labels) else 0.0,
    }


class DomainRegistry:
    """Holds domains; exposes protocol views. Target-train is UNLABELED by convention (labels unused)."""

    def __init__(self, domains: dict[str, DomainData]):
        assert domains, "empty domain dict"
        self.domains = domains

    def ids(self) -> list[str]:
        return sorted(self.domains)

    def single(self, domain_id: str) -> dict[str, DomainData]:
        return {domain_id: self.domains[domain_id]}

    def source_target(self, source: str | list[str], target: str) -> tuple[dict, dict]:
        srcs = [source] if isinstance(source, str) else list(source)
        return ({s: self.domains[s] for s in srcs}, {target: self.domains[target]})

    def lodo(self) -> list[tuple[dict, dict]]:
        """Leave-one-domain-out: [(sources, {held_out}), ...]."""
        return [({i: d for i, d in self.domains.items() if i != h}, {h: self.domains[h]})
                for h in self.ids()]

    def multi_source(self, target: str) -> tuple[dict, dict]:
        return ({i: d for i in self.ids() if i != target}, {target: self.domains[target]})

    def summary(self) -> list[dict]:
        return [domain_stats(d) for d in self.domains.values()]
