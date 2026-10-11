"""5. Cross-domain explanation: how normal differs across domains for this anomaly."""

from __future__ import annotations

import numpy as np


def cross_domain_part(
    z: np.ndarray,
    proto_by_domain: dict[str, np.ndarray],
    gate_probs: np.ndarray | None,
    stats_by_domain: dict[str, dict] | None = None,
) -> dict:
    """z: (d,) anomaly embedding. proto_by_domain: domain -> prototypes (k, d)."""
    dists = {
        dom: float(np.linalg.norm(P - z[None, :], axis=1).min())
        for dom, P in proto_by_domain.items()
    }
    closest = min(dists, key=lambda k: dists[k])
    out: dict = {
        "dist_to_each_domains_normal": dists,
        "closest_normal": closest,
        "verdict": (
            "closest to " + closest + " normal; far from others " "=> domain-specific deviation"
            if len(set(np.round(list(dists.values()), 3))) > 1
            else "equidistant => domain-agnostic deviation"
        ),
    }
    if gate_probs is not None:
        g = np.asarray(gate_probs, dtype=float)
        out["invariant_dims"] = [int(i) for i in np.where(g > 0.5)[0]]
        out["specific_dims"] = [int(i) for i in np.where(g <= 0.5)[0]]
        out["frac_invariant"] = float((g > 0.5).mean())
    if stats_by_domain is not None:
        out["domain_stats"] = stats_by_domain
    return out
