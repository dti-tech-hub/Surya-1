"""Transfer metrics: source-only vs adapted vs target-only; negative-transfer flag."""

from __future__ import annotations


def transfer_report(
    source_only_f1: float, adapted_f1: float, target_only_f1: float | None = None
) -> dict:
    """gain = adapted - source_only. negative_transfer = adapted < source_only (strictly worse)."""
    gain = float(adapted_f1 - source_only_f1)
    out = {
        "source_only_f1": float(source_only_f1),
        "adapted_f1": float(adapted_f1),
        "transfer_gain": gain,
        "negative_transfer": bool(gain < 0),
    }
    if target_only_f1 is not None:
        out["target_only_f1"] = float(target_only_f1)
        out["gap_to_target_only"] = float(target_only_f1 - adapted_f1)
    return out
