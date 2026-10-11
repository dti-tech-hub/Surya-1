"""Results table writer: CSV + markdown (strict and PA side-by-side, PA flagged)."""

from __future__ import annotations

import csv
from pathlib import Path


def write_table(rows: list[dict], out_prefix: str | Path) -> tuple[Path, Path]:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    csv_p = out_prefix.with_suffix(".csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    md = [
        "# Results (strict + PA — PA inflates, compare strict_*)",
        "",
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for r in rows:
        md.append(
            "| "
            + " | ".join(
                str(round(r.get(k, ""), 4) if isinstance(r.get(k), float) else r.get(k, ""))
                for k in keys
            )
            + " |"
        )
    md_p = out_prefix.with_suffix(".md")
    md_p.write_text("\n".join(md) + "\n")
    return csv_p, md_p
