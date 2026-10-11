"""Shared download helpers: checksum verify + graceful missing-file errors."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tsad.data.base import DatasetMissingError


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def require(path: Path | str, instructions: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise DatasetMissingError(f"Missing {p}.\n{instructions}")
    return p


def verify_checksum(path: Path, expected: str | None) -> bool:
    if expected is None:
        return True
    return sha256(path) == expected
