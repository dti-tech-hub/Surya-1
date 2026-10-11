"""Real-dataset loaders. Missing files -> DatasetMissingError with download instructions (never fake data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsad.data.base import BaseTSADDataset, DatasetMissingError, DomainData
from tsad.data.splits import split_series


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise DatasetMissingError(f"Missing {path}")
    return np.load(path)


class SMDDataset(BaseTSADDataset):
    """SMD: 28 machines x 38 channels. Domain = machine. Expects data/smd/machine-<id>_{train,test,test_label}.npy."""

    name = "smd"
    instructions = (
        "SMD (OmniAnomaly): download from https://github.com/NetManAIOps/OmniAnomaly "
        "-> place machine-*-{train,test,test_label}.npy under data/smd/."
    )

    def __init__(self, root: str | Path = "data/smd", machines: list[int] | None = None):
        self.root = Path(root)
        self.machines = machines or list(range(1, 29))

    def load(self) -> dict[str, DomainData]:
        if not self.root.exists():
            raise DatasetMissingError(f"Missing {self.root}.\n{self.instructions}")
        out = {}
        for m in self.machines:
            tr_p, te_p, lb_p = (
                self.root / f"machine-{m}-{k}.npy" for k in ("train", "test", "test_label")
            )
            if not (tr_p.exists() and te_p.exists() and lb_p.exists()):
                continue  # skip machines not downloaded; summary reports which are present
            tr, te, lb = _load_npy(tr_p), _load_npy(te_p), _load_npy(lb_p).astype(np.int64).ravel()
            (tr_x, _tr_y), (va_x, _va_y), _ = split_series(tr, np.zeros(len(tr), np.int64))
            # SMD train is assumed normal; carve val from train tail. Test = full labelled test series.
            out[f"machine-{m}"] = DomainData(
                f"machine-{m}",
                tr_x,
                va_x,
                te,
                np.zeros(len(tr_x), np.int64),
                np.zeros(len(va_x), np.int64),
                lb[: len(te)],
            )
        if not out:
            raise DatasetMissingError(f"No SMD files under {self.root}.\n{self.instructions}")
        return out


class SMAPMSLDataset(BaseTSADDataset):
    """SMAP/MSL (NASA): domain = channel/entity file. Expects data/smap_msl/{train,test,test_label}/*.npy."""

    name = "smap_msl"
    instructions = (
        "SMAP/MSL: download from https://github.com/khundrakpam/telemanom "
        "(nasa-annthyroid style mirrors) -> data/smap_msl/{train,test,test_label}/*.npy"
    )

    def __init__(self, root: str | Path = "data/smap_msl"):
        self.root = Path(root)

    def load(self) -> dict[str, DomainData]:
        te_d = self.root / "test"
        if not te_d.exists():
            raise DatasetMissingError(f"Missing {te_d}.\n{self.instructions}")
        out = {}
        for te_p in sorted(te_d.glob("*.npy")):
            eid = te_p.stem
            tr_p, lb_p = self.root / "train" / f"{eid}.npy", self.root / "test_label" / f"{eid}.npy"
            if not (tr_p.exists() and lb_p.exists()):
                continue
            tr, te = _load_npy(tr_p), _load_npy(te_p)
            lb = _load_npy(lb_p).astype(np.int64).ravel()[: len(te)]
            tr = tr.reshape(len(tr), -1)
            te = te.reshape(len(te), -1)
            (tr_x, _), (va_x, _), _ = split_series(tr, np.zeros(len(tr), np.int64))
            out[eid] = DomainData(
                eid,
                tr_x,
                va_x,
                te,
                np.zeros(len(tr_x), np.int64),
                np.zeros(len(va_x), np.int64),
                lb,
            )
        if not out:
            raise DatasetMissingError(f"No entity files under {self.root}.\n{self.instructions}")
        return out


class UCRDataset(BaseTSADDataset):
    """UCR Anomaly Archive (univariate). Domain = dataset group. Expects data/ucr/*.npz (T, label)."""

    name = "ucr"
    instructions = (
        "UCR Anomaly Archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ "
        "-> convert each series to data/ucr/<group>__<name>.npz with arrays series, labels"
    )

    def __init__(self, root: str | Path = "data/ucr"):
        self.root = Path(root)

    def load(self) -> dict[str, DomainData]:
        if not self.root.exists():
            raise DatasetMissingError(f"Missing {self.root}.\n{self.instructions}")
        files = sorted(self.root.glob("*.npz"))
        if not files:
            raise DatasetMissingError(f"No .npz under {self.root}.\n{self.instructions}")
        out = {}
        for f in files:
            z = np.load(f)
            x = np.asarray(z["series"], dtype=np.float32).reshape(-1, 1)
            y = np.asarray(z["labels"], dtype=np.int64).ravel()
            (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_series(x, y)
            out[f.stem] = DomainData(f.stem, tr_x, va_x, te_x, tr_y, va_y, te_y)
        return out


class SWaTDataset(BaseTSADDataset):
    """SWaT (access-required). Graceful skip with instructions."""

    name = "swat"
    access_required = True
    instructions = (
        "SWaT: request access at https://itrust.sutd.edu.sg/itrust-labs_datasets/ "
        "-> place SWaT_Dataset_Normal_v1.xlsx / Attack_v0.xlsx under data/swat/."
    )

    def __init__(self, root: str | Path = "data/swat"):
        self.root = Path(root)

    def load(self) -> dict[str, DomainData]:
        raise DatasetMissingError(
            f"SWaT is access-required; no files bundled.\n{self.instructions}"
        )


class YahooDataset(BaseTSADDataset):
    """Yahoo S5 (access-required). Graceful skip with instructions."""

    name = "yahoo"
    access_required = True
    instructions = (
        "Yahoo S5: https://webscope.sandbox.yahoo.com (L6 - Time Series Anomaly Detection) "
        "-> place A1Benchmark CSVs under data/yahoo/."
    )

    def __init__(self, root: str | Path = "data/yahoo"):
        self.root = Path(root)

    def load(self) -> dict[str, DomainData]:
        raise DatasetMissingError(
            f"Yahoo S5 is access-required; no files bundled.\n{self.instructions}"
        )
