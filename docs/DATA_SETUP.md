# Dataset setup: real URLs + access notes. Synthetic generator is the offline fallback.

## Public (download scripts print the same)
- SMD (28 machines, 38ch): https://github.com/NetManAIOps/OmniAnomaly → `data/smd/machine-<i>-{train,test,test_label}.npy`
- SMAP/MSL: https://github.com/khundrakpam/telemanom → `data/smap_msl/{train,test,test_label}/*.npy`
- UCR Anomaly Archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ → convert to `data/ucr/<group>__<name>.npz` (arrays `series`, `labels`)

## Access-required (loaders raise DatasetMissingError with these steps, never fake data)
- SWaT: request at https://itrust.sutd.edu.sg/itrust-labs_datasets/ → `data/swat/`
- Yahoo S5: Yahoo Webscope L6 → `data/yahoo/` (A1Benchmark CSVs)

## Synthetic (always available)
```
python scripts/make_synthetic.py --config configs/smoke.yaml
python scripts/data_summary.py --config configs/smoke.yaml   # reports/data_summary.{csv,md,png}
```
Controllable shift: mean / var_scale / freq / phase / amp per domain (`harmful_last=true` adds an
unrelated domain for the E4 negative-transfer study). 6 anomaly types with GT labels + channel masks.

## Splits / normalization discipline
Time-ordered series-level split (train→val→test), windowing AFTER splitting (no overlap leakage),
per-domain z-score fit on train only, constant channels (std→1) and NaNs handled.
