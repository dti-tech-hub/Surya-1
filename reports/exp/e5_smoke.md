# Results (strict + PA — PA inflates, compare strict_*)

| ablation | auc_roc | condition | domain | domain_weights | event_f1 | event_precision | event_recall | mmd_after | mmd_before | model | pa_f1 | pa_precision | pa_recall | proto_note | range_f1 | range_precision | range_recall | seed | strict_auc_pr | strict_auc_roc | strict_f1 | strict_precision | strict_recall | strict_threshold | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enc-tcn |  |  | domain_0 |  | 1.0 | 1.0 | 1.0 |  |  | contrast-tcn | 0.0745 | 0.0387 | 1.0 | clean-normal-windows | 0.6811 | 0.5165 | 1.0 | 0 | 0.1209 | 0.8095 | 0.0745 | 0.0387 | 1.0 | 1.7168 | 1.7168 |
| enc-transformer |  |  | domain_0 |  | 1.0 | 1.0 | 1.0 |  |  | contrast-transformer | 0.07 | 0.0363 | 1.0 | clean-normal-windows | 0.6826 | 0.5181 | 1.0 | 0 | 0.0343 | 0.2338 | 0.07 | 0.0363 | 1.0 | 1.4753 | 1.4753 |
| align-full | 0.9661 | align-full |  | [0.775, 0.225] |  |  |  | 0.0632 | 0.011 |  |  |  |  |  |  |  |  | 0 |  |  | 0.2373 |  |  |  |  |
| align-no-gate | 0.8023 | align-no-gate |  | [0.775, 0.225] |  |  |  | 0.0943 | 0.011 |  |  |  |  |  |  |  |  | 0 |  |  | 0.0 |  |  |  |  |
