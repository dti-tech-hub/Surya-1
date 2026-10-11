# Results (strict + PA — PA inflates, compare strict_*)

| auc_roc | condition | dissimilarity | domain_weights | harmful_kind | mmd_after | mmd_before | seed | strict_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9293 | uniform-mixed | 0.0 | [0.5, 0.5] | clean | 0.0271 | 0.0137 | 0 | 0.1538 |
| 0.7713 | adaptive-full | 0.0 | [0.666, 0.334] | clean | 0.0154 | 0.0137 | 0 | 0.0 |
| 0.7374 | uniform-mixed | 0.0429 | [0.5, 0.5] | noise | 0.0549 | 0.011 | 0 | 0.0426 |
| 0.9076 | adaptive-full | 0.0429 | [0.775, 0.225] | noise | 0.0777 | 0.011 | 0 | 0.1875 |
