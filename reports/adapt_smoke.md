# Results (strict + PA — PA inflates, compare strict_*)

| auc_roc | condition | domain_weights | mmd_after | mmd_before | strict_f1 | t_adapted_f1 | t_negative_transfer | t_transfer_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8009 | source-only | [1.0, 1.0] | 0.0617 | 0.011 | 0.0645 |  |  |  |
| 0.7965 | uniform-clean | [1.0] | 0.014 | 0.0101 | 0.0 | 0.0 | True | -0.0645 |
| 0.9084 | uniform-mixed | [0.5, 0.5] | 0.0758 | 0.011 | 0.0 | 0.0 | True | -0.0645 |
| 0.9069 | adaptive-full | [0.775, 0.225] | 0.0648 | 0.011 | 0.25 | 0.25 | False | 0.1855 |
