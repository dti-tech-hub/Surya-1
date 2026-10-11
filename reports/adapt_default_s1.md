# Results (strict + PA — PA inflates, compare strict_*)

| auc_roc | condition | domain_weights | mmd_after | mmd_before | strict_f1 | t_adapted_f1 | t_negative_transfer | t_transfer_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7579 | source-only | [1.0] | 0.1196 | 0.0259 | 0.2446 |  |  |  |
| 0.7607 | uniform-clean | [1.0] | 0.0231 | 0.027 | 0.2479 | 0.2479 | False | 0.0033 |
| 0.7832 | uniform-mixed | [0.5, 0.5] | 0.1426 | 0.0259 | 0.3019 | 0.3019 | False | 0.0573 |
| 0.7809 | adaptive-full | [0.67, 0.33] | 0.1602 | 0.0259 | 0.3059 | 0.3059 | False | 0.0613 |
