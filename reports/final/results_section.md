# Results (auto-drafted from actual runs — verify before submitting)

## Single-domain (e1_smoke): n=9, mean=0.541, max=0.727, min=0.106
## Single-domain (baselines_smoke): n=16, mean=0.510, max=0.727, min=0.111
## Single-domain (contrast_smoke): n=1, mean=0.074, max=0.074, min=0.074

## Transfer / negative transfer (adapt_smoke)
| condition | strict_F1 | AUC-ROC |
|---|---|---|
| source-only | 0.065 | 0.801 |
| uniform-clean | 0.000 | 0.796 |
| uniform-mixed | 0.000 | 0.908 |
| adaptive-full | 0.250 | 0.907 |

## Alignment comparison (e3_smoke)
| source-only | 0.111 | 0.804 |
| uniform-clean | 0.000 | 0.795 |
| uniform-mixed | 0.043 | 0.737 |
| adaptive-full | 0.188 | 0.908 |

## Negative-transfer study (e4_smoke)
| kind | condition | strict_F1 | dissimilarity |
|---|---|---|---|
| clean | uniform-mixed | 0.154 | 0.000 |
| clean | adaptive-full | 0.000 | 0.000 |
| noise | uniform-mixed | 0.043 | 0.043 |
| noise | adaptive-full | 0.188 | 0.043 |

## Explainability (_metrics): explainer fidelity (deletion-AUC)
- saliency: deletion-AUC 0.783, P@k 1.000
- ig: deletion-AUC 0.791, P@k 1.000
- smoothgrad: deletion-AUC 0.788, P@k 1.000
- saliency: deletion-AUC 0.788, P@k 1.000
- ig: deletion-AUC 0.792, P@k 1.000
- smoothgrad: deletion-AUC 0.791, P@k 1.000
- saliency: deletion-AUC 0.749, P@k 1.000
- ig: deletion-AUC 0.748, P@k 1.000
- smoothgrad: deletion-AUC 0.738, P@k 1.000
- saliency: deletion-AUC 0.558, P@k 0.400
- ig: deletion-AUC 0.550, P@k 0.200
- smoothgrad: deletion-AUC 0.624, P@k 0.400
- saliency: deletion-AUC 0.531, P@k 0.167
- ig: deletion-AUC 0.532, P@k 0.333
- smoothgrad: deletion-AUC 0.558, P@k 0.167
- saliency: deletion-AUC 0.634, P@k 0.333
- ig: deletion-AUC 0.478, P@k 0.167
- smoothgrad: deletion-AUC 0.616, P@k 0.333
- saliency: deletion-AUC 0.606, P@k 0.167
- ig: deletion-AUC 0.644, P@k 0.167
- smoothgrad: deletion-AUC 0.644, P@k 0.333
- saliency: deletion-AUC 0.529, P@k 0.000
- ig: deletion-AUC 0.580, P@k 0.000
- smoothgrad: deletion-AUC 0.575, P@k 0.000
- saliency: deletion-AUC -0.214, P@k 0.000
- ig: deletion-AUC 0.147, P@k 0.000
- smoothgrad: deletion-AUC -0.299, P@k 0.000
- saliency: deletion-AUC -0.088, P@k 0.000
- ig: deletion-AUC 0.242, P@k 0.000
- smoothgrad: deletion-AUC 0.167, P@k 0.000

> Placeholders marked NOT RUN must be replaced by real runs before any submission.
