# Results Summary

## Cross-View Action Recognition on NW-UCLA RGB

| Stage | Accuracy (CV3) | Macro-F1 | Correct / 418 |
|---|:---:|:---:|:---:|
| **Baseline (Before Aug.) [21]** | 70.8% | 0.695 | 296 |
| **Augmentation Run [36]** | 78.5% | 0.784 | 328 |
| **Improved Arch. (After) [39]** | **88.97%** | **0.83** | **372** |
| Base Paper [1] (80 ep) | 89.76% | — | — |

## Improved Architecture — All Splits

| Split | Accuracy | Macro-F1 | Correct / Total |
|---|:---:|:---:|:---:|
| Cross-view 1 (views 2,3 → 1) | 80.62% | 0.79 | 337/418 |
| Cross-view 2 (views 1,3 → 2) | 83.25% | 0.81 | 348/418 |
| **Cross-view 3 (views 1,2 → 3)** | **88.97%** | **0.83** | **372/418** |
| Cross-subject | 83.72% | 0.831 | 36/43 |
| **CV Mean** | **84.28%** | — | 1057/1254 |

## Augmentation Results Detail

The clip-consistent augmentation pipeline reports **78.47% accuracy (328 of 418)** on target view 3 and **0.7845 macro-F1**.

Original logs and prediction files are preserved in their respective folders.
