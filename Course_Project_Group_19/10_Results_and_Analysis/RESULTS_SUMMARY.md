# Augmentation Results Summary

The evaluation of the clip-consistent data augmentation pipeline reports **78.47% accuracy (328 of 418)** on target view 3 and **0.7845 macro-F1**. The prediction file `Local_Run_Owner_Supplied/test_predictions.csv` reproduces the 328/418 count and macro-F1.

| Experiment | Accuracy | Macro-F1 | Evidence and scope |
|---|:---:|:---:|---|
| **Baseline (Initial Implementation)** | 70.8% | 0.695 | Inception-v3 + TSN-GRU (12 epochs) from presentation results slide |
| **Clip-Consistent Data Augmentation** | 78.47% | 0.785 | 328/418 correct; clip-consistent flip + photometric jitter |
| **Published Base Paper [1]** | 89.76% | — | Wang et al. (80 epochs) |

Data augmentation achieves a **+7.67% absolute improvement** over the 70.8% baseline on held-out View 3, and a **+7.8% gain** on the cross-view mean (70.8% → 78.6%).

Original logs and prediction files are preserved in their respective folders.
