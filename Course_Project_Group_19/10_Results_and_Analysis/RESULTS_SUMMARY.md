# Augmentation results summary

The owner-supplied local evaluation reports **78.47% accuracy (328 of 418)** on target view 3 and **0.7845 macro-F1**. The supplied `Local_Run_Owner_Supplied/test_predictions.csv` reproduces the correct count and macro-F1 (0.7844817 before rounding). The supplied execution provenance and corresponding trained checkpoint have not been independently established.

| Experiment | Accuracy | Evidence and scope |
|---|---:|---|
| Owner-supplied local augmentation | 78.47% | 328/418; supplied prediction file reproduces this count |
| Separate recorded Kaggle augmentation | 39.47% | 165/418; Kaggle evaluation and prediction files in this folder |
| Earlier presentation | 36.6% | Historical Inception-v3 result in slides 26–31 |

These are separate experiments. Their differences do not by themselves establish an augmentation gain. The local comparison file cites a 69.6% historical baseline, whereas the original presentation reports 36.6%; do not conflate the two references. The bundled demo uses the Kaggle checkpoint, not an identified checkpoint for the 78.47% local run.

Original logs and prediction files are preserved in their respective folders.
