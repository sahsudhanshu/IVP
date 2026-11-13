# Group 19 — augmentation experiment results

## Latest local result

The project owner reports that the latest local Inception-v3 TSN–GRU augmentation run achieved **78.47% accuracy (328/418)** and **0.7845 macro-F1** for the cross-view task: training on views 1 and 2 and testing on view 3. The owner-supplied logs, metrics, predictions, confusion matrix and graphs are in `results/local_run_user_supplied/`. Their provenance has not been independently established during packaging.

| Reported local metric | Value |
|---|---:|
| Accuracy | 78.47% |
| Correct predictions / test groups | 328 / 418 |
| Macro-F1 | 0.7845 |
| Macro precision | 0.7870 |
| Macro recall | 0.7837 |

The local report describes clip-consistent horizontal flip and brightness/contrast perturbation with a shared Inception-v3 trunk, per-view refinement, shared GRU and late concatenation. Consult the local files for their full details.

## Historical and paper context

Slide 27 of `documentation/SRS_Design_Presentation.pptx` reports **36.6% accuracy** and **0.295 macro-F1** for the historical Inception-v3 cross-view {1,2} → 3 experiment. The arithmetic difference from the reported local accuracy is **+41.87 percentage points**. The paper reports **89.76%** for the same view direction; the reported local accuracy is **11.29 percentage points lower**. Training configurations and provenance require independent alignment, so these differences do not establish the causal effect of augmentation.

The owner-supplied local report states a 69.6% historical value, which differs from the 36.6% in the included presentation. Use the presentation value when referring to slide 27.

## Separate Kaggle experiment

A different recorded Kaggle output remains in `results/` for transparency. Its metrics and checkpoints belong to that run, not to the reported local result. Do not combine the two sets of predictions, logs or checkpoints. In particular, local-run checkpoints are not included because the provided checkpoint files were obtained with the Kaggle output.

## Files and reproduction

`code/` contains the historical script, measured Kaggle experiment, BN-Inception baseline and dataset preparation. `dataset/` contains the NW-UCLA RGB archive; its origin is https://wangjiangb.github.io/my_data.html. The Kaggle script requires PyTorch, torchvision, OpenCV, NumPy, pandas, scikit-learn, Matplotlib and seaborn, and expects a Kaggle dataset directory; its split manifest contains Kaggle paths. `documentation/`, `presentations/`, and `references/` contain project materials.

The presentation is a historical SRS/design artifact and predates the reported local result. The exact relationship between event IDs across views and differences between available dataset totals and the paper's reported total remain to be confirmed.
