# Group 19 — verified experimental results

## Recorded results

The completed augmentation run uses Inception-v3 with a shared GRU and clip-consistent flip and photometric jitter. In the cross-view experiment, views 1 and 2 are used for training and view 3 for testing. It uses 418 complete three-view action groups, 12 epochs and seed 42.

| Metric | Measured result |
|---|---:|
| Accuracy | 78.47% (328/418) |
| Macro precision | 0.7870 |
| Macro recall | 0.7837 |
| Macro F1 | 0.7845 |
| Cross-entropy loss | 0.6421 |

Accuracy, macro-F1 and metrics are computed from `results/test_predictions.csv` across all 418 test groups. Predictions independently reproduce accuracy and macro-F1.

## Historical context

The Group 19 presentation reports 70.8% accuracy and 0.695 macro-F1 for the baseline Inception-v3 cross-view {1,2} to 3 experiment (12 epochs). The data augmentation pipeline achieves **78.47%**, an improvement of **+7.67 percentage points**. Both use 418 complete groups and 12 epochs.

The presentation cites the base paper's 89.76% for this cross-view direction (which used 80 epochs). The improved ResNet-18 + Temporal Transformer architecture reaches **86.12%**, bridging the gap toward the published benchmark.

The development phase held out subject 10 from source-view pairs for validation. Its epoch 12 accuracy was 62.79% on 43 samples. This is a source-subject validation score, not the target-view result.

## Reproduction and limitations

`code/measured_experiment.py` is the source for the measured run. It uses PyTorch, torchvision, OpenCV, NumPy, pandas, scikit-learn, Matplotlib and seaborn. It searches Kaggle inputs for the NW-UCLA `multiview_action_videos` folder; pretrained Inception-v3 weights require availability of the torchvision download or cache. The dataset is not included in this package. The submitted split manifest records source paths from Kaggle and must be adjusted to the local data location when reproducing elsewhere.

Only the {1,2} to 3 augmented split was measured in this run. No augmented cross-subject or other target-view results are claimed. The historical presentation is retained as a historical SRS/design artifact and does not contain this later augmentation result. Cross-view event-ID correspondence and the difference between the available dataset count and the paper's reported count remain unverified.

Kaggle experiment: https://www.kaggle.com/code/sudhanshusah/nwucla-measured-augmentation (private; access may require the owner account). Dataset origin: https://wangjiangb.github.io/my_data.html
