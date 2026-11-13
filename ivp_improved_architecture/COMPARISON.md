# Comparative architectural analysis

## Evidence status

The new model is a **candidate**; its complete training and final evaluation have not been performed. Any accuracy gain is a hypothesis, not a result. A tiny diagnostic using actual clips checked the training, checkpoint, validation, refit, test-evaluation and prediction code paths. Its metrics are not representative and are not reported as research results.

| Aspect | Existing recorded Kaggle experiment | New candidate | Reasoning and trade-off |
|---|---|---|---|
| Spatial encoder | ImageNet-pretrained Inception-v3 | ImageNet-pretrained ResNet-18 | Smaller backbone can reduce compute and overfitting risk; it may also lose useful visual capacity. |
| Temporal samples | 4 frames | 8 by default | More temporal coverage may capture motion phases but costs more computation and can repeat frames in short clips. |
| Temporal aggregation | Shared GRU, last output | Two-layer Transformer with position embeddings and learned attention pooling | Attention can combine non-adjacent frames and learn which frames matter; on a small dataset it may instead overfit. |
| Camera branches | Two branch-specific convolutional refinements and concatenation | Same model processes each source camera independently | A shared classifier encourages a common representation; removes branch-specific modeling capacity. |
| Unseen camera inference | Target view duplicated into both input branches | One target-view video through the single-view model | Avoids feeding duplicate views to a two-branch model, but does not guarantee better cross-view transfer. |
| Training objective | Action classification | Classification on both paired source views plus cosine consistency of event embeddings | The pair penalty encourages representations of corresponding views to align; excessive weight may erase useful action features. |
| Image perturbations | Clip-consistent flip and mild photometric jitter | Same families of clip-consistent augmentation | Maintains temporal coherence; it is not evidence of an augmentation improvement. |
| Model selection | Recorded run used 12 development and 12 refit epochs | Validation-selected epoch, then fresh source-view refit | May reduce overtraining; the 43-group subject-10 validation split is small and noisy. |

## Protocol and numerical reference

The available recorded Kaggle run achieved **39.47% (165/418)** and macro-F1 **0.3487** on view 3. A separate owner-supplied local result reports **78.47% (328/418)** and macro-F1 **0.7845**; the supplied prediction file reproduces these values, but the run provenance and corresponding checkpoint were not independently established. The earlier presentation reports **36.6%** for a historical experiment. These numbers describe separate experiments and are not scores for the new model.

The candidate holds subject 10 out from source-view development, averages the two source-view probabilities for validation, refits on all complete view-1/view-2 groups, and optionally evaluates each view-3 clip once. This preserves the same source/target camera direction and grouping as the recorded baseline, but changes the architecture, loss, number of sampled frames, optimization, model selection and validation computation. Therefore a difference in accuracy cannot be attributed to one component or augmentation alone. Subject identities appear across camera splits; this is cross-view, not subject-independent evaluation.

For defensible evidence, run the original and candidate code on the same data manifest and camera direction, record seeds and compute budgets, repeat multiple seeds, and reserve target-view predictions for the final fixed configurations. Report accuracy, macro precision/recall/F1, per-class support, confusion matrix, runtime and model size. Ablate attention, paired-view loss and frame count separately. Investigate the existing model's weak walk/carry classes rather than relying on selected demonstration clips. No segmentation comparison is possible without segmentation labels and appropriate metrics.
