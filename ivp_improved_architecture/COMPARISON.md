# Comparative Architectural Analysis: NW-UCLA Action Recognition

## Evidence Status & Benchmark Summary

All experimental configurations have now been evaluated under consistent protocols on the NW-UCLA dataset (418 complete 3-view groups across 10 action classes).

### Performance Matrix Across All 3 Experimental Stages

| Protocol | Historical Baseline | Data Augmentation | **Improved Architecture** | Base Paper [1] (80 ep) |
|---|:---:|:---:|:---:|:---:|
| **Cross-subject** | 71.8% (F1: 0.708) | 79.2% (F1: 0.785) | **83.5% (F1: 0.831)** | 93.81% |
| **Cross-view -> 1** | 71.3% (F1: 0.702) | 79.4% (F1: 0.788) | **80.71% (F1: 0.802)** | 82.58% |
| **Cross-view -> 2** | 70.2% (F1: 0.691) | 77.8% (F1: 0.771) | **83.34% (F1: 0.828)** | **81.00% (Exceeds Paper)** |
| **Cross-view -> 3** | **69.6% (F1: 0.681)** | **78.5% (F1: 0.785)** | **86.12% (F1: 0.858)** | **89.76%** |
| **Cross-view Mean** | **70.4% (F1: 0.691)** | **78.6% (F1: 0.781)** | **83.39% (F1: 0.829)** | **84.4%** |

---

## Architectural Comparison Table

| Aspect | Historical / Augmentation Baseline | Improved Architecture | Rationale & Impact |
|---|---|---|---|
| **Spatial encoder** | ImageNet-pretrained Inception-v3 (27M) | ImageNet-pretrained ResNet-18 (11.2M) | Compact feature extractor prevents overfitting on small action datasets. |
| **Temporal samples** | 4 ordered segments (TSN) | 8 ordered segments | Doubles temporal coverage, resolving motion ambiguity in multi-phase actions. |
| **Temporal aggregation** | Shared GRU, last hidden output | 2-layer Transformer + Learned Attention Pooling | Self-attention aggregates non-adjacent frames with content-based weighting. |
| **Camera branches** | Dual branch convolutional refinement + concat | Single shared model with paired-view training | Encourages view-invariant feature representation without doubling params. |
| **Inference mode** | Target view duplicated into both branches | Direct single-view forward pass | Eliminates artificial branch duplication during test time. |
| **Training objective** | Cross-entropy only | CrossEntropy (smoothing=0.05) + Cosine Consistency (0.1) | Direct geometric alignment of embeddings from different viewpoints. |
| **Optimization** | SGD with step decay | AdamW (lr=3e-4, bb=3e-5) with Cosine Annealing | Smoother gradient descent and stable Transformer fine-tuning. |

See complete experiment outputs in [`runs/experiment_01/`](runs/experiment_01/).
