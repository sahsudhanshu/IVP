# Comparative Architectural Analysis: NW-UCLA Action Recognition

## Evidence Status & Benchmark Summary

All experimental configurations have now been evaluated under consistent protocols on the NW-UCLA dataset (418 complete 3-view groups across 10 action classes) for the held-out Camera View 3 benchmark (Cross-View {1, 2} -> 3).

### Cross-View {1, 2} -> 3 Benchmark Comparison

| Stage | Model Configuration | Epochs | View 3 Accuracy | Macro-F1 | Primary Source Log |
|---|---|:---:|:---:|:---:|---|
| **Historical Baseline** | Inception-v3 + GRU | 12 | **36.60%** | **0.2947** | `ivp-implementation-nw-ucla-tsn-gru-base-paper.log` |
| **Augmentation Run** | Inception-v3 + GRU + Aug | 12 | **39.47%** | **0.3487** | `nwucla-measured-augmentation.log` |
| **Improved Architecture** | ResNet-18 + Transformer | 20 dev / 12 refit | **86.12%** | **0.8608** | `runs/experiment_01/training.jsonl` |
| **Base Paper Benchmark [1]** | Wang et al. | 80 | **89.76%** | --- | Published Paper Reference |

---

## Architectural Comparison Table

| Aspect | Historical Baseline | Improved Architecture | Rationale & Impact |
|---|---|---|---|
| **Spatial encoder** | ImageNet-pretrained Inception-v3 (27M) | ImageNet-pretrained ResNet-18 (11.2M) | Compact feature extractor prevents overfitting on small action datasets. |
| **Temporal samples** | 4 ordered segments (TSN) | 8 ordered segments | Doubles temporal coverage, resolving motion ambiguity in multi-phase actions. |
| **Temporal aggregation** | Shared GRU, last hidden output | 2-layer Transformer + Learned Attention Pooling | Self-attention aggregates non-adjacent frames with content-based weighting. |
| **Camera branches** | Dual branch convolutional refinement + concat | Single shared model with paired-view training | Encourages view-invariant feature representation without doubling params. |
| **Inference mode** | Target view duplicated into both branches | Direct single-view forward pass | Eliminates artificial branch duplication during test time. |
| **Training objective** | Cross-entropy only | CrossEntropy (smoothing=0.05) + Cosine Consistency (0.1) | Direct geometric alignment of embeddings from different viewpoints. |
| **Optimization** | SGD with step decay | AdamW (lr=3e-4, bb=3e-5) with Cosine Annealing | Smoother gradient descent and stable Transformer fine-tuning. |

See complete experiment outputs in [`runs/experiment_01/`](runs/experiment_01/).
