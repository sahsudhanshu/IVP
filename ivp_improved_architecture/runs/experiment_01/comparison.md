# Comparative Analysis: NW-UCLA Cross-View {1,2} → 3 Benchmark

This document provides a comparison of the three experiment stages for the held-out Camera View 3 benchmark (Protocol: Cross-View {1, 2} -> 3). See [`COMPARISON.md`](../../COMPARISON.md) for the full multi-protocol summary including CV→1, CV→2, and Cross-Subject.

---

## 1. Held-Out Target View 3 Benchmark Matrix

| Experiment | Model Architecture | Epochs | Held-Out View 3 Accuracy | Macro-F1 | Evidence |
|---|---|:---:|:---:|:---:|---|
| **Baseline (Inception-v3 + GRU)** | Inception-v3 + GRU | 12 | **70.8%** | **0.695** | Baseline implementation |
| **+ Data Augmentation** | Inception-v3 + GRU + Aug | 12 | **78.47%** | **0.785** | `augementation/augmentation_training.log` |
| **Improved Architecture** | ResNet-18 + Transformer | 20 dev / 12 refit | **86.12%** | **0.8608** | `runs/experiment_01/training.jsonl` |
| **Base Paper [1]** | Wang et al. | 80 | **89.76%** | — | Published Paper Reference |

> [!NOTE]
> This file covers the CV→3 protocol only (`--protocol cv3`). The CV→1 (80.71%), CV→2 (83.34%), and CS (83.5%) protocol runs are logged in `runs/cv1/`, `runs/cv2/`, and `runs/cs/` respectively.

---

## 2. Key Architectural Innovations Driving the Improvement

1. **Lightweight Pretrained Backbone (ResNet-18 vs. Inception-v3)**:
   - Inception-v3 contains ~27M parameters and easily overfits on NW-UCLA (418 video groups).
   - ResNet-18 has 11.2M parameters, providing regularized spatial representations with substantially lower parameter count.
2. **Global Temporal Attention (2-Layer Transformer vs. Sequential GRU)**:
   - The GRU relied strictly on final hidden state sequential processing, making it vulnerable to temporal boundary noise.
   - The 2-layer Transformer with learned positional embeddings and attention pooling aggregates non-adjacent frames with content-aware weighting.
3. **Paired-View Geometric Alignment (Cosine Consistency Loss)**:
   - At training time, paired views of the same action/event are penalized via cosine distance between their feature embeddings ($weight=0.1$).
   - This explicitly forces the model to construct view-invariant action representations.
4. **Single-View Target Inference**:
   - The historical model unnaturally duplicated the single test video into two branches at test time.
   - The improved architecture accepts a single video directly at inference.
5. **Exact Cosine Annealing Optimization**:
   - `CosineAnnealingLR` over 20 epochs avoids premature stagnation and smoothly decays learning rates from $3\times 10^{-4}$ to $1.85\times 10^{-6}$.
6. **Multi-Protocol Support (`--protocol` flag)**:
   - `train.py` now accepts `--protocol cv1|cv2|cv3|cs`, making all four standard NW-UCLA benchmarks reproducible from one codebase.
