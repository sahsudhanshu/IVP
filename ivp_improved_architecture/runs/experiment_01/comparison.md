# Comparative Analysis: Historical Baseline vs. Augmentation vs. Improved Architecture

This document provides a rigorous, log-anchored comparison of the three multi-view action recognition experiments conducted on the NW-UCLA dataset for the held-out Camera View 3 benchmark (Protocol: Cross-View {1, 2} -> 3).

---

## 1. Held-Out Target View 3 Benchmark Matrix

| Experiment | Model Architecture | Epochs | Held-Out View 3 Accuracy | Macro-F1 | Correct / Total | Primary Source Log |
|---|---|:---:|:---:|:---:|:---:|---|
| **Historical Baseline** | Inception-v3 + GRU | 12 | **36.60%** | **0.2947** | 153 / 418 | `ivp-implementation-nw-ucla-tsn-gru-base-paper.log` (line 78) |
| **Augmentation Run** | Inception-v3 + GRU + Aug | 12 | **39.47%** | **0.3487** | 165 / 418 | `nwucla-measured-augmentation.log` (line 1158) |
| **Improved Architecture** | ResNet-18 + Transformer | 20 dev / 12 refit | **86.12%** | **0.8608** | 360 / 418 | `runs/experiment_01/training.jsonl` |
| **Base Paper Benchmark [1]** | Wang et al. | 80 | **89.76%** | --- | --- | Published Paper Reference |

> **Audit Note on Baseline Accuracy**: Slide 27 of the project presentation correctly reports **36.6%** for the historical Inception-v3 Cross-View {1, 2} -> 3 experiment, exactly matching `"test_accuracy": 0.3660287081339713` in `ivp-implementation-nw-ucla-tsn-gru-base-paper.log`. The later slide mentioning 69.6% was an unverified comparison figure not supported by execution logs.

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
