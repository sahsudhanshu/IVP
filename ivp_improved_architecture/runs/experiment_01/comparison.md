# Three-Stage Experimental Comparison: Historical -> Augmentation -> Improved Architecture

This report compares all three stages of our multi-view action recognition experiments on the NW-UCLA dataset.

---

## 1. Multi-Protocol Performance Matrix

| Evaluation Protocol | Historical Baseline | Augmentation (12 ep) | **Improved Arch (20/12 ep)** | Delta (vs. Hist.) | Paper [1] (80 ep) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Cross-subject** | 71.8% (F1: 0.708) | 79.2% (F1: 0.785) | **83.5% (F1: 0.831)** | **+11.7%** | 93.81% |
| **Cross-view -> 1** | 71.3% (F1: 0.702) | 79.4% (F1: 0.788) | **80.71% (F1: 0.802)** | **+9.41%** | 82.58% |
| **Cross-view -> 2** | 70.2% (F1: 0.691) | 77.8% (F1: 0.771) | **83.34% (F1: 0.828)** | **+13.14%** | 81.00% |
| **Cross-view -> 3** | **69.6% (F1: 0.681)** | **78.5% (F1: 0.785)** | **86.12% (F1: 0.858)** | **+16.52%** | 89.76% |
| **Cross-view Mean** | **70.4% (F1: 0.691)** | **78.6% (F1: 0.781)** | **83.39% (F1: 0.829)** | **+12.99%** | 84.4% |

---

## 2. Gap Analysis vs. Paper [1] (80-Epoch Target)

| Protocol | Our Improved Result | Paper [1] Target | Gap |
|:---|:---:|:---:|:---:|
| **Cross-view -> 1** | **80.71%** | 82.58% | -1.87% |
| **Cross-view -> 2** | **83.34%** | 81.00% | **+2.34% (Exceeds Paper)** |
| **Cross-view -> 3** | **86.12%** | 89.76% | -3.64% |
| **Cross-view Mean** | **83.39%** | 84.4% | -1.01% |

Notably, **Cross-view -> 2 reaches 83.34%**, surpassing the base paper's 81.00% benchmark by **+2.34%**, demonstrating the efficacy of learned attention pooling and paired-view cosine consistency on this viewpoint transition.

---

## 3. Architectural Advancements

| Component | Historical / Augmentation Baseline | **Improved Architecture** |
|:---|:---|:---|
| **Backbone** | Inception-v3 (27M parameters) | **ResNet-18 (11.2M parameters)** |
| **Temporal Coverage** | 4 ordered frames (TSN) | **8 ordered frames** |
| **Temporal Aggregation** | Shared GRU (last hidden state) | **2-layer Transformer + Learned Attention Pooling** |
| **Multi-View Processing** | Branch concatenation (256-D) | **Shared weights + Probability Averaging** |
| **Target View Inference** | Duplicated into both branches | **Direct single-view forward pass** |
| **Regularization & Loss** | Cross-entropy only | **CE (label_smoothing=0.05) + Cosine Consistency (0.1)** |
| **Optimization** | SGD with step decay | **AdamW with Cosine Annealing LR Schedule** |
