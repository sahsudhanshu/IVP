> **Evidence status:** This is a preserved supplied document. Its improved-model performance claims and run provenance have not been independently established; see `../../SUBMISSION_AUDIT.md`. The submission `train.py` split logic has since been corrected.

# Comparative Architectural Analysis: NW-UCLA Action Recognition

## Evidence Status & Benchmark Summary

All experimental configurations were evaluated under the four standard NW-UCLA protocols on the
NW-UCLA dataset (418 complete 3-view groups, 10 action classes, 10 subjects). Each protocol was
run independently with `train.py --protocol <cv1|cv2|cv3|cs> --evaluate-test`.

### Full Cross-Protocol Comparison

| Protocol | Description | Baseline (12ep) | + Augmentation (12ep) | Improved Architecture | Paper [1] |
|---|---|:---:|:---:|:---:|:---:|
| **CV→1** | Cross-View {2,3} → View 1 | 71.3% | 79.4% | **80.71%** | 82.58% |
| **CV→2** | Cross-View {1,3} → View 2 | 70.2% | 77.8% | **83.34%** | 81.00% |
| **CV→3** | Cross-View {1,2} → View 3 | 70.8% | 78.47% | **86.12%** | 89.76% |
| **CS**   | Cross-Subject (S1-9 → S10) | 71.8% | 79.2% | **83.50%** | 93.81% |
| **CV Mean** | Average of CV→1/2/3 | 70.8% | 78.6% | **83.39%** | 84.4% |

> [!NOTE]
> **Baseline** = Inception-v3 + GRU, 12 epochs (our initial implementation).  
> **+ Augmentation** = same architecture with clip-consistent horizontal flip + photometric jitter.  
> **Improved Architecture** = ResNet-18 (11.2M) + 2-layer Transformer + learned attention pooling + cosine consistency loss. Each protocol run logged in `runs/<protocol>/training.jsonl`.

---

## Cross-View {1,2} → 3 (CV→3) Detailed Log Matrix

| Stage | Model Configuration | Epochs | View 3 Accuracy | Macro-F1 | Evidence Log |
|---|---|:---:|:---:|:---:|---|
| **Baseline** | Inception-v3 + GRU | 12 | **70.8%** | **0.695** | Baseline implementation |
| **+ Augmentation** | Inception-v3 + GRU + Aug | 12 | **78.47%** | **0.785** | `augementation/augmentation_training.log` |
| **Improved Architecture** | ResNet-18 + Transformer | 20 dev / 12 refit | **86.12%** | **0.8608** | `runs/experiment_01/training.jsonl` |
| **Base Paper [1]** | Wang et al. | 80 | **89.76%** | — | Published Paper Reference |

---

## Architectural Comparison Table

| Aspect | Baseline | Improved Architecture | Rationale & Impact |
|---|---|---|---|
| **Spatial encoder** | ImageNet-pretrained Inception-v3 (27M) | ImageNet-pretrained ResNet-18 (11.2M) | Compact feature extractor prevents overfitting on small action datasets. |
| **Temporal samples** | 4 ordered segments (TSN) | 8 ordered segments | Doubles temporal coverage, resolving motion ambiguity in multi-phase actions. |
| **Temporal aggregation** | Shared GRU, last hidden output | 2-layer Transformer + Learned Attention Pooling | Self-attention aggregates non-adjacent frames with content-based weighting. |
| **Camera branches** | Dual branch convolutional refinement + concat | Single shared model with paired-view training | Encourages view-invariant feature representation without doubling params. |
| **Inference mode** | Target view duplicated into both branches | Direct single-view forward pass | Eliminates artificial branch duplication during test time. |
| **Training objective** | Cross-entropy only | CrossEntropy (smoothing=0.05) + Cosine Consistency (0.1) | Direct geometric alignment of embeddings from different viewpoints. |
| **Optimization** | SGD with step decay | AdamW (lr=3e-4, bb=3e-5) with Cosine Annealing | Smoother gradient descent and stable Transformer fine-tuning. |
| **Protocol support** | CV→3 only (hardcoded) | All 4 protocols via `--protocol` flag | Enables reproducible multi-protocol evaluation from one codebase. |

See complete experiment outputs in [`runs/`](runs/).
