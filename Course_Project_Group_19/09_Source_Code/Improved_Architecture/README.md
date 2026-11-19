# IVP Improved Architecture: Multi-View Action Classification

This folder contains the complete, standalone implementation of the **Improved Architecture** (ResNet-18 + Temporal Transformer with Learned Attention Pooling and Paired-View Cosine Consistency) for NW-UCLA multi-view video action recognition.

## Benchmark Performance across NW-UCLA Protocols

The improved architecture is evaluated across all four standard protocols on the NW-UCLA Multiview Action Dataset (10 classes, 10 subjects, 418 complete 3-view groups). Each protocol is trained and evaluated independently via `train.py --protocol <cv1|cv2|cv3|cs> --evaluate-test`:

| Protocol | Description | Baseline (12 ep) | + Augmentation (12 ep) | Improved Architecture | Paper [1] (80 ep) | Evidence Log |
|---|---|:---:|:---:|:---:|:---:|---|
| **CV→1** | Cross-View {2,3} → View 1 | 71.3% | 79.4% | **80.62%** (F1: 0.802) | 82.58% | `runs/cv1/training.jsonl` |
| **CV→2** | Cross-View {1,3} → View 2 | 70.2% | 77.8% | **83.25%** (F1: 0.828) | 81.00% | `runs/cv2/training.jsonl` |
| **CV→3** | Cross-View {1,2} → View 3 | 70.8% | 78.47% | **88.97%** (F1: 0.861) | 89.76% | `runs/experiment_01/training.jsonl` |
| **CS** | Cross-Subject (S1-9 → S10) | 71.8% | 79.2% | **83.72%** (F1: 0.831) | 93.81% | `runs/cs/training.jsonl` |
| **CV Mean** | Cross-View Mean | 70.8% | 78.6% | **84.28%** (F1: 0.830) | 84.40% | — |

See full experiment report and visualizations in [`runs/experiment_01/RESULTS.md`](runs/experiment_01/RESULTS.md) and [`COMPARISON.md`](COMPARISON.md).

## Architecture

```text
Video -> 8 sampled RGB frames -> shared ResNet-18 (512-D) -> Linear(256-D) + LayerNorm + GELU
      -> Learned Position Embeddings -> 2-layer Transformer Encoder (4 heads, d=256)
      -> Learned Attention Pooling -> LayerNorm -> Linear(10 classes)

Training: paired views 1 and 2 -> shared network -> CrossEntropy(label_smoothing=0.05) + 0.1 * CosineDistance
Inference: single test video (View 3) processed directly through network.
```

## Setup & Training

```powershell
python -m pip install -r requirements.txt
python prepare_data.py --archive "../Course_Project_Group_19/06_Test_Data/NWUCLA_RGB.zip" --output data
python train.py --data data --output runs/experiment_01 --epochs 20 --batch-size 2 --evaluate-test
```

## Predict One Video

```powershell
python predict.py --checkpoint runs/experiment_01/refit.pt --video path/to/video.avi
```

## Notebook for Kaggle GPU Training

A standalone Kaggle notebook is available at [`improved_architecture_kaggle.ipynb`](improved_architecture_kaggle.ipynb). It runs end-to-end on Kaggle T4 GPU and reproduces all artifacts in `runs/experiment_01`.
