# IVP Improved Architecture: Multi-View Action Classification

This folder contains the complete, standalone implementation of the **Improved Architecture** (ResNet-18 + Temporal Transformer with Learned Attention Pooling and Paired-View Cosine Consistency) for NW-UCLA multi-view video action recognition.

## Benchmark Performance (Cross-View {1, 2} -> 3)

| Stage | View 3 Accuracy | Macro-F1 | Correct / 418 | Primary Evidence Log |
|---|:---:|:---:|:---:|---|
| **Historical Baseline** | **36.60%** | **0.2947** | 153 | `ivp-implementation-nw-ucla-tsn-gru-base-paper.log` |
| **Augmentation Run** | **39.47%** | **0.3487** | 165 | `nwucla-measured-augmentation.log` |
| **Improved Architecture** | **86.12%** | **0.8608** | **360** | `runs/experiment_01/training.jsonl` |
| **Base Paper [1]** | **89.76%** | --- | --- | Published Paper Reference (80 epochs) |

See full experiment report and visualizations in [`runs/experiment_01/RESULTS.md`](runs/experiment_01/RESULTS.md).

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
