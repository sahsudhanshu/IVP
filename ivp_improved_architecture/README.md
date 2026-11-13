# IVP improved architecture: multi-view action classification

This is an independently implemented candidate architecture for NW-UCLA **video classification**. It predicts one action for a complete video. It is not a segmentation network: this dataset/package does not supply pixel-level segmentation masks, so a segmentation model and segmentation accuracy cannot be trained or measured here without a suitable labelled dataset.

**Status:** code and a tiny diagnostic training/refit/inference workflow have been executed successfully. The full architecture has **not** been trained or evaluated, and no improvement has been measured. See [COMPARISON.md](COMPARISON.md).

## Architecture

Each of 8 ordered sampled RGB frames is encoded by a shared ImageNet-pretrained ResNet-18. A learned projection and position embedding feed a two-layer temporal Transformer. Attention pooling produces a clip embedding, and a linear classifier predicts the action. During training, views 1 and 2 of the same action/subject/event are processed independently by the same network. Both predictions receive classification loss, with an additional cosine-distance loss on their paired embeddings. At inference the model takes **one video**, including the target view 3; it does not duplicate it to emulate two source cameras.

```text
Video -> 8 sampled RGB frames -> shared ResNet-18 -> temporal Transformer
      -> learned attention pooling -> 10 action classes

Training only: paired view 1 and view 2 -> shared model -> classification + paired-view consistency loss
```

Training uses clip-consistent horizontal flipping and mild brightness/contrast perturbation, applied once per video; deterministic center-of-segment sampling and no random augmentation are used for validation and test. The model expects the NW-UCLA action IDs 1, 2, 3, 4, 5, 6, 8, 9, 11, and 12.

## Setup on Windows or Kaggle

Use Python 3.9 or later with a compatible PyTorch and torchvision pair. A CUDA-enabled PyTorch build and GPU are recommended; CPU works but is much slower. For CUDA, install the appropriate PyTorch build for the machine first. Then from this folder run:

```powershell
python -m pip install -r requirements.txt
python prepare_data.py --archive "../Course_Project_Group_19/06_Test_Data/NWUCLA_RGB.zip" --output data
```

`prepare_data.py` extracts only the dataset AVI files and refuses to overwrite existing files. The archive must be available locally; the repository ignores large dataset archives. The first real training run needs access to torchvision's pretrained ResNet-18 weights, either already cached or downloadable. Use `--no-pretrained` for diagnostics only; do not claim a pretrained experiment if those weights were unavailable.

## Train and evaluate

First perform a quick correctness check, not a real accuracy run:

```powershell
python train.py --data data --output runs/smoke --smoke --no-pretrained --frames 2 --size 64
```

For a real run, choose the configuration before inspecting target-view metrics. `--output` must name a **new** directory so previous artifacts are not overwritten:

```powershell
python train.py --data data --output runs/experiment_01 --epochs 20 --batch-size 2
```

This trains on source views 1 and 2, holds subject 10 out for source-view development validation, selects an epoch by validation accuracy (macro-F1 breaks ties), then initializes a fresh model and refits on all source-view groups for the selected number of epochs. It saves `config.json`, `split_manifest.json`, timestamped `training.jsonl`, `development_best.pt`, and `refit.pt`. No final target-view result is produced by this command.

To evaluate view 3 immediately after refitting in a planned experiment, append `--evaluate-test` to the real training command. The program then writes actual `test_metrics.json`, `test_predictions.csv`, and `confusion_matrix.csv`. Treat the view-3 test as held out; do not tune hyperparameters or select epochs using it. Development validation averages the source-view probabilities, whereas the previous baseline concatenated view branches, so compare results with this difference disclosed. The cross-view split does not hold out subjects from training.

For memory pressure, reduce `--batch-size` or `--frames` (the latter changes the model and experiment). On Windows, the default `--workers 0` is reliable; increase it only if appropriate. The default first run downloads pretrained weights if not cached. Exact CUDA determinism is not guaranteed; repeat with multiple seeds to estimate variability.

## Predict one clip

```powershell
python predict.py --checkpoint runs/experiment_01/refit.pt --video path/to/video.avi
```

The new checkpoint has a different architecture from the existing demo checkpoint and cannot be loaded into the old demo unchanged. No checkpoint for this new architecture is supplied until a full training run is performed. `--no-pretrained` or a smoke checkpoint must not be described as a trained performance result.

## Files

- `model.py`: encoder, temporal aggregation, and paired loss.
- `data.py`: video inventory, segment sampling, clip-consistent augmentation and dataset splits.
- `prepare_data.py`: extraction of dataset videos from the local archive.
- `train.py`: development, refit, optional held-out evaluation, artifacts and logs.
- `predict.py`: single-video inference.
- `COMPARISON.md`: architectural rationale, limitations and evidence status.
