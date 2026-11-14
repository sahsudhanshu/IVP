# Improved architecture — ResNet-18 + Temporal Transformer

This folder contains the complete PyTorch implementation of the improved architecture combining a ResNet-18 visual backbone with a Temporal Transformer encoder and multi-view attention fusion. The implementation supports both Cross-View (views {1,2} → 3, {2,3} → 1, {1,3} → 2) and Cross-Subject (Subjects 1–9 train/val, Subject 10 test) protocols with strict data separation.

Comprehensive benchmark results and comparative analyses are documented in `COMPARISON.md` and `../../SUBMISSION_AUDIT.md`.

## Setup

Use Python 3.9+ and a compatible PyTorch/torchvision pair. Install a CUDA-enabled PyTorch build appropriate for the machine if training on GPU, then from this folder run:

```powershell
python -m pip install -r requirements.txt
python prepare_data.py --archive ../../06_Test_Data/NWUCLA_RGB.zip --output data
python train.py --data data --output runs/cv3 --protocol cv3 --epochs 20 --batch-size 2 --evaluate-test
```

The first training run needs the ImageNet pretrained ResNet-18 weights cached or downloadable. `--output` must be a new directory. The held-out test view should be evaluated only after fixing the configuration. For `cs`, subject 10 is the test subject; subject 9 is used only for development validation and is included in the final refit.

A `--smoke --no-pretrained --frames 2 --size 64` run is a diagnostic only, not an accuracy experiment. After a genuine refit, run:

```powershell
python predict.py --checkpoint runs/cv3/refit.pt --video path/to/video.avi
```

Trained checkpoints are stored in `12_Trained_Models/` and can be loaded directly for inference or further fine-tuning.
