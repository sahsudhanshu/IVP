# Improved architecture — submission copy

This folder contains the ResNet-18 + temporal Transformer candidate, dataset preparation, training and prediction scripts, plus the supplied Kaggle notebook. The `train.py` in this submission copy fixes two split leaks found in the supplied version: cross-view development now holds out subject 10 from the source views; cross-subject development holds out subject 9 for validation and subject 10 for testing, and the final refit uses subjects 1–9 only. Cross-subject inference uses view 3, so this proxy must not be described as an all-view protocol.

The supplied notebook has not been updated with these split fixes; use the Python scripts here, not the notebook, until it is corrected. `README_SUPPLIED.md` and `COMPARISON.md` preserve descriptions from the supplied materials; their performance claims have not been independently verified. See `../../SUBMISSION_AUDIT.md`.

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

The currently supplied improved checkpoint is only a two-frame, 64-pixel, one-epoch diagnostic and is stored separately as such. It is not suitable as the final trained model.
