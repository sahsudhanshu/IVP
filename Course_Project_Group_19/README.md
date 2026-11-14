# IVP Group 19 — Multi-view action recognition

## Quick start: demonstrate one video

1. Keep `04_Project_Demo/` and `12_Trained_Models/` together within this project folder. The demo requires `12_Trained_Models/refit_checkpoint.pt`.
2. On this computer double-click `04_Project_Demo/run_demo.bat`. The included sample video opens in a looping preview.
3. Click **Classify**. The predicted action appears on the video. Click **Save output video** to export an annotated MP4 (without audio). Use **Choose video** to select another AVI, MP4, MOV or MKV file.
4. The included sample is NW-UCLA action a01, pick up with one hand. The program displays one predicted action for the whole clip. Predictions can be incorrect.

## Installation on another Windows system

Install Python 3 with Tkinter. Open a terminal in `04_Project_Demo` and run:

```text
python -m pip install -r requirements.txt
python demo.py
```

The required Python packages are PyTorch, torchvision, NumPy, OpenCV and Pillow. Compatible PyTorch/torchvision builds should be chosen for the target Python and operating system. A CPU is sufficient for the demo; a supported GPU is recommended for training. The recorded training environment used an NVIDIA Tesla T4 GPU, CUDA 12.8 and PyTorch 2.11.0+cu128. The original RGB dataset and ImageNet model-weight source are linked below.

## Project structure

- `01_SRS_and_Design_Report/`: requirements and design documents.
- `02_PPT/`: existing group presentations; select the one actually presented for the final upload.
- `03_Literature_Survey/`: workbook with 16 numbered paper rows, survey document and source PDFs.
- `04_Project_Demo/`: application, requirements and sample clip. A demonstration recording must be added.
- `05_Summary_Paper/`: 15-page summary PDF, DOCX, LaTeX source, Springer LNCS class, and figures.
- `06_Test_Data/`: NW-UCLA RGB archive. Public source: https://wangjiangb.github.io/my_data.html
- `07_Research_Papers/`: two labelled base papers and PDFs for the report's cited papers.
- `08_Poster/`: project poster PDF.
- `09_Source_Code/`: baseline and improved architecture source code and notebooks.
- `10_Results_and_Analysis/`: evaluation metrics, predictions, confusion matrices, and comparative analysis across all protocols.
- `11_Test_Cases_and_Results/`: verification checks and benchmark tests.
- `12_Trained_Models/`: trained model checkpoints for baseline, data augmentation, and improved architectures.
- `13_Individual_Contributions/`: contributions transcribed from the team slide.

## Recorded evaluation and benchmark progression

The baseline implementation achieved 70.8% cross-view accuracy (Macro-F1: 0.695) on held-out View 3 and 71.8% on Cross-subject (12 epochs), matching the official presentation results. With clip-consistent data augmentation (spatial flips and photometric perturbations), accuracy improved to 78.47% (328/418) with a Macro-F1 of 0.7845 on View 3. The improved ResNet-18 + Temporal Transformer architecture further advances performance to 86.12% on View 3 and 83.50% on Cross-subject. DVANet is reviewed in the report as related literature; it is not implemented as part of this demo.

## Submission details

Record a screen demonstration in MP4 or AVI format and place it in `04_Project_Demo/`. Provide the complete numbered folder directly in the Group 19 Google Drive folder for final evaluation.
