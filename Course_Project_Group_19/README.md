# IVP Group 19 — Multi-view action recognition

## Quick start: demonstrate one video

1. Keep `04_Project_Demo/` and `12_Trained_Models/` together within this project folder. The demo requires `12_Trained_Models/refit_checkpoint.pt`.
2. On this computer double-click `04_Project_Demo/run_demo.bat`. The included sample video opens in a looping preview.
3. Click **Classify**. The predicted action and model score appear. Use **Choose video** to select another AVI, MP4, MOV or MKV file.
4. The included sample is NW-UCLA action a01, pick up with one hand. The program reports a softmax model score, not a calibrated probability. Predictions can be incorrect.

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
- `05_Summary_Paper/`: `Project_Report.pdf` and `Project_Report.tex` (12 pages), Springer LNCS class and graphics. The earlier six-page summary is also retained.
- `06_Test_Data/`: NW-UCLA RGB archive. Public source: https://wangjiangb.github.io/my_data.html
- `07_Base_Papers/`: two labelled base papers and PDFs for the report's cited papers.
- `08_Poster/`: project poster PDF.
- `09_Source_Code/`: experiment source and baseline notebook.
- `10_Results_and_Analysis/`: original recorded metrics, predictions, graphs and logs; the owner-supplied local result is stored separately.
- `11_Test_Cases_and_Results/`: checks of the saved output consistency.
- `12_Trained_Models/`: recorded Kaggle checkpoints.
- `13_Individual_Contributions/`: member-approved statements still needed if requested.

## Recorded evaluation and limitations

The recorded 12-epoch Inception-v3 augmentation experiment achieved 39.47% accuracy (165/418) and 0.3487 macro-F1 on the view-3 cross-view test. The project owner separately supplied a report of 78.47%; its execution provenance and corresponding model checkpoint were not established during packaging. Do not use the Kaggle checkpoint as though it produced the reported local score. DVANet is reviewed in the report; it is not implemented as part of this demo.

## Before submission

Record a screen demonstration in MP4 or AVI format and place it in `04_Project_Demo/`. Review the report for accuracy, authorship and the instructor's originality requirements; no plagiarism or AI-detection score has been measured. Provide the poster, cited papers, source, data and report directly in the Group 19 Drive folder rather than only as a ZIP. The account last checked did not have access to the designated Drive folder; coordinate the single group access request.
