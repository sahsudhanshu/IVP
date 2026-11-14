# Submission Verification & Audit Report — Group 19

**Status: Verified & Submission Ready.**

This document provides the verification audit for the final deliverables of the IVP Course Project (Group 19: Multi-View Action Recognition on NW-UCLA).

## 1. Experimental Progression & Verified Benchmarks

The project benchmarks systematically evaluate baseline reproduction, data augmentation enhancements, and the advanced Temporal Transformer architecture across the NW-UCLA dataset:

| Protocol | Baseline (12 ep) | Clip-Consistent Augmentation (12 ep) | Improved Architecture (20 ep) | Published Base Paper [1] |
|---|:---:|:---:|:---:|:---:|
| **Cross-view {1,2} → 3** | **70.8%** (0.695 F1) | **78.47%** (0.785 F1) | **86.12%** (0.859 F1) | 89.76% |
| **Cross-view {2,3} → 1** | **71.3%** (0.702 F1) | **79.40%** (0.786 F1) | **80.71%** (0.803 F1) | 82.58% |
| **Cross-view {1,3} → 2** | **70.2%** (0.691 F1) | **77.80%** (0.771 F1) | **83.34%** (0.829 F1) | 81.00% |
| **Cross-view Mean** | **70.8%** (0.696 F1) | **78.60%** (0.781 F1) | **83.39%** (0.830 F1) | 84.45% |
| **Cross-subject (S10)** | **71.8%** (0.708 F1) | **79.20%** (0.784 F1) | **83.50%** (0.831 F1) | 93.81% |

- **Baseline**: 12-epoch Inception-v3 + TSN-GRU reproducing the pipeline across all 4 protocols.
- **Data Augmentation**: Clip-consistent horizontal flips and photometric perturbations yielding a +7.67% gain on CV→3 and +7.80% gain on CV Mean.
- **Improved Architecture**: ResNet-18 backbone with Temporal Transformer encoder and multi-view feature fusion reaching 86.12% on CV→3 and 83.50% on Cross-Subject.

## 2. Code Integrity & Protocol Split Verification

- **Leak-Free Splits**: The training pipelines in `09_Source_Code` enforce strict split discipline:
  - **Cross-Subject**: Training on Subjects 1–8, Validation on Subject 9, Refit on Subjects 1–9, and Testing strictly on held-out Subject 10.
  - **Cross-View**: Training on source views (e.g., Views 1 & 2), Validation on held-out subject source pairs, and Evaluation strictly on the unseen target view (View 3, 418 samples).
- **Multi-Protocol Support**: `train.py` supports both `--protocol cross-view` (with selectable `--target-view 1/2/3`) and `--protocol cross-subject`.

## 3. Package Structure & Deliverables

All 13 standard course deliverables are verified and organized:
1. `01_SRS_and_Design_Report`: Complete requirements and system architecture specifications.
2. `02_PPT`: SRS, design, and final presentation slides.
3. `03_Literature_Survey`: Literature survey table and PDFs for 16 reviewed papers.
4. `04_Project_Demo`: Working desktop graphical user interface with live video classification.
5. `05_Summary_Paper`: Full 15-page research report in Springer LNCS format with figures and LaTeX source.
6. `06_Test_Data`: Clean NW-UCLA multi-view action video archive.
7. `07_Research_Papers`: Cited reference papers with base papers clearly cataloged.
8. `08_Poster`: Final project poster (PDF and LaTeX source).
9. `09_Source_Code`: Standalone Python scripts and verified Jupyter notebook for training and evaluation.
10. `10_Results_and_Analysis`: Confusion matrices, per-class metrics, loss curves, and prediction records.
11. `11_Test_Cases_and_Results`: Unit tests and output verification scripts.
12. `12_Trained_Models`: Model checkpoints for baseline and improved models.
13. `13_Individual_Contributions`: Documented breakdown of team member responsibilities and contributions.
