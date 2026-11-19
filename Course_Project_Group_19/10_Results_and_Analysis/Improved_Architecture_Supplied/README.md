# Improved Architecture Evaluation Artifacts

This directory contains the evaluation artifacts, logs, confusion matrices, and prediction files for the Improved Architecture (ResNet-18 + Temporal Transformer with Learned Attention Pooling) across all standard NW-UCLA protocols:

- **CV→3 (`runs/experiment_01/` & `runs/cv3/`)**: Cross-View {1,2} → View 3 evaluation achieving **88.97% accuracy (372/418)** and **0.859 macro-F1**.
- **CV→1 (`runs/cv1/`)**: Cross-View {2,3} → View 1 evaluation achieving **80.62% accuracy** and **0.803 macro-F1**.
- **CV→2 (`runs/cv2/`)**: Cross-View {1,3} → View 2 evaluation achieving **83.25% accuracy** and **0.829 macro-F1**.
- **Cross-Subject (`runs/cs/`)**: Cross-subject evaluation on held-out Subject 10 achieving **83.72% accuracy** and **0.831 macro-F1**.

All predictions, per-class metrics, confusion matrices, and test summaries are preserved in their respective subdirectories.
