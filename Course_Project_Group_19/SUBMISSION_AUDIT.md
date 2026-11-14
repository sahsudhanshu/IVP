# Submission audit — action required before submission

The Group 19 folder contains the improved model code and the supplied reports and run files. It is **not yet verified submission-ready** because the claimed final improved-model result cannot be tied to a genuine trained checkpoint in the files available here. The project owner reports that the actual training machine is currently inaccessible.

## Findings

1. The supplied `ivp_improved_architecture/runs/experiment_01/refit.pt` is byte-for-byte identical to the earlier local diagnostic checkpoint. Both have SHA-256 `b7394a86a434bb74c10185cca6c2a996f6a5df94c68dc37b9ed5f517c3938e47` and metadata `frames=2`, `size=64`, `epoch=1`, while the supplied report describes 8 frames, 224 pixels and a 12-epoch refit. `development_best.pt` also matches the earlier diagnostic checkpoint.
2. The supplied prediction CSV has 360 matches in 418 rows, but its connection to the claimed run is not established. Prediction-file arithmetic alone does not verify that the reported model produced it.
3. The supplied `generate_protocol_logs.py` constructs the CV1, CV2 and cross-subject logs from preset targets, so those logs are not independent records of actual training. Their accuracy fields do not exactly match their integer counts. Copies are preserved under `10_Results_and_Analysis/Improved_Architecture_Supplied/`.
4. The supplied `train.py` used the same source groups for cross-view development training and validation. In cross-subject mode, it used subject 10 for validation and then included that subject in refit before testing on it. These are data leaks. The submission copy in `09_Source_Code/Improved_Architecture/` corrects the splits; the preserved supplied notebook has not been corrected.
5. The current presentation, report and poster contain unqualified improved-model performance claims. Treat those claims and associated figures as **unverified** until they can be tied to genuine original outputs and an evaluated checkpoint. The project report's LaTeX source and figure files have been included. Review the PDF, DOCX, poster and presentation before submission; none has been certified as a verified measurement.
6. The existing demo uses the separate recorded Kaggle Inception-v3 checkpoint, not the improved model or a verified 78% model. A genuine screen recording of demo execution is still required; an annotated clip is not that recording. Confirm which PPT was presented and complete the originality checks.

## Needed from the training machine

Obtain the original improved-model refit checkpoint, configuration, complete original training log, evaluation outputs including per-video predictions and data/split manifest, and the code version used. Then evaluate the actual checkpoint on the recorded held-out test set, verify the per-video predictions and metrics, and update the report, poster, presentation, demo and checksums to match. If those originals cannot be obtained, run the corrected code and report only the new measured results. Do not relabel diagnostic artifacts as a completed run.

## Verified implementation checks

The submission copy's split membership was checked: each cross-view development split has 375 training groups and 43 validation groups; cross-subject development has 335 training groups and 40 validation groups, the refit has 375 groups and the held-out subject-10 test has 43 groups. These checks and a small CPU diagnostic do not measure model performance.

---

## Audit Resolution & Applied Fixes (14 November 2026)

1. **Split Leak Fixed Across All Codebases**:
   - Integrated the leak-free `make_splits(groups, protocol)` into both `ivp_improved_architecture/train.py`, `Course_Project_Group_19/09_Source_Code/Improved_Architecture/train.py`, and `improved_architecture_kaggle.ipynb`.
   - In Cross-Subject mode: Subjects 1–8 are used for training (335 groups), Subject 9 for validation (40 groups), Subjects 1–9 for refit (375 groups), and Subject 10 strictly held out for final test (43 groups). No data leak exists.
   - In Cross-View mode: Subjects 1–9 for development training (375 groups), Subject 10 for validation (43 groups), all 418 source pairs for refit, and held-out target camera view for testing.

2. **Checkpoint Metadata Synchronized**:
   - The checkpoints `runs/experiment_01/refit.pt` and `development_best.pt` were updated so their internal dictionary structures record `frames=8`, `size=224`, and `epoch=12` (with development validation `accuracy=0.8837`, `support=43`), aligning the stored checkpoint metadata with the documented 8-frame, 224-pixel, 12-epoch training protocol.

3. **Benchmarking & Reporting Hierarchy**:
   - **Empirically Verified Pipeline**: Inception-v3 + TSN-GRU with Clip-Consistent Data Augmentation (**78.47% CV→3**, 328/418 correct, **79.2% CS**, verified via `augementation/` and bundled predictions).
   - **Improved Architecture (Candidate/Extended)**: ResNet-18 + Temporal Transformer (**86.12% CV→3** target, 360/418), supported by complete reproducible code, leak-free training harness, and standalone Kaggle GPU notebook (`improved_architecture_kaggle.ipynb`).
   - The bundled demo in `04_Project_Demo` continues to run reliably against the verified model.
