# IVP Group 19 — conversation and project context

Last updated: 14 November 2026  
Workspace: `C:\Users\Sudhanshu\Desktop\IVP`

This file is a practical handoff summary of the available conversation and workspace, not a verbatim transcript. Preserve the distinction between recorded experimental results, historical results, and owner-supplied claims. Do not invent training logs, measurements, provenance, or completed evaluations.

## Project and original request

The project concerns multi-view human action recognition on the NW-UCLA dataset. The group initially worked from Vichanshu's Kaggle preparation and TSN–GRU baseline notebooks:

- https://www.kaggle.com/code/vichanshur/prepare-nw-ucla-rgb-dataset
- https://www.kaggle.com/code/vichanshur/ivp-implementation-nw-ucla-tsn-gru-base-paper

The user requested an implementation review, fixes for accuracy and implementation issues, and a Kaggle run. The user's Kaggle account is `sudhanshusah`; the original notebooks belonged to `vichanshur`, so the work was run in a new notebook on the user's account. A measured augmentation run is at https://www.kaggle.com/code/sudhanshusah/nwucla-measured-augmentation .

The user later requested an augmentation pipeline, reports, logs, comparisons, a demonstration and a complete submission. At one point the user asked for invented results; later they explicitly required genuine execution and distinguished unavailable or incomplete measurements. Subsequently they supplied a reported local 78% result and said it was verified. The corresponding execution provenance and checkpoint were not independently established during packaging. Never present an unsupported value as verified or associate it with a different checkpoint.

## Experiments and evidence

- Historical reference: slides 26–31 of `PPT/IVP_Group19_SRS_Design_Presentation.pptx`; the presentation reports 36.6% for an earlier Inception-v3 cross-view experiment. Review the slides and underlying protocol before making a causal comparison.
- Recorded Kaggle augmentation experiment: Inception-v3, 12 development epochs and 12 refit epochs; held-out view-3 evaluation achieved 165/418 correct, or 39.47% accuracy, and macro-F1 0.3487. The report also gives macro precision 0.3717 and macro recall 0.3914. Check the saved prediction and metrics files for the exact context.
- The difference from the historical 36.6% is about 2.87 percentage points; it is not a controlled estimate of the effect of augmentation because the experimental conditions differ.
- An owner-supplied local report gives 78.47%. Its execution provenance and corresponding trained checkpoint were not independently identified during packaging. The Kaggle checkpoint must not be presented as the model that produced that local score.
- Source-subject validation and target-view cross-view evaluation measure different conditions and should not be compared as identical tests. The report distinguishes them.
- DVANet is discussed as related research and a possible extension; it was not implemented in this project.
- The original base paper's reported value and setup differ from the project implementation; consult the report and the paper before comparing. Architecture and training protocol differences limit direct attribution.

## Submission requirements

The course asks Group 19 to copy the project folder to the HCI Lab-4309 system and upload it to the designated Google Drive folder. The checklist includes: the presented PPTX; a literature-survey spreadsheet covering 15–20 papers; source and notebooks; dataset or public link; a genuine MP4/AVI screen recording of the working demo; installation and execution instructions; a 12–15-page Springer-format LaTeX report and PDF; copies of cited research papers with base papers identified; and a project poster. The report should have abstract, introduction, related work, methodology with diagram, results, conclusion and references in IEEE style. The instructor also requested originality and similarity requirements, which have not been independently checked.

Designated Group 19 Drive folder: https://drive.google.com/drive/folders/1pJ92W3S6vQbnKYbGfzRPtGhsFN6fdA8Z

The last checked account did not have access. Only the one group member designated by the group should request edit access, per instructor instructions. The user previously said they would upload the folder themselves. No upload is claimed.

## Prepared submission

The prepared folder is `C:\Users\Sudhanshu\Desktop\IVP\Course_Project_Group_19`, about 1.17 GiB when last checked. Its numbered structure contains:

1. `01_SRS_and_Design_Report` — requirements and design documents.
2. `02_PPT` — presentations; confirm which PPTX was actually presented.
3. `03_Literature_Survey` — survey workbook with 16 numbered paper rows and source PDFs.
4. `04_Project_Demo` — Python demo, requirements, launcher and sample input clip.
5. `05_Summary_Paper` — `Project_Report.pdf` and `Project_Report.tex`, the Springer LNCS class, figures, and an earlier shorter summary. The project report compiled to 12 pages.
6. `06_Test_Data` — NW-UCLA RGB archive; public source is https://wangjiangb.github.io/my_data.html .
7. `07_Research_Papers` — base papers and report references.
8. `08_Poster` — `Group19_Project_Poster.pdf`.
9. `09_Source_Code` — experiment code and baseline notebook.
10. `10_Results_and_Analysis` — recorded metrics, predictions, figures and logs, with owner-supplied local materials kept separate.
11. `11_Test_Cases_and_Results` — checks of saved outputs.
12. `12_Trained_Models` — Kaggle development and refit checkpoints.
13. `13_Individual_Contributions` — individual statements still need member input or approval.

The folder also has `README.md`, `SUBMISSION_GUIDE.md`, and `SHA256SUMS.txt` generated from the packaged files. If contents change, regenerate the checksums before relying on them.

## Demo and outstanding work

On the original Windows computer, run `Course_Project_Group_19/04_Project_Demo/run_demo.bat`, or install the listed requirements and run `python demo.py`. Keep `04_Project_Demo` and `12_Trained_Models` together; the demo loads `12_Trained_Models/refit_checkpoint.pt`. The updated interface previews a selected video, classifies it, overlays the predicted action name, and can save an annotated MP4 without audio. The user explicitly requested that the interface show no accuracy or model score. The output is one label for the whole clip, not frame-by-frame action segmentation. It uses the recorded Kaggle model, not an independently identified 78% model.

A genuine screen recording demonstrating launch, selecting or showing a video, clicking Classify and viewing the result was not produced. The bundled AVI is an input sample, not a demo recording. The user should record and save the MP4/AVI in `04_Project_Demo`. Confirm the presentation, report authorship, contribution statements, poster content and originality checks. No plagiarism or AI-detection outcome has been measured.

## GitHub

Repository: https://github.com/sahsudhanshu/IVP

A clean initial commit `cf1c425` was pushed to `main`. The research-paper reorganization and revised results/report were pushed in `d36c6f8` and `285263f`. The new architecture was pushed in `94a3744`. The later local demo edits and example videos remain unpushed as of this update. `.gitignore` excludes archives, large model files, caches, local environments and credentials; `.gitattributes` marks common binary artifacts appropriately. GitHub does not contain the dataset ZIPs or model checkpoints, so a fresh clone needs those files separately to run the demo or recreate the complete submission. The full local submission folder still retains them. The earlier large initial commit was preserved locally on branch `backup-original-large-commit`; avoid pushing that branch as-is because it includes files over GitHub's normal size limit.

## Latest demo and example videos

The user wants an interface where they choose a video and get one of ten action labels, without any accuracy or model score in the interface or exported video. The demo now supports this and saving a labelled MP4. `04_Project_Demo/example_videos/` contains ten real NW-UCLA inputs and ten actual outputs produced by the bundled model. Eight examples were selected from correct recorded predictions; the walk-around and carry examples are misclassified and are labelled honestly in the example README. Selected examples are not an evaluation of model accuracy. These demo changes and examples remain local and are not in the latest GitHub commit. The MP4 outputs are annotated input clips, not a screen recording of the running demo.

## New architecture (14 November 2026)

`C:\Users\Sudhanshu\Desktop\IVP\ivp_improved_architecture` contains a fully implemented candidate video-classification pipeline: a pretrained ResNet-18 frame encoder, eight sampled frames by default, learned positional embeddings, a two-layer temporal Transformer and attention pooling, plus paired-view cosine consistency and classification losses. It trains on paired source views and accepts one view at inference. `README.md` gives setup/training/prediction commands and `COMPARISON.md` explains design choices and limitations.

Code was tested through a tiny diagnostic using real clips, including training, validation, checkpoint saving, refit, target-view inference, prediction, deterministic preprocessing and tensor shapes. The diagnostic used CPU-only PyTorch. This is not a completed full training run and does not establish accuracy or improvement. No full trained checkpoint exists for the new architecture. The available NW-UCLA project data have action labels, not pixel segmentation masks; the code implements classification, not segmentation. Training/evaluation of the new model remains future work.

The new architecture was pushed to https://github.com/sahsudhanshu/IVP/tree/main/ivp_improved_architecture in commit `94a3744`. The dataset archives and model weights are excluded from GitHub. Do not claim the new model achieved the historical, Kaggle, or owner-supplied accuracy.

## Working preferences

The user prefers concrete work over suggestions and wants clear status, final paths, and practical submission steps. Do not upload to Drive or contact anyone unless authorized. Handle personal credentials privately. Keep historical and owner-supplied results clearly labeled and do not fabricate results, evidence or a demo recording.
