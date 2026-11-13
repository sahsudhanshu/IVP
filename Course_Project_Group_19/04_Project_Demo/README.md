# Group 19 video classification demo

## Show the demo on this computer

1. Double-click `run_demo.bat`. The sample clip appears in a looping preview.
2. Click **Classify**. The result and model score appear below the video. The first run may take longer because the model loads into memory.
3. Optionally select a different AVI, MP4, MOV or MKV clip using **Choose video**.

The included sample is an NW-UCLA video for action a01, pick up with one hand. A local inference test predicted that action with model score 21.2%. The score is not a calibrated probability. Predictions on unfamiliar videos may be wrong.

## Files to submit

Include this entire `04_Project_Demo` folder and `../12_Trained_Models/refit_checkpoint.pt`. The checkpoint is required for inference; it is the recorded Kaggle Inception-v3 run, not a verified checkpoint for the reported 78% local run. If the instructor requires a recorded demonstration, record the preview, classification and result; no recording is included here.

For another computer, install Python with Tkinter, run `pip install -r requirements.txt`, then `python demo.py`. Keep the checkpoint in the adjacent `12_Trained_Models` folder. The batch launcher specifically uses this computer's Miniconda `matsya` environment and local `.model_test_deps` directory; use the manual installation on another computer. The code samples four evenly spaced frames and passes the same clip through both branches, matching the recorded cross-view test protocol.

For a screen recording, open this demo, display the sample video, click Classify and capture the resulting prediction in MP4 or AVI.
