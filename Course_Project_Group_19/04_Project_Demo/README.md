# Group 19 video classification demo

## Show the demo on this computer

1. Double-click `run_demo.bat`. The sample clip appears in a looping preview.
2. Click **Classify**. The predicted action appears below the video and is overlaid on the playing video. The first run may take longer because the model loads into memory.
3. Click **Save output video** to export the annotated clip as an MP4. The exported video does not include audio. `sample_classified_output.mp4` is an output produced from the bundled sample.
4. Optionally select a different AVI, MP4, MOV or MKV clip using **Choose video**. The overlay is one prediction for the entire clip; it is not real-time detection or a separate prediction for each frame.

The `example_videos/input/` folder contains ten NW-UCLA clips covering the supported actions; `example_videos/output/` contains the actual demo predictions for them. See `example_videos/README.md` for the dataset action and predicted action for each clip. The examples are selected for demonstration and are not an accuracy evaluation.

The included sample is an NW-UCLA video for action a01, pick up with one hand. A local inference test predicted that action. Predictions on unfamiliar videos may be wrong.

## Files to submit

Include this entire `04_Project_Demo` folder and `../12_Trained_Models/refit_checkpoint.pt`. The checkpoint is required for inference; it is the recorded Kaggle Inception-v3 run, not a verified checkpoint for the reported 78% local run. If the instructor requires a recorded demonstration, record the preview, classification and result; no recording is included here.

For another computer, install Python with Tkinter, run `pip install -r requirements.txt`, then `python demo.py`. Keep the checkpoint in the adjacent `12_Trained_Models` folder. The launcher uses this computer's Miniconda `matsya` environment and local `.model_test_deps` directory when available, and otherwise tries the Python on your PATH. The code samples four evenly spaced frames and passes the same clip through both branches, matching the recorded cross-view test protocol.

For a screen recording, open this demo, display the sample video, click Classify and capture the resulting prediction in MP4 or AVI.
