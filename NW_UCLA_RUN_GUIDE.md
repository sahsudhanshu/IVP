# NW-UCLA baseline: running and verification

`ivp_implementation_nw_ucla_tsn_gru_base_paper.py` is a Kaggle script, not an updated Kaggle notebook. Copy its contents to the implementation notebook. Attach the RGB dataset. The implementation needs `pretrainedmodels==0.7.4` in addition to PyTorch, OpenCV, pandas, scikit-learn, matplotlib and seaborn. In a Kaggle code cell, install it if necessary:

```python
%pip install pretrainedmodels==0.7.4
```

The BN-Inception ImageNet weights are downloaded by the package on first use; this needs Kaggle internet access. If the package's weight host is unavailable, supply its compatible original ImageNet state dict as a Kaggle input and set `BN_WEIGHTS` to its absolute path. No random initialization fallback is used.

Set environment variables before running the script. `DATA_ROOT` is optional when exactly one input contains `multiview_action_videos`; it otherwise points directly to that directory. `OUTPUT_DIR` defaults to `/kaggle/working/basepaper_nwucla_results`. `EPOCHS` defaults to 80 and the paper's learning-rate drops are at epochs 30 and 60. To limit a first smoke run to one epoch and one cross-view split:

```python
import os
os.environ['EPOCHS'] = '1'  # smoke test only
os.environ['EXPERIMENT'] = 'cross_view_{1,2}_to_3'
# Run the script cell after setting these variables.
```

For development, set `VAL_SUBJECT` (for example `10`) and choose one experiment. That subject is held out from the source training data; the outer test data is not evaluated in development mode. For the final fixed configuration unset `VAL_SUBJECT`, set `EPOCHS` to 80, and run each split. Leave `EXPERIMENT` unset or set to `all` to run 10 cross-subject and three cross-view experiments. This can exceed one Kaggle session. To resume a stopped run set `RESUME=1` and use the same output directory, data paths, experiment and settings. Resume files contain optimizer and random states and should be loaded only if you trust their source. Persist output files between Kaggle sessions; `/kaggle/working` alone is not permanent across sessions.

The script saves inventories, file-list audit, split manifests, predictions, model weights and per-epoch resume files. Original archive inspection found 1,475 AVIs and no differences between the shipped `videos.txt` lists and AVIs. This only establishes consistency with the lists inside that archive: it does not prove those lists match the original download or the paper's sample set. The paper reports 1,745 RGB videos. Compare source checksums/manifests or obtain the original paper split before claiming identical data.

Views are paired by matching action, subject and event identifiers. The cameras were recorded simultaneously, but these filenames have not been independently verified as synchronized instances. Audit matched examples against original dataset documentation or recordings before interpreting multi-view accuracy. Cross-view uses paired source views and every available target-view clip; it is not a subject-independent test because the same subjects can occur in both views. The ten cross-subject runs use complete three-view groups and held-out subjects.

The paper does not give every architectural dimension, and 256 refinement channels, 128 GRU units, four segments, batch size 2, and deterministic center crop remain explicit implementation choices. The center crop is preprocessing, with no augmentation added. Changing crop, segments, batch, or architecture requires development on training/validation data, not repeated selection on the held-out test set. A fixed 80 epochs is not guaranteed to achieve the paper's results.
