# Northwestern-UCLA / NUMA RGB benchmark

[Official dataset creator page](https://wangjiangb.github.io/my_data.html)

Use the “Northwestern-UCLA Multiview Action 3D Dataset” section and its RGB-only archive. The dataset has 10 action categories, 10 actors, and three camera viewpoints. Audit actual file totals and complete camera groups after download.

## Required files to add

- Real benchmark RGB clips or the instructor-approved test subset.
- Archive provenance and checksum.
- A verified class map and train/validation/test manifests.
- Exclusion log for unreadable or unpaired clips.

No dataset video or populated manifest is included in this proposal package.

## Protocol

Cross-subject: 10 outer subject folds, complete three-camera groups, action-level prediction. Use a source subject for validation, fix the configuration, and refit on the remaining nine before final testing.

Cross-view: {1,2} to 3, {1,3} to 2, {2,3} to 1. Use source-camera data for development. Refit on source training data and evaluate the complete target-camera split with fixed settings. At test time duplicate the one target-camera clip into both trained branches, following the primary base paper.

Report per-fold accuracy, macro-F1, support, three seeds, and both model-only and end-to-end latency. Keep raw prediction records. Full details are in SRS Sections 6–8.
