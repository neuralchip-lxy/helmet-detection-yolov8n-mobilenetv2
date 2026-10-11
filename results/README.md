# Result provenance

[`shwd_test.csv`](shwd_test.csv) contains the held-out SHWD test metrics used in the repository root table. The baseline, local-light, and paper-combo rows were transcribed from the nine-checkpoint evaluation output (`test_nine_weights.py`, seeds 0, 1, and 2). The YOLO11n row comes from a separate seed-0 test using the same test protocol. This is a summary table, not raw predictions or a new evaluation run.

All entries used 640-pixel input, batch 4, confidence 0.001, NMS IoU 0.6, `max_det=300`, `rect=True`, and no test-time augmentation. The 1,517-image test split contains 24,436 instances; the class order is `hat`, `person`. Validation-set results were used to select each run's `best.pt` checkpoint. Means and sample standard deviations in the root README are computed from the three test rows per model. YOLO11n has one seed and therefore no standard deviation.

[`shwd_split_manifest.csv`](shwd_split_manifest.csv) lists each split's image filename and SHA-256 hashes of the image and its YOLO label file. It contains no image pixels or annotation contents. Run `python verify_shwd_manifest.py` from `experiments/yolov8_comparison` to check a local copy against this manifest.

The `parameters` and `gflops_640` columns are fused-model summaries printed by Ultralytics. GFLOPs are computation estimates; they are not measured execution times. Raw run outputs and checkpoints are deliberately excluded from Git because they are generated artifacts. A public checkpoint download should be added to the root README before announcing inference reproducibility.
