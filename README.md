# Lightweight helmet detection with MobileNetV2 and a four-scale YOLOv8n head

This repository contains a two-class helmet detector using the [SHWD dataset's](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) `hat` (helmet-wearing head) and `person` (head without a helmet) labels. The paper model replaces the YOLOv8n backbone with MobileNetV2 (width multiplier 0.625), adds a P2 output to the P3/P4/P5 pyramid, and uses task-specific adjacent-scale fusion in the P2/P3 detection head. It does not associate these head detections with full-body person boxes or track people.

The paper implementation consists of the [model YAML](experiments/yolov8_comparison/yolov8n_mobilenetv2_w0625_p2_task_combo.yaml), [custom head](experiments/yolov8_comparison/p2_task_structure.py), [backbone](experiments/yolov8_comparison/mobilenetv2_backbone.py), and [training entry point](experiments/yolov8_comparison/train_yolov8n_mobilenetv2_p2_task_s0.py). The [experiment guide](experiments/yolov8_comparison/README.md) explains the other ablations. The older [YOLOv5 implementation](legacy/README.md) is preserved under `legacy/` as historical work; it is not the paper model evaluated below.

## Results

These are SHWD **test** results at 640 input, batch 4, confidence 0.001, NMS IoU 0.6, and max_det 300. The test split has 1,517 images and 24,436 annotated instances. The [per-run values](results/shwd_test.csv) are the source for this table. Values are rounded to three decimals; ± is the sample standard deviation across three seeds.

| Model | Seeds | Parameters (M) | GFLOPs @ 640 | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| Original YOLOv8n | 0, 1, 2 | 3.006 | 8.1 | 0.928 ± 0.001 | 0.592 ± 0.001 |
| MobileNetV2 + P2 local-light head | 0, 1, 2 | 2.082 | 6.6 | 0.942 ± 0.001 | 0.604 ± 0.002 |
| **MobileNetV2 + P2 task-specific head (paper model)** | **0, 1, 2** | **2.085** | **6.5** | **0.944 ± 0.001** | **0.608 ± 0.001** |
| YOLO11n comparison | 0 only | 2.583 | 6.3 | 0.927 | 0.589 |

The paper model has higher test mAP50–95 and fewer parameters than the original YOLOv8n. Compared with YOLO11n it has higher accuracy and fewer parameters, but **0.2 more GFLOPs**. GFLOPs are model estimates, not measured FPGA or video latency. YOLO11n has only one seed and is not a three-seed comparison.

## Reproduce

1. Use Python 3.11 and a PyTorch build appropriate for your machine, then install `ultralytics==8.4.89`. The reported runs used Python 3.11.15, PyTorch 2.11.0+cu130, Ultralytics 8.4.89, 640 input, batch 8, SGD, up to 200 epochs, patience 50, and random initialization.
2. Obtain [the original SHWD dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) and convert its annotations to the YOLO labels used here; neither the images nor the converted labels are included in this repository. Place the fixed split under SHWD_YOLO/images/{train,val,test} and SHWD_YOLO/labels/{train,val,test}. Expected image counts are 5,457 / 607 / 1,517. Class indices are 0: hat and 1: person. See [the dataset YAML](SHWD_YOLO/shwd.yaml) and [the exact split/file hashes](results/shwd_split_manifest.csv).
3. If the dataset or output directory is elsewhere, copy [experiment_paths.example.json](experiment_paths.example.json) to experiment_paths.local.json and edit shwd_root and runs. This local file is ignored by Git. See [PATHS.md](PATHS.md).
4. From experiments/yolov8_comparison, run `python verify_shwd_manifest.py`, then run the paper training entry point with `--check` and without `--check` to train seed 0. See the [experiment guide](experiments/yolov8_comparison/README.md) for commands and the other seeds.
5. To evaluate a checkpoint, run the [evaluation entry point](experiments/yolov8_comparison/evaluate_paper_model.py) with `--weights PATH_TO_BEST_PT --split test`. It creates a new result directory and refuses to overwrite an existing one. **A stable public download for the trained checkpoints is still needed**; source code alone cannot reproduce inference.
