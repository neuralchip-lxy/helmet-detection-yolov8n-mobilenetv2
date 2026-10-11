# YOLOv8n helmet detection experiments

Run the commands below from this directory. The final paper architecture is [the combo YAML](yolov8n_mobilenetv2_w0625_p2_task_combo.yaml), built by [P2TaskYOLO](p2_task_structure.py). A plain Ultralytics YOLO(yaml) call does **not** install the custom head.

## Data and environment

The [root path configuration](../../PATHS.md) has portable defaults. Place SHWD images and YOLO labels under the repository's SHWD_YOLO directory, or set shwd_root in an ignored experiment_paths.local.json. Outputs default to outputs/analysis_runs. An existing run directory blocks a new training run.

Reference environment: Windows, Python 3.11.15, PyTorch 2.11.0+cu130, Ultralytics 8.4.89, CUDA RTX 5060 Ti. Training uses scratch initialization, 640 input, batch 8, up to 200 epochs, patience 50, SGD, AMP, workers 4, seeds 0/1/2. The exact shared options are in [train_yolov8n_s0.py](train_yolov8n_s0.py).

## Commands

After installing the environment and dataset:

    python verify_shwd_manifest.py
    python train_yolov8n_mobilenetv2_p2_task_s0.py --check
    python train_yolov8n_mobilenetv2_p2_task_s0.py
    python train_p2_seed_repeat.py --variant combo --seed 1
    python train_p2_seed_repeat.py --variant combo --seed 2
    python evaluate_paper_model.py --weights PATH_TO_BEST_PT --split test

Use --resume only for an interrupted training run. The evaluation command does not train.

## Architecture and ablations

- [MobileNetV2 backbone](mobilenetv2_backbone.py) at width 0.625.
- [Shallow P2 output](shallow_detail.py), making four detection scales P2/P3/P4/P5 at strides 4/8/16/32.
- [Task-specific adjacent-scale fusion](p2_task_structure.py) in P2/P3 regression and classification towers; P4/P5 use separate ordinary towers.
- [Paper-model architecture notes](ABLATION_P2_TASK_COMBO.md) and [local-light control](ABLATION_P2_LOCAL_LIGHT.md).

Other ABLATION_*.md files and YAMLs record experiments that were not selected as the paper model. The research branch has later exploratory modules and video pilots; those results are not substituted into the paper metrics.

The [test-result table](../../results/shwd_test.csv) gives each seed. Validation selects best.pt; the held-out test split supplies the final reported metrics.
The [SHWD manifest](../../results/shwd_split_manifest.csv) records the exact image and label hashes for each split; `verify_shwd_manifest.py` compares a local dataset to it.
