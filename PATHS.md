# Dataset and output paths

The YOLOv8 experiment scripts resolve relative paths from the repository root, regardless of the shell's current directory. Defaults are in [`experiment_paths.example.json`](experiment_paths.example.json). The paper model uses `SHWD_YOLO/shwd.yaml` and writes generated files under `outputs/analysis_runs/`.

The [original SHWD dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) uses Pascal VOC annotations. Images and the converted YOLO labels for this experiment are not included in Git. Put the dataset in the following layout:

```text
SHWD_YOLO/
  shwd.yaml
  images/{train,val,test}/
  labels/{train,val,test}/
```

The checked-in YAML has `path: .`; the scripts make a runtime YAML with an absolute dataset root before invoking Ultralytics. This keeps the source YAML portable and avoids an output path being interpreted relative to an Ultralytics `runs/detect` directory.

If the dataset or output directory is elsewhere, make a local configuration file at the repository root:

```powershell
Copy-Item experiment_paths.example.json experiment_paths.local.json
```

On Command Prompt, use `copy experiment_paths.example.json experiment_paths.local.json`. Edit `shwd_root` to the directory containing `images/` and `labels/`, and `runs` to the desired results directory. Each value can be an absolute path or a path relative to the repository root. The local file is ignored by Git and should not be published. Other entries are for the separate SHEL5K experiments and can retain their defaults.

The scripts refuse to overwrite an existing run directory. A checkpoint from one machine can be evaluated on another by passing its path to [`evaluate_paper_model.py`](experiments/yolov8_comparison/evaluate_paper_model.py); the dataset and checkpoint must be obtained separately. For exact paper comparisons, use the same SHWD split, class order, and evaluation settings described in the [experiment guide](experiments/yolov8_comparison/README.md).
