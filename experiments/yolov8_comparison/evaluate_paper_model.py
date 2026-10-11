"""Evaluate a P2TaskYOLO checkpoint using the paper's SHWD settings."""

import argparse
from pathlib import Path

import ultralytics
from ultralytics.cfg import get_cfg

from experiment_paths import path_for, shwd_data
from p2_task_structure import P2TaskYOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="Combo best.pt checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--name", help="New output directory name")
    parser.add_argument("--check", action="store_true", help="Check paths and options only")
    args = parser.parse_args()

    if ultralytics.__version__ != "8.4.89":
        raise RuntimeError("This evaluation protocol requires Ultralytics 8.4.89")
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    data = shwd_data()
    if not data.is_file():
        raise FileNotFoundError(data)
    project = path_for("runs")
    name = args.name or f"paper_combo_{weights.parent.parent.name}_{args.split}"
    run = project / name
    if run.exists():
        raise FileExistsError(f"Evaluation directory already exists: {run}; choose another --name")
    options = dict(
        data=str(data), split=args.split, imgsz=640, batch=4, device=0,
        workers=4, conf=0.001, iou=0.6, max_det=300,
        quantize=32, rect=True, augment=False, agnostic_nms=False,
        plots=True, save_json=True, seed=0, deterministic=True,
        project=str(project), name=name, exist_ok=False,
    )
    get_cfg(overrides=options)
    print(f"Weights: {weights}\nData: {data}\nOutput: {run}", flush=True)
    if args.check:
        print("CHECK PASSED; no inference was run.")
        return
    model = P2TaskYOLO(str(weights), task="detect")
    if model.model.yaml.get("p2_task_design") != "adjacent_task_fusion_v1":
        raise ValueError("Checkpoint is not the paper combo architecture")
    metrics = model.val(**options)
    print(f"mAP50: {metrics.box.map50:.6f}")
    print(f"mAP50-95: {metrics.box.map:.6f}")
    print(f"Results: {metrics.save_dir}")


if __name__ == "__main__":
    main()
