"""Check that a local SHWD copy matches the paper's image/label split."""

import argparse
import csv
import hashlib
from pathlib import Path

from experiment_paths import REPO_ROOT, path_for


MANIFEST = REPO_ROOT / "results" / "shwd_split_manifest.csv"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_rows(root):
    rows = []
    for split in SPLITS:
        images = sorted(
            (p for p in (root / "images" / split).iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda p: p.name,
        )
        for image in images:
            label = root / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(label)
            rows.append((split, image.name, sha256(image), sha256(label)))
        print(f"{split}: {len(images)} image/label pairs", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=path_for("shwd_root"))
    parser.add_argument("--write", action="store_true", help="Create the reference manifest")
    args = parser.parse_args()
    rows = dataset_rows(args.root.expanduser().resolve())
    if args.write:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("split", "image", "sha256_image", "sha256_label"))
            writer.writerows(rows)
        print(f"Wrote {MANIFEST}")
        return
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        expected = [tuple(row.values()) for row in csv.DictReader(stream)]
    if rows != expected:
        raise ValueError("SHWD split or file contents differ from the paper manifest")
    print("SHWD dataset matches the paper manifest.")


if __name__ == "__main__":
    main()
