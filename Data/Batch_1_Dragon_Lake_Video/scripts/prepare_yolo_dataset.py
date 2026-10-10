
import csv
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = PROJECT_DIR / "frames_raw"
LABELS_DIR = PROJECT_DIR / "labels_yolo"
CSV_PATH = PROJECT_DIR / "metadata" / "frames_metadata.csv"

DATASET_DIR = PROJECT_DIR / "yolo_dataset"

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "medium_vehicle",
}


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    with CSV_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))

    rows.sort(key=lambda row: float(row["timestamp_seconds"]))

    if not rows:
        raise RuntimeError("No rows found in frames_metadata.csv")

    # Confirm all source images and labels exist before copying.
    for row in rows:
        image = IMAGES_DIR / row["filename"]
        label = LABELS_DIR / f"{Path(row['filename']).stem}.txt"

        if not image.exists():
            raise FileNotFoundError(f"Missing image: {image}")

        if not label.exists():
            raise FileNotFoundError(f"Missing label: {label}")

    total = len(rows)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_rows = {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }

    # Create split directories and copy files.
    # Original frames and labels remain untouched.
    for split, split_items in split_rows.items():
        images_out = DATASET_DIR / "images" / split
        labels_out = DATASET_DIR / "labels" / split

        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for row in split_items:
            filename = row["filename"]
            label_name = f"{Path(filename).stem}.txt"

            shutil.copy2(
                IMAGES_DIR / filename,
                images_out / filename,
            )

            shutil.copy2(
                LABELS_DIR / label_name,
                labels_out / label_name,
            )

    # Write Ultralytics YOLO configuration.
    yaml_path = DATASET_DIR / "data.yaml"

    yaml_lines = [
        f"path: {DATASET_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]

    for class_id, class_name in CLASS_NAMES.items():
        yaml_lines.append(f"  {class_id}: {class_name}")

    yaml_path.write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    # Save the split manifest for reproducibility.
    manifest_path = DATASET_DIR / "split_manifest.csv"

    with manifest_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [
            "filename",
            "original_frame",
            "timestamp_seconds",
            "split",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for split, split_items in split_rows.items():
            for row in split_items:
                writer.writerow({
                    "filename": row["filename"],
                    "original_frame": row["original_frame"],
                    "timestamp_seconds": row["timestamp_seconds"],
                    "split": split,
                })

    print("=" * 60)
    print("YOLO DATASET PREPARATION COMPLETE")
    print("=" * 60)

    for split, split_items in split_rows.items():
        print(f"{split}: {len(split_items)} images")

    print(f"\nDataset: {DATASET_DIR}")
    print(f"Configuration: {yaml_path}")
    print(f"Split manifest: {manifest_path}")
    print("\nOriginal images and labels were not modified.")


if __name__ == "__main__":
    main()
