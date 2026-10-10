
import csv
import math
import random
import cv2

from pathlib import Path
from collections import Counter

# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = PROJECT_DIR / "frames_raw"
LABELS_DIR = PROJECT_DIR / "labels_yolo"
METADATA_CSV = PROJECT_DIR / "metadata" / "frames_metadata.csv"

REPORT_PATH = PROJECT_DIR / "metadata" / "dataset_qa_report.txt"
PREVIEW_DIR = PROJECT_DIR / "metadata" / "qa_previews"

CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "medium_vehicle",
}

EXPECTED_WIDTH = 3840
EXPECTED_HEIGHT = 2160
MAX_PREVIEWS = 12

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

errors = []
warnings = []
class_counts = Counter()
class_images = Counter()

total_images = 0
total_boxes = 0
empty_labels = 0
duplicate_boxes = 0
invalid_boxes = 0
dimension_mismatches = 0
images_with_boxes = 0

# ============================================================
# CHECK REQUIRED DIRECTORIES AND FILES
# ============================================================

for path in [IMAGES_DIR, LABELS_DIR, METADATA_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

image_paths = sorted(IMAGES_DIR.glob("frame_*.jpg"))

if not image_paths:
    raise RuntimeError(f"No frame_*.jpg images found in {IMAGES_DIR}")

print("=" * 65)
print("DRAGON LAKE DATASET QUALITY CHECK")
print("=" * 65)
print(f"Images directory: {IMAGES_DIR}")
print(f"Labels directory: {LABELS_DIR}")
print(f"Images found: {len(image_paths)}")

# ============================================================
# CHECK CSV METADATA
# ============================================================

metadata_by_filename = {}

with METADATA_CSV.open(
    "r", newline="", encoding="utf-8-sig"
) as file:
    reader = csv.DictReader(file)

    required = {"filename", "original_frame", "timestamp_seconds"}

    if not required.issubset(reader.fieldnames or []):
        errors.append(
            "CSV is missing one or more required columns: "
            + ", ".join(sorted(required))
        )
    else:
        for row in reader:
            filename = row["filename"]
            if filename in metadata_by_filename:
                errors.append(f"Duplicate CSV entry: {filename}")
            metadata_by_filename[filename] = row

image_names = {path.name for path in image_paths}

for name in sorted(image_names - set(metadata_by_filename)):
    warnings.append(f"Image missing from CSV: {name}")

for name in sorted(set(metadata_by_filename) - image_names):
    warnings.append(f"CSV references missing image: {name}")

# ============================================================
# CHECK IMAGE-LABEL PAIRS AND YOLO LABELS
# ============================================================

preview_candidates = []

for image_path in image_paths:
    total_images += 1
    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        errors.append(f"Missing label file: {label_path.name}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        errors.append(f"Unreadable image: {image_path.name}")
        continue

    height, width = image.shape[:2]

    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        dimension_mismatches += 1
        warnings.append(
            f"{image_path.name}: expected "
            f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, got {width}x{height}"
        )

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"Could not read {label_path.name}: {exc}")
        continue

    valid_boxes_for_image = []
    seen_lines = set()
    image_has_valid_box = False

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"{label_path.name}:{line_number}: "
                "expected 5 values, found "
                f"{len(parts)}"
            )
            invalid_boxes += 1
            continue

        try:
            class_value = float(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(
                f"{label_path.name}:{line_number}: non-numeric value"
            )
            invalid_boxes += 1
            continue

        if (
            not math.isfinite(class_value)
            or class_value != int(class_value)
            or int(class_value) not in CLASS_NAMES
        ):
            errors.append(
                f"{label_path.name}:{line_number}: "
                f"invalid class ID {parts[0]}"
            )
            invalid_boxes += 1
            continue

        if not all(math.isfinite(value) for value in coords):
            errors.append(
                f"{label_path.name}:{line_number}: "
                "coordinates contain NaN or infinity"
            )
            invalid_boxes += 1
            continue

        xc, yc, bw, bh = coords

        if bw <= 0 or bh <= 0:
            errors.append(
                f"{label_path.name}:{line_number}: "
                "box width/height must be positive"
            )
            invalid_boxes += 1
            continue

        if not all(0 <= value <= 1 for value in coords):
            errors.append(
                f"{label_path.name}:{line_number}: "
                "normalized value outside [0, 1]"
            )
            invalid_boxes += 1
            continue

        # Check that the box edges remain within the image.
        tolerance = 1e-5

        if (
            xc - bw / 2 < -tolerance
            or yc - bh / 2 < -tolerance
            or xc + bw / 2 > 1 + tolerance
            or yc + bh / 2 > 1 + tolerance
        ):
            errors.append(
                f"{label_path.name}:{line_number}: "
                "box extends outside image boundaries"
            )
            invalid_boxes += 1
            continue

        normalized_line = " ".join(
            [str(int(class_value))]
            + [f"{value:.8f}" for value in coords]
        )

        if normalized_line in seen_lines:
            duplicate_boxes += 1
            warnings.append(
                f"{label_path.name}:{line_number}: "
                "duplicate box in same image"
            )
        seen_lines.add(normalized_line)

        class_id = int(class_value)
        class_counts[class_id] += 1
        image_has_valid_box = True

        valid_boxes_for_image.append(
            (class_id, xc, yc, bw, bh)
        )

    total_boxes += len(valid_boxes_for_image)

    if image_has_valid_box:
        images_with_boxes += 1
        for class_id in set(
            box[0] for box in valid_boxes_for_image
        ):
            class_images[class_id] += 1
    else:
        empty_labels += 1

    preview_candidates.append(
        (image_path, valid_boxes_for_image)
    )

# ============================================================
# CREATE VISUAL PREVIEWS ACROSS THE WHOLE VIDEO
# ============================================================

# Select approximately evenly spaced images, not just the first frames.
if preview_candidates:
    count = min(MAX_PREVIEWS, len(preview_candidates))

    if count == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            round(i * (len(preview_candidates) - 1) / (count - 1))
            for i in range(count)
        ]

    for index in selected_indices:
        image_path, boxes = preview_candidates[index]
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        for class_id, xc, yc, bw, bh in boxes:
            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 255, 0), 3
            )

            cv2.putText(
                image,
                CLASS_NAMES[class_id],
                (x1, max(25, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Downsize only the preview, not the training image.
        preview = cv2.resize(image, (1920, 1080))

        output_path = PREVIEW_DIR / f"qa_{image_path.name}"
        cv2.imwrite(str(output_path), preview)

# ============================================================
# REPORT
# ============================================================

report = []

report.append("DRAGON LAKE DATASET QA REPORT")
report.append("=" * 50)
report.append(f"Images found: {len(image_paths)}")
report.append(f"Images checked: {total_images}")
report.append(f"Images with valid boxes: {images_with_boxes}")
report.append(f"Images with no valid boxes: {empty_labels}")
report.append(f"Total valid boxes: {total_boxes}")
report.append(f"Invalid boxes: {invalid_boxes}")
report.append(f"Duplicate boxes detected: {duplicate_boxes}")
report.append(f"Image dimension mismatches: {dimension_mismatches}")
report.append("")

report.append("CLASS DISTRIBUTION")
for class_id, name in CLASS_NAMES.items():
    report.append(
        f"{class_id} ({name}): {class_counts[class_id]} boxes "
        f"across {class_images[class_id]} images"
    )

report.append("")
report.append(f"Errors: {len(errors)}")
for error in errors[:200]:
    report.append(f"ERROR: {error}")

if len(errors) > 200:
    report.append(f"... {len(errors) - 200} additional errors omitted")

report.append("")
report.append(f"Warnings: {len(warnings)}")
for warning in warnings[:200]:
    report.append(f"WARNING: {warning}")

if len(warnings) > 200:
    report.append(
        f"... {len(warnings) - 200} additional warnings omitted"
    )

report.append("")
report.append(f"Visual previews: {PREVIEW_DIR}")

REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

print("\n" + "\n".join(report[:20]))
print(f"\nFull report saved to: {REPORT_PATH}")
print(f"Visual previews saved to: {PREVIEW_DIR}")

if errors:
    print("\nRESULT: ISSUES FOUND — fix errors before training.")
else:
    print("\nRESULT: AUTOMATED CHECKS PASSED.")
    print("Review warnings and visual previews before training.")
