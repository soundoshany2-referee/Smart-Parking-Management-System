
import csv
import cv2
import xml.etree.ElementTree as ET

from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

XML_PATH = PROJECT_DIR / "metadata" / "dragon_lake_sample_metadata.xml"
CSV_PATH = PROJECT_DIR / "metadata" / "frames_metadata.csv"
IMAGES_DIR = PROJECT_DIR / "frames_raw"

LABELS_DIR = PROJECT_DIR / "labels_yolo"
PREVIEWS_DIR = PROJECT_DIR / "previews_yolo"

LABELS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160

# Class IDs must remain consistent throughout the project.
CLASS_MAP = {
    "Car": 0,
    "Bus": 1,
    "Medium Vehicle": 2,
}

CLASS_NAMES = {
    0: "car",
    1: "bus",
    2: "medium_vehicle",
}

# Number of images to save as visual previews.
MAX_PREVIEWS = 10


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

for path in (XML_PATH, CSV_PATH, IMAGES_DIR):
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

print("=" * 65)
print("DRAGON LAKE XML TO YOLO LABEL CONVERTER")
print("=" * 65)


# ============================================================
# READ EXTRACTED FRAME METADATA
# ============================================================

# Map each original video frame ID to its extracted image filename.
frame_to_image = {}

with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    required_columns = {"original_frame", "filename"}

    if not required_columns.issubset(reader.fieldnames or []):
        raise ValueError(
            "frames_metadata.csv must contain: "
            "original_frame and filename"
        )

    for row in reader:
        original_frame = int(row["original_frame"])
        filename = row["filename"]

        image_path = IMAGES_DIR / filename

        if image_path.exists():
            frame_to_image[original_frame] = filename
        else:
            print(f"Warning: Image not found: {image_path}")

if not frame_to_image:
    raise RuntimeError("No extracted images matched the CSV metadata.")

wanted_frames = set(frame_to_image)

print(f"Extracted images found: {len(frame_to_image)}")


# ============================================================
# CREATE EMPTY LABEL FILES
# ============================================================

# Every extracted image gets a label file, even if it contains
# no objects from the selected classes.

for filename in frame_to_image.values():
    label_path = LABELS_DIR / f"{Path(filename).stem}.txt"
    label_path.write_text("", encoding="utf-8")


# ============================================================
# CONVERT FOUR CORNERS TO A YOLO BOUNDING BOX
# ============================================================

def trajectory_to_yolo(obj):
    """
    Convert the four image-space corners from the XML into:
    class_id x_center y_center width height

    Coordinates are normalized to the image dimensions.
    """

    vehicle_type = obj.get("type")

    if vehicle_type not in CLASS_MAP:
        return None

    corner_names = [
        ("front_left_x", "front_left_y"),
        ("front_right_x", "front_right_y"),
        ("rear_left_x", "rear_left_y"),
        ("rear_right_x", "rear_right_y"),
    ]

    try:
        points = [
            (
                float(obj.attrib[x_name]),
                float(obj.attrib[y_name]),
            )
            for x_name, y_name in corner_names
        ]
    except (KeyError, ValueError):
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    # Use the original image coordinates, not the undistorted
    # coordinates, because our extracted images come from the
    # original video.
    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(IMAGE_WIDTH), max(xs))
    y_max = min(float(IMAGE_HEIGHT), max(ys))

    box_width = x_max - x_min
    box_height = y_max - y_min

    if box_width <= 1 or box_height <= 1:
        return None

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    # Normalize coordinates to the range 0-1.
    x_center /= IMAGE_WIDTH
    y_center /= IMAGE_HEIGHT
    box_width /= IMAGE_WIDTH
    box_height /= IMAGE_HEIGHT

    class_id = CLASS_MAP[vehicle_type]

    return (
        class_id,
        x_center,
        y_center,
        box_width,
        box_height,
    )


# ============================================================
# STREAM XML AND GENERATE LABELS
# ============================================================

print("Reading XML and matching original frame IDs...")
print("This may take some time for a large XML file.")

object_counts = Counter()
matched_frames = 0
matched_objects = 0
preview_count = 0

# Only retain annotations for the extracted frames we need.
# XML frames are processed incrementally.
for event, frame_element in ET.iterparse(XML_PATH, events=("end",)):

    if frame_element.tag != "frame":
        continue

    try:
        original_frame = int(frame_element.get("id"))
    except (TypeError, ValueError):
        frame_element.clear()
        continue

    if original_frame not in wanted_frames:
        frame_element.clear()
        continue

    filename = frame_to_image[original_frame]
    label_path = LABELS_DIR / f"{Path(filename).stem}.txt"

    yolo_lines = []

    for obj in frame_element.findall("trajectory"):
        vehicle_type = obj.get("type", "Unknown")

        yolo_box = trajectory_to_yolo(obj)

        if yolo_box is None:
            continue

        class_id, xc, yc, bw, bh = yolo_box

        yolo_lines.append(
            f"{class_id} {xc:.6f} {yc:.6f} "
            f"{bw:.6f} {bh:.6f}"
        )

        object_counts[vehicle_type] += 1
        matched_objects += 1

    # Write labels for this image.
    label_path.write_text(
        "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
        encoding="utf-8",
    )

    matched_frames += 1

    # Save previews for visual validation.
    if preview_count < MAX_PREVIEWS and yolo_lines:
        image_path = IMAGES_DIR / filename
        image = cv2.imread(str(image_path))

        if image is not None:
            for line in yolo_lines:
                values = line.split()

                class_id = int(values[0])
                xc, yc, bw, bh = map(float, values[1:])

                x1 = int((xc - bw / 2) * IMAGE_WIDTH)
                y1 = int((yc - bh / 2) * IMAGE_HEIGHT)
                x2 = int((xc + bw / 2) * IMAGE_WIDTH)
                y2 = int((yc + bh / 2) * IMAGE_HEIGHT)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3,
                )

                cv2.putText(
                    image,
                    CLASS_NAMES[class_id],
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            # Smaller previews are easier to open and inspect.
            preview = cv2.resize(image, (1920, 1080))

            preview_path = PREVIEWS_DIR / (
                f"preview_{Path(filename).stem}.jpg"
            )

            cv2.imwrite(preview_path.as_posix(), preview)
            preview_count += 1

    frame_element.clear()

    if matched_frames % 100 == 0:
        print(f"Matched {matched_frames} extracted frames...")


# ============================================================
# CHECK FRAME MATCHING
# ============================================================

missing_frames = wanted_frames - {
    int(path.stem.replace("frame_", ""))
    for path in LABELS_DIR.glob("frame_*.txt")
    if path.stat().st_size >= 0
}

# The label filenames correspond to extracted frame IDs, not
# original video frame IDs, so verify XML matching separately.
# Empty label files can be legitimate; use matched_frames above.

print("\n" + "=" * 65)
print("LABEL GENERATION SUMMARY")
print("=" * 65)

print(f"Extracted images       : {len(frame_to_image)}")
print(f"XML frames matched     : {matched_frames}")
print(f"Objects labeled        : {matched_objects}")
print(f"Preview images saved   : {preview_count}")
print(f"Labels directory       : {LABELS_DIR}")
print(f"Previews directory     : {PREVIEWS_DIR}")

print("\nLabeled object counts:")
for vehicle_type, count in sorted(object_counts.items()):
    print(f"  {vehicle_type}: {count}")

print("\nClass IDs:")
for class_id, class_name in CLASS_NAMES.items():
    print(f"  {class_id}: {class_name}")

if matched_frames != len(frame_to_image):
    print(
        "\nWARNING: Some extracted frames did not match an XML frame. "
        "Do not train until the mapping has been checked."
    )
else:
    print("\nAll extracted frames matched XML frame IDs.")

print("\nInspect the previews before training.")
print("=" * 65)
