# Smart Parking Management System --- Batch 1 Data Preparation

## 1. Overview

This document summarizes the preparation of **Batch 1** for the Smart
Parking Management System. The batch uses a short Dragon Lake Parking
video sample to create a YOLO vehicle-detection dataset.

## 2. Source Data

-   **Video:** `dragon_lake_video_sample.MOV`
-   **Duration:** approximately 7 minutes 32 seconds
-   **Original resolution:** 3840 × 2160
-   **Frame rate:** 25 FPS
-   **Sampling rate:** 2 frames per second
-   **Extracted images:** 905

The original video and XML metadata are retained in the project for
reproducibility.

## 3. Frame Extraction

Frames were extracted using OpenCV with a target sampling rate of 2 FPS.
The images retain their original resolution.

A CSV file, `metadata/frames_metadata.csv`, records each extracted
image's filename, extracted-frame ID, original video frame number, and
timestamp.

## 4. Annotation Generation

The XML metadata contains frame-level object annotations, including
object types and image-space corner coordinates.

A Python script converts the four vehicle corner points into
axis-aligned YOLO bounding boxes and writes one `.txt` label file per
image.

Current classes:

    Class ID Class name
  ---------- ------------------
           0 `car`
           1 `bus`
           2 `medium_vehicle`

Pedestrians and `Undefined` objects are excluded from this
vehicle-detection dataset.

Each YOLO label line follows this format:

``` text
class_id x_center y_center width height
```

The coordinates are normalized to the image dimensions. The numeric
format is expected for YOLO labels.

## 5. Quality Checks

The automated dataset quality check reported:

  Check                                 Result
  ---------------------------------- ---------
  Images found and checked                 905
  Images with valid bounding boxes         905
  Total valid boxes                    194,274
  Invalid boxes                              0
  Duplicate label lines detected             0
  Image dimension mismatches                 0
  Errors                                     0
  Warnings                                   0

### Label counts

  Class              Labeled instances
  ---------------- -------------------
  Car                          190,612
  Bus                            1,810
  Medium vehicle                 1,852
  **Total**                **194,274**

These are labeled object instances across frames, not counts of unique
vehicles.

The generated previews were visually inspected, and the displayed
bounding boxes appeared to align well with the vehicles. Automated
checks validate label structure and geometry, but do not independently
prove that every annotation is semantically correct.

## 6. YOLO Dataset Structure

The images and labels were organized into training, validation, and test
splits:

``` text
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
└── split_manifest.csv
```

The preparation script uses a chronological split of approximately 70%
training, 15% validation, and 15% test. The manifest records each
image's split.

## 7. Main Scripts

-   `scripts/extract_frames.py` --- extracts video frames and writes the
    frame metadata CSV.
-   `scripts/inspect_metadata.py` --- inspects the XML annotation
    structure.
-   `scripts/generate_yolo_labels.py` --- converts XML vehicle
    annotations into YOLO label files and visual previews.
-   `scripts/check_dataset.py` --- checks dataset files, label format,
    coordinate ranges, class IDs, image dimensions, and duplicate label
    lines.
-   `scripts/prepare_yolo_dataset.py` --- creates the
    train/validation/test folder structure, `data.yaml`, and split
    manifest.


### Important limitations

-   This dataset trains vehicle detection and vehicle-type
    classification; it does not directly classify parking spaces as
    empty or occupied.
-   The images come from one video and one camera view, so performance
    on other parking lots is not yet established.
-   The class `medium_vehicle` should retain a consistent definition
    when future datasets are added.
-   Future batches must use compatible class mappings and annotations.
-   The test split should remain separate from training and model
    tuning.

