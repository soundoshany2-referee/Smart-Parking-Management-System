# How to Generate the Dragon Lake YOLO Dataset

This guide explains how to use the five Python scripts in `scripts/` to
turn the authorized Dragon Lake video and its XML annotations into a
YOLO vehicle-detection dataset.

> **Important:** The commands below assume the folder structure shown
> here. The scripts may contain local paths from the machine where they
> were developed. Before running them, inspect each script's
> configuration and update its input/output paths to match your local
> project. Do not redistribute the Dragon Lake source data or derived
> files unless the dataset owners permit it.

## 1. Required folder structure

Run the commands from the repository root:

``` text
Smart-Parking-Management-System/
└── Data/
    └── Batch_1_Dragon_Lake_Video/
        ├── video/
        │   └── dragon_lake_video_sample.MOV
        ├── metadata/
        │   └── dragon_lake_sample_metadata.xml
        └── scripts/
            ├── extract_frames.py
            ├── inspect_metadata.py
            ├── generate_yolo_labels.py
            ├── check_dataset.py
            └── prepare_yolo_dataset.py
```

The scripts generate these additional folders/files during the workflow:

``` text
Batch_1_Dragon_Lake_Video/
├── frames_raw/                 # Extracted images
├── metadata/
│   └── frames_metadata.csv     # Extracted image ↔ source frame/timestamp mapping
├── labels_yolo/                # One YOLO .txt label file per image
├── previews_yolo/              # Images with boxes drawn for visual checking
└── yolo_dataset/               # Final training dataset
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

Place the source video and XML annotation file in the indicated folders.
If the XML filename differs, use the actual filename in the script
configuration.

## 2. Open the project root in PowerShell

For the clean clone used in this project:

``` powershell
cd "D:\IEEE\Smart Parking System\Smart-Parking-Clean"
```

Check that the scripts exist:

``` powershell
Get-ChildItem "Data\Batch_1_Dragon_Lake_Video\scripts"
```

Set a convenient variable for the folder:

``` powershell
$base = "Data\Batch_1_Dragon_Lake_Video"
```

## 3. Check and configure the script paths first

Open each `.py` file in VS Code and inspect its configuration section
near the top. Confirm that input and output paths point to the current
`Data\Batch_1_Dragon_Lake_Video` folder, rather than an old absolute
path such as a previous `D:\...` workspace.

Expected paths include:

-   Video input: `video\dragon_lake_video_sample.MOV`
-   XML annotations: `metadata\dragon_lake_sample_metadata.xml`
-   Extracted frames: `frames_raw\`
-   Frame metadata CSV: `metadata\frames_metadata.csv`
-   Generated labels: `labels_yolo\`
-   Visual previews: `previews_yolo\`
-   Final split dataset: `yolo_dataset\`

Create output directories if the scripts do not create them
automatically. Keep the class IDs consistent across the label-generation
script and `data.yaml`.

## 4. Run the scripts in this order

Run one command at a time. Wait for each script to finish and check its
output before moving to the next stage.

### Step 1 --- Inspect the source XML

``` powershell
python "$base\scripts\inspect_metadata.py"
```

Purpose: - Inspect the XML structure. - Check frame IDs, timestamps,
object types, and object coordinates. - Confirm that the expected
vehicle corner-coordinate fields exist.

If the script cannot find the XML, correct the configured XML path
inside `inspect_metadata.py`, save it, and run the command again.

### Step 2 --- Extract frames from the video

``` powershell
python "$base\scripts\extract_frames.py"
```

Purpose: - Read the source video using OpenCV. - Extract frames at the
sampling rate configured in the script. - Save the extracted images to
`frames_raw/`. - Write `metadata\frames_metadata.csv` with the image
filename, original video frame number, and timestamp information.

The recorded Batch 1 workflow used a target sampling rate of **2 frames
per second** and produced **905 images** from a sample video of about 7
minutes 32 seconds. Your output may differ if you change the video or
sampling rate.

If extraction fails, check that the video path is correct and that
OpenCV is installed:

``` powershell
python -m pip install opencv-python
```

### Step 3 --- Generate YOLO labels from XML

``` powershell
python "$base\scripts\generate_yolo_labels.py"
```

Purpose: - Match extracted images to their corresponding source frames
using the frame metadata. - Read vehicle annotations from the XML. -
Convert the four corner points of each included object into an
axis-aligned bounding box. - Normalize the box coordinates and write
YOLO `.txt` files under `labels_yolo/`. - Generate visual previews under
`previews_yolo/` if configured by the script.

The class mapping in the prepared Batch 1 dataset is:

    Class ID YOLO class name    XML type
  ---------- ------------------ ------------------
           0 `car`              `Car`
           1 `bus`              `Bus`
           2 `medium_vehicle`   `Medium Vehicle`

`Pedestrian` and `Undefined` objects were excluded from this
vehicle-detection dataset. Do not change the class mapping unless you
regenerate labels and update `data.yaml` consistently.

Each YOLO label line has this format:

``` text
class_id x_center y_center width height
```

The coordinates are normalized relative to the image width and height,
so the coordinate values should be between 0 and 1.

### Step 4 --- Validate the images and labels

``` powershell
python "$base\scripts\check_dataset.py"
```

Purpose: - Check whether images and labels exist. - Validate YOLO label
format and coordinate ranges. - Check class IDs, image dimensions,
invalid boxes, and duplicate label lines.

The recorded Batch 1 QA run checked 905 images and reported 194,274
valid object instances, zero invalid boxes, zero duplicate label lines,
zero dimension mismatches, zero errors, and zero warnings. Treat these
as the previous run's results---not a guarantee that every future run
will produce the same result.

Also inspect some images in `previews_yolo/` manually. A successful
automated check confirms file format and geometry checks, but does not
prove every annotation is semantically correct.

### Step 5 --- Create the train/validation/test dataset

``` powershell
python "$base\scripts\prepare_yolo_dataset.py"
```

Purpose: - Arrange images and labels into matching training, validation,
and test directories. - Generate `yolo_dataset\data.yaml`. - Generate
`yolo_dataset\split_manifest.csv`, recording which split each image
belongs to.

The recorded preparation workflow uses a chronological split of
approximately **70% train, 15% validation, and 15% test**. Keep the test
set separate from training and model tuning.

After the script completes, inspect the output:

``` powershell
Get-ChildItem "$base\yolo_dataset" -Recurse -File |
    Measure-Object
```

Check that image files and label files have matching basenames in each
split.

## 5. Final result

When all stages finish, the main file used to configure YOLO training
is:

``` text
Data\Batch_1_Dragon_Lake_Video\yolo_dataset\data.yaml
```

The prepared dataset should contain:

-   `images/train`, `images/val`, `images/test`
-   `labels/train`, `labels/val`, `labels/test`
-   `data.yaml`
-   `split_manifest.csv`

Use the generated `data.yaml` path when configuring the YOLO training
command or training code. Do not start training if the configuration
points to a path that does not exist.

## 6. Troubleshooting

### `FileNotFoundError`

Open the script and correct its configured video, XML, image, label, or
output path. Relative paths are interpreted according to the script's
current working directory, so run commands from the repository root as
shown above.

### No frames or labels were generated

Check the script's input paths and console output. For labels, confirm
that the XML frame IDs can be matched to the original frame numbers
recorded in `frames_metadata.csv`.

### YOLO reports invalid labels

Run `check_dataset.py`, inspect the reported filenames, and view their
corresponding previews. Verify class IDs, normalized coordinates, and
image/label basename matching.

### Dataset split folders are empty

Confirm that label generation and validation completed successfully and
that `prepare_yolo_dataset.py` is configured to read the correct source
image and label directories.

### Python package/import errors

Install only the dependencies actually imported by the scripts.
`opencv-python` is needed for the OpenCV frame-extraction workflow.
Check the other scripts' imports before creating a shared
`requirements.txt`.

## 7. Scope and limitations

This pipeline produces a **vehicle-detection dataset**. It does not
itself label parking slots as empty or occupied. Occupancy requires the
vehicle detections to be combined with parking-space regions and the
project's parking logic.

The current Batch 1 data comes from one video/camera view. Performance
on other locations and camera angles is not established. Keep the
`medium_vehicle` class definition consistent if adding other datasets
later.

## 8. Dataset licensing and GitHub

Keep the source video, XML annotations, extracted images, previews,
generated labels, and final YOLO dataset out of the public GitHub
repository unless the dataset owners explicitly authorize
redistribution. Publish the reusable scripts and documentation instead.
Teammates should obtain the data under the owners' terms and run this
pipeline locally.
