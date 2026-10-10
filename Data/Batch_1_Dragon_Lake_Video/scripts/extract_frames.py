import cv2
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# This file is located in:
# dragon_lake_dataset/scripts/extract_frames.py

SCRIPT_DIR = Path(__file__).resolve().parent

# Go one level up:
# dragon_lake_dataset/
PROJECT_DIR = SCRIPT_DIR.parent

VIDEO_PATH = PROJECT_DIR / "video" / "dragon_lake_video_sample.MOV"
OUTPUT_DIR = PROJECT_DIR / "frames_raw"
METADATA_DIR = PROJECT_DIR / "metadata"


# ============================================================
# CONFIGURATION
# ============================================================

# Extract 2 frames per second
TARGET_FPS = 2


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK VIDEO
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )


print("=" * 60)
print("DRAGON LAKE FRAME EXTRACTION")
print("=" * 60)

print(f"Video: {VIDEO_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Target FPS: {TARGET_FPS}")

print("=" * 60)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

original_fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

duration_seconds = total_frames / original_fps


print()
print("VIDEO INFORMATION")
print("-" * 60)

print(f"Original FPS     : {original_fps}")
print(f"Total frames     : {total_frames}")
print(f"Resolution       : {width} x {height}")
print(f"Duration         : {duration_seconds:.2f} seconds")
print(f"Duration         : {duration_seconds / 60:.2f} minutes")

print("-" * 60)


# ============================================================
# FRAME EXTRACTION
# ============================================================

frame_interval = original_fps / TARGET_FPS

metadata = []

frame_number = 0
saved_frame_number = 0


with tqdm(
    total=total_frames,
    desc="Extracting frames",
    unit="frame"
) as progress:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Save frame according to target FPS
        if frame_number >= (
            saved_frame_number * frame_interval
        ):

            timestamp = frame_number / original_fps

            filename = (
                f"frame_{saved_frame_number:06d}.jpg"
            )

            output_path = OUTPUT_DIR / filename

            # Keep original 3840 × 2160 resolution
            success = cv2.imwrite(
                str(output_path),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

            if not success:
                print(
                    f"\nWarning: Could not save "
                    f"{filename}"
                )

            metadata.append({
                "frame_id": saved_frame_number,
                "filename": filename,
                "original_frame": frame_number,
                "timestamp_seconds": round(
                    timestamp,
                    3
                ),
                "timestamp_minutes": round(
                    timestamp / 60,
                    3
                )
            })

            saved_frame_number += 1

        frame_number += 1

        progress.update(1)


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()


# ============================================================
# SAVE METADATA
# ============================================================

metadata_df = pd.DataFrame(metadata)

metadata_path = (
    METADATA_DIR / "frames_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Frames extracted : {len(metadata_df)}"
)

print(
    f"Frames directory : {OUTPUT_DIR}"
)

print(
    f"Metadata file    : {metadata_path}"
)

if len(metadata_df) > 0:

    print(
        f"First timestamp  : "
        f"{metadata_df.iloc[0]['timestamp_seconds']} sec"
    )

    print(
        f"Last timestamp   : "
        f"{metadata_df.iloc[-1]['timestamp_seconds']} sec"
    )

print("=" * 60)