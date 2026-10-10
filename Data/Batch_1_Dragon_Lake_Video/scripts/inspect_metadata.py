
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
XML_PATH = PROJECT_DIR / "metadata" / "dragon_lake_sample_metadata.xml"

frames_checked = 0
all_types = Counter()

print("Inspecting XML annotations...")

for event, element in ET.iterparse(XML_PATH, events=("end",)):
    if element.tag != "frame":
        continue

    frame_id = element.get("id")
    timestamp = element.get("timestamp")
    trajectories = element.findall("trajectory")

    frame_types = Counter(
        obj.get("type", "Unknown") for obj in trajectories
    )
    all_types.update(frame_types)

    print(
        f"Frame {frame_id} | timestamp={timestamp} | "
        f"objects={len(trajectories)} | types={dict(frame_types)}"
    )

    frames_checked += 1
    element.clear()

    if frames_checked >= 3:
        break

print("\nObject types found in inspected frames:")
print(dict(all_types))
