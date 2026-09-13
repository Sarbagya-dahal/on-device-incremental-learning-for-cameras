"""
scripts/create_test_fixture.py
-------------------------------
Generate a deterministic test fixture video (tests/fixtures/replay_clip.avi)
for CI-safe testing of the live pipeline without a physical camera.

The video contains 80 frames of a plain-colour background with a labelled
rectangle to simulate a detected object.  The simulator generates matching
synthetic detections that are embedded as frame metadata in a companion JSON.

Run once (or on CI):
    py -3 scripts/create_test_fixture.py
"""

import sys
import os
import json
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from simulator import generate_frames

OUT_VIDEO = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "replay_clip.avi")
OUT_JSON  = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "replay_detections.json")

W, H     = 640, 480
FPS      = 5
N_FRAMES = 80
SEED     = 42
SCENARIO = "Severe Distribution Shift"   # induces drift — good for integration tests

os.makedirs(os.path.dirname(OUT_VIDEO), exist_ok=True)

# Use MJPG for maximum compatibility (no codec dependencies)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (W, H))

rng     = np.random.default_rng(SEED)
frames  = generate_frames(SCENARIO, N_FRAMES, SEED)

all_detections = {}

for fd in frames:
    fnum = fd["frame"]
    dets = fd["detections"]

    # Draw a simple frame: dark bg + bounding box per detection
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (20, 24, 30)   # dark background matching app theme

    colors = {
        "person":     (74, 158, 107),
        "cup":        (62, 127, 206),
        "cell phone": (184, 135, 42),
        "laptop":     (192, 80, 64),
        "bottle":     (150, 90, 200),
        # legacy classes from old config (for compatibility during transition)
        "car":        (62, 127, 206),
        "bicycle":    (150, 90, 200),
    }

    for det in dets:
        x, y, w, h = det["bbox"]
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = max(10, min(w, W - x)); h = max(10, min(h, H - y))
        color = colors.get(det["object"], (200, 200, 200))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{det['object']} {det['confidence']:.2f}",
                    (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Frame index overlay
    cv2.putText(img, f"Frame {fnum}/{N_FRAMES}  {SCENARIO}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 110, 130), 1)

    writer.write(img)
    all_detections[fnum] = dets

writer.release()

with open(OUT_JSON, "w") as f:
    json.dump(all_detections, f, indent=2)

print(f"[OK] Wrote {N_FRAMES}-frame test fixture:")
print(f"     Video:      {os.path.abspath(OUT_VIDEO)}")
print(f"     Detections: {os.path.abspath(OUT_JSON)}")
print(f"     Scenario:   {SCENARIO}  |  Seed: {SEED}")
