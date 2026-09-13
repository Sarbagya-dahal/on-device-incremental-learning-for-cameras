"""
object_detector.py
------------------
Real on-device object detection via YOLOv8n + CNN feature extraction.

WHAT IS REAL HERE (Phase 2)
  - YOLOv8n (nano) runs on CPU to detect objects in each webcam frame.
  - Only OBJECT_CLASSES (indoor/desk COCO classes) are reported.
  - A MobileNetV2 FeatureExtractor computes real 1280-dim embeddings per crop.
  - Detection schema is identical to simulator.py so the rest of the pipeline
    (self_labeling, drift_detection, etc.) requires no changes.

WHAT THIS REPLACES IN THE REAL SYSTEM
  - Review 1 [R1-SIMPLIFIED]: simulator.py generated random bounding boxes and
    random Gaussian feature vectors from np.random.
  - Phase 2 [REAL]: YOLOv8n runs real inference; FeatureExtractor produces
    real semantic embeddings.
  - Future: Replace YOLOv8n with a quantized INT8 model on an NPU/DSP for
    embedded deployment.

MockDetector
  - Uses simulator.py's generate_frames() to produce frame dicts compatible
    with the detection schema.
  - Enables the full pipeline to run in CI, offline presentations, and the
    "Synthetic Demo" Streamlit mode without a camera or heavy model.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional

import config
from feature_extractor import FeatureExtractor

# Lazy import to avoid loading ultralytics at module level in mock-only usage
_YOLO_CLASS = None


def _get_yolo():
    """Import and return the YOLO class (lazy, so tests without ultralytics work)."""
    global _YOLO_CLASS
    if _YOLO_CLASS is None:
        from ultralytics import YOLO as _YOLO
        _YOLO_CLASS = _YOLO
    return _YOLO_CLASS


# ── COCO class name → index lookup ────────────────────────────────────────────
# Build the set of COCO class names we care about (for fast filtering).
_TARGET_CLASSES = set(config.OBJECT_CLASSES)


def _yolo_results_to_dets(results,
                           bgr_frame: np.ndarray,
                           extractor: FeatureExtractor) -> list[dict]:
    """
    Convert a ultralytics Results object to our internal detection schema.

    Each detection dict:
        object      : str   class name
        confidence  : float [0, 1]
        bbox        : [x, y, w, h]  (pixel coords, int)
        features    : list[float]   (1280-dim embedding)
    """
    detections = []
    if results is None or len(results) == 0:
        return detections

    result = results[0]   # single image

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    names = result.names  # {idx: 'class_name', …}

    for box in result.boxes:
        cls_idx  = int(box.cls[0].item())
        cls_name = names.get(cls_idx, "")
        if cls_name not in _TARGET_CLASSES:
            continue

        conf = float(box.conf[0].item())
        if conf < config.DETECTOR_CONF_THRESHOLD:
            continue

        # xyxy → x, y, w, h
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        bbox = [int(x1), int(y1), int(w), int(h)]

        features = extractor.extract(bgr_frame, bbox)

        detections.append({
            "object":     cls_name,
            "confidence": round(conf, 4),
            "bbox":       bbox,
            "features":   features,
        })

    return detections


# ── Real Detector ──────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Real YOLOv8n detector + MobileNetV2 feature extractor.

    [REAL for Phase 2] Replaces simulator.py's synthetic detection generator.

    Parameters
    ----------
    model_name : str
        YOLOv8 model variant.  "yolov8n.pt" is downloaded on first use (~6 MB).
    img_size : int
        Inference resolution.  Smaller = faster CPU, lower recall.
    device : str
        Torch device — always 'cpu' for the laptop prototype.
    """

    def __init__(self,
                 model_name: str = config.YOLO_MODEL_NAME,
                 img_size:   int = config.DETECTOR_IMG_SIZE,
                 device:     str = "cpu"):
        YOLO = _get_yolo()
        self._model    = YOLO(model_name)
        self._img_size = img_size
        self._device   = device
        self._extractor = FeatureExtractor(device=device)

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray   H×W×3 uint8

        Returns
        -------
        list of detection dicts (may be empty if nothing is found).
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        results = self._model.predict(
            source=bgr_frame,
            imgsz=self._img_size,
            device=self._device,
            verbose=False,
        )
        return _yolo_results_to_dets(results, bgr_frame, self._extractor)


# ── Mock Detector ──────────────────────────────────────────────────────────────

class MockDetector:
    """
    Offline / CI stand-in for ObjectDetector.

    Uses simulator.py's generate_frames() to produce detection dicts that are
    schema-compatible with ObjectDetector.detect().  Feature vectors are
    synthetic (16-dim Gaussian) up-padded to FEATURE_DIM so the drift
    formulas still run — this is acceptable in synthetic-demo / test mode.

    [SYNTHETIC] Not used in Live or Replay modes.
    """

    def __init__(self,
                 scenario:   str = config.DEFAULT_SCENARIO,
                 num_frames: int = config.DEFAULT_NUM_FRAMES,
                 seed:       int = config.DEFAULT_SEED):
        from simulator import generate_frames
        self._frames     = generate_frames(scenario, num_frames, seed)
        self._frame_iter = iter(self._frames)
        self._rng        = np.random.default_rng(seed)

    def detect(self, bgr_frame: Optional[np.ndarray] = None) -> list[dict]:
        """
        Return the next batch of synthetic detections.
        bgr_frame is accepted but ignored — detections come from the simulator.
        """
        try:
            fd = next(self._frame_iter)
        except StopIteration:
            return []

        dets = fd["detections"]

        # Pad 16-dim synthetic features to FEATURE_DIM to satisfy drift_detection
        if config.FEATURE_DIM > 16:
            for det in dets:
                feats = det["features"]
                if len(feats) < config.FEATURE_DIM:
                    # Repeat-pad with small Gaussian noise
                    pad_size = config.FEATURE_DIM - len(feats)
                    pad = self._rng.normal(0, 0.1, pad_size).tolist()
                    det["features"] = feats + pad
        return dets

    def reset(self, scenario: str, num_frames: int, seed: int):
        """Re-initialize with a new scenario."""
        from simulator import generate_frames
        self._frames     = generate_frames(scenario, num_frames, seed)
        self._frame_iter = iter(self._frames)


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of the frame.

    Returns the annotated BGR frame suitable for st.image() or cv2.imshow().
    """
    out = frame.copy()
    colors = {
        "person":     (74, 158, 107),   # green
        "cup":        (62, 127, 206),   # blue
        "cell phone": (184, 135, 42),   # amber
        "laptop":     (192, 80, 64),    # red
        "bottle":     (150, 90, 200),   # purple
    }
    for det in detections:
        x, y, w, h = det["bbox"]
        cls  = det["object"]
        conf = det["confidence"]
        color = colors.get(cls, (200, 200, 200))

        # Box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        # Label background
        label = f"{cls} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - lh - 6), (x + lw + 4, y), color, -1)
        cv2.putText(out, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out
