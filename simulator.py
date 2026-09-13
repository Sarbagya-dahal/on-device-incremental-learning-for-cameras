"""
simulator.py
------------
Simulated inference engine for the On-Device Incremental Learning Simulator.

PHASE 2 NOTE
  This module is used ONLY in "Synthetic Demo" mode.  In Live Camera and
  Replay Recorded modes, real frames → object_detector.py → feature_extractor.py
  replace this entire module.

WHAT IS SIMULATED HERE
  - A physical embedded camera (Raspberry Pi / Jetson / IP-cam).
  - An on-device object detector (e.g. YOLOv8-nano, MobileNet-SSD).
  - The resulting per-frame detection stream.

WHAT THIS REPLACES IN THE REAL SYSTEM
  - Real camera       → camera_capture.py  (Phase 2)
  - Real detector     → object_detector.py  (Phase 2)
  - Real feature vecs → feature_extractor.py  (Phase 2)

Review 1: All data is synthetically generated using NumPy random distributions.
           The four scenarios inject drift in a controlled, reproducible way so
           that the drift-detection pipeline can be demonstrated clearly.

[SYNTHETIC] Used exclusively in Synthetic Demo mode.
"""

import numpy as np
from config import OBJECT_CLASSES

# Synthetic feature dimensionality — kept at 16 for the simulator to preserve
# the validated drift-detection behaviour from Review 1.
# Phase 2 real embeddings use config.FEATURE_DIM = 1280.
_SYNTHETIC_FEATURE_DIM = 16

# ── Bounding-box parameters per class (x, y, w, h) in a 640×480 frame ─────────
# Updated for Phase 2 indoor/desk classes.
CLASS_BBOX_CENTERS = {
    "person":     (320, 240, 80, 200),
    "cup":        (300, 300, 60,  80),
    "cell phone": (200, 260, 50,  90),
    "laptop":     (280, 250, 200, 130),
    "bottle":     (380, 290, 50, 120),
    # Legacy outdoor classes kept for backward compat with old saved data
    "car":        (300, 300, 200, 120),
    "bicycle":    (200, 260, 100, 120),
}

# ── Per-scenario baseline confidence ranges ────────────────────────────────────
SCENARIO_CONF_BASE = {
    "Normal Environment":       (0.88, 0.97),
    "Lighting Change":          (0.80, 0.95),
    "Background Change":        (0.75, 0.92),
    "Severe Distribution Shift":(0.55, 0.85),
}


def _sigmoid_ramp(frame: int, total: int, steepness: float = 10.0) -> float:
    """
    Smoothly ramps from 0 → 1 over the frame sequence.
    Produces a gradual S-curve so drift does NOT appear suddenly.
    """
    midpoint = total * 0.55
    x = (frame - midpoint) / (total * steepness / 100)
    return float(1.0 / (1.0 + np.exp(-x)))


def _linear_ramp(frame: int, total: int) -> float:
    """Linear 0 → 1 over the frame sequence."""
    return min(1.0, frame / max(total, 1))


def _bbox_jitter(center: tuple, rng: np.random.Generator, jitter_scale: float = 5.0) -> list:
    """
    Add Gaussian jitter to a bounding box centre to simulate natural movement.
    jitter_scale controls how much the box moves between frames.
    """
    x, y, w, h = center
    x += rng.normal(0, jitter_scale)
    y += rng.normal(0, jitter_scale)
    w += rng.normal(0, jitter_scale * 0.3)
    h += rng.normal(0, jitter_scale * 0.3)
    return [int(max(0, x)), int(max(0, y)), int(max(10, w)), int(max(10, h))]


def _feature_vector(rng: np.random.Generator,
                    mean_shift: float = 0.0,
                    noise_scale: float = 1.0) -> list:
    """
    Generate a _SYNTHETIC_FEATURE_DIM-dimensional synthetic feature vector.

    [SYNTHETIC] Zero-mean Gaussian vector with optional mean shift.
    Phase 2: Replace with feature_extractor.FeatureExtractor.extract() for real embeddings.

    mean_shift  – how far the distribution has moved from the baseline (0 = none)
    noise_scale – how much intra-class variance to simulate
    """
    base = rng.normal(0.0, noise_scale, _SYNTHETIC_FEATURE_DIM)
    shift_vec = np.ones(_SYNTHETIC_FEATURE_DIM) * mean_shift
    return (base + shift_vec).tolist()


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_frames(scenario: str, num_frames: int, seed: int) -> list[dict]:
    """
    Generate a list of simulated camera frames for a given scenario.

    Each frame dict contains:
        frame       – frame index (1-based)
        detections  – list of detection dicts (may be multiple per frame)
        scenario    – the scenario name (for downstream logging)

    Each detection dict contains:
        object      – class label
        confidence  – simulated detector confidence (0–1)
        bbox        – [x, y, w, h] bounding box in pixel coordinates
        features    – FEATURE_DIM-dim simulated feature vector

    The drift parameters for each scenario increase gradually over the frame
    sequence so that the drift-detection pipeline can observe the transition.
    """
    rng = np.random.default_rng(seed)
    frames = []

    conf_low, conf_high = SCENARIO_CONF_BASE.get(
        scenario, SCENARIO_CONF_BASE["Normal Environment"]
    )

    for frame_idx in range(1, num_frames + 1):
        progress = frame_idx / num_frames          # 0 → 1
        ramp     = _sigmoid_ramp(frame_idx, num_frames)   # smooth S-curve

        # ── Pick 1-3 objects for this frame ─────────────────────────────────
        n_objects = rng.integers(1, 4)
        classes   = rng.choice(OBJECT_CLASSES, n_objects, replace=True)

        detections = []
        for obj_class in classes:
            # ── Confidence ───────────────────────────────────────────────────
            if scenario == "Normal Environment":
                conf = rng.uniform(conf_low, conf_high)
                feature_shift = 0.0
                bbox_jitter   = 4.0

            elif scenario == "Lighting Change":
                # Confidence dips moderately; feature distribution shifts
                conf_penalty  = 0.15 * ramp
                conf          = rng.uniform(conf_low - conf_penalty,
                                            conf_high - conf_penalty)
                feature_shift = 1.2 * ramp          # up to ~1.2 sigma shift
                bbox_jitter   = 5.0                  # boxes stay mostly stable

            elif scenario == "Background Change":
                # Feature distribution changes noticeably; confidence moderate
                conf_penalty  = 0.12 * ramp
                conf          = rng.uniform(conf_low - conf_penalty,
                                            conf_high - conf_penalty)
                feature_shift = 1.8 * ramp           # larger feature shift
                bbox_jitter   = 7.0

            elif scenario == "Severe Distribution Shift":
                # Heavy drift on all signals
                conf_penalty  = 0.35 * ramp
                conf          = rng.uniform(max(0.3, conf_low - conf_penalty),
                                            conf_high - conf_penalty)
                feature_shift = 3.0 * ramp           # very large feature shift
                bbox_jitter   = 20.0 * ramp + 4.0   # unstable bounding boxes

            else:
                conf          = rng.uniform(conf_low, conf_high)
                feature_shift = 0.0
                bbox_jitter   = 4.0

            conf = float(np.clip(conf, 0.0, 1.0))
            center = CLASS_BBOX_CENTERS.get(obj_class, (320, 240, 100, 100))

            detections.append({
                "object":     obj_class,
                "confidence": round(conf, 4),
                "bbox":       _bbox_jitter(center, rng, bbox_jitter),
                "features":   _feature_vector(rng,
                                              mean_shift=feature_shift,
                                              noise_scale=1.0 + 0.5 * ramp
                                              if scenario == "Severe Distribution Shift"
                                              else 1.0),
            })

        frames.append({
            "frame":      frame_idx,
            "scenario":   scenario,
            "detections": detections,
        })

    return frames
