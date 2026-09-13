"""
config.py
---------
Central configuration for the On-Device Incremental Learning prototype.

All tunable parameters live here so that:
  - The Streamlit sidebar can override them at runtime.
  - Future researchers can swap values without touching business logic.

Phase 2 changes from Review 1:
  - OBJECT_CLASSES re-scoped to indoor/desk COCO classes that a laptop webcam
    will actually detect (person, cup, cell phone, laptop, bottle).
  - FEATURE_DIM updated to 1280 — the actual output dimensionality of the
    MobileNetV2 penultimate layer used in feature_extractor.py.
  - New tunables added for camera, detector, and incremental-training control.
  - All drift-fusion weights (WEIGHT_*) and DRIFT_THRESHOLD / SUSTAINED_FRAMES
    are UNCHANGED — they are validated by test_all.py and test_integration.py.
"""

# ── Object classes ─────────────────────────────────────────────────────────────
# Phase 2: re-scoped to indoor/desk objects visible on a laptop webcam (COCO labels).
# Keep this list configurable so no other module needs to be changed when adding classes.
OBJECT_CLASSES = ["person", "cup", "cell phone", "laptop", "bottle"]

# ── Confidence thresholds for pseudo-label acceptance ─────────────────────────
# Per-class values allow different acceptance bars for easy vs. hard classes.
# Phase 2: thresholds tuned for YOLOv8n on indoor scenes.
# Review 1: These are hand-tuned defaults; future work will learn them adaptively.
CLASS_CONFIDENCE_THRESHOLDS = {
    "person":     0.70,
    "cup":        0.60,
    "cell phone": 0.65,
    "laptop":     0.65,
    "bottle":     0.60,
}

# ── Pseudo-label buffer ───────────────────────────────────────────────────────
MAX_BUFFER_SIZE = 200   # FIFO eviction once full  (larger for real detector)
MIN_BUFFER_SIZE = 30    # Retraining blocked until at least this many samples

# ── Drift detection ───────────────────────────────────────────────────────────
DRIFT_THRESHOLD   = 0.65   # Composite score must exceed this …
SUSTAINED_FRAMES  = 5      # … for this many consecutive frames to trigger retraining

# Sliding window for uncertainty signal
UNCERTAINTY_WINDOW = 20

# ── Multi-signal fusion weights ───────────────────────────────────────────────
# Must sum to 1.0 for the composite score to remain in [0, 1].
# DO NOT change without re-running test_all.py — these are validated and tuned.
WEIGHT_FEATURE_SHIFT = 0.40
WEIGHT_UNCERTAINTY   = 0.35
WEIGHT_TRACKER_DRIFT = 0.25

# ── Simulation defaults (used in Synthetic Demo mode) ─────────────────────────
DEFAULT_SCENARIO   = "Normal Environment"
DEFAULT_NUM_FRAMES = 100
DEFAULT_SEED       = 42

# ── Feature vector dimensionality ─────────────────────────────────────────────
# Phase 2: 1280-dim — MobileNetV2 penultimate layer (features + adaptive avg pool).
# Review 1 used 16-dim synthetic Gaussian embeddings.
FEATURE_DIM = 1280

# ── Baseline feature distribution (used for feature-shift calculation) ─────────
# Still scalar stats (mean / std across all dimensions).
# Phase 2: will be computed from the first N real frames of each session, but
#          the scalar-baseline formula in drift_detection.py accepts any FEATURE_DIM.
BASELINE_FEATURE_MEAN = 0.0
BASELINE_FEATURE_STD  = 1.0

# ── Scenario display names (Synthetic Demo mode) ─────────────────────────────
SCENARIOS = [
    "Normal Environment",
    "Lighting Change",
    "Background Change",
    "Severe Distribution Shift",
]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 additions — Live camera & real detector settings
# ─────────────────────────────────────────────────────────────────────────────

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0       # OpenCV device index for the built-in webcam
# Path to a video file or directory of frames for Replay mode.
# Empty string → use live camera when Live mode is selected.
REPLAY_VIDEO_PATH = ""

# ── Detector ──────────────────────────────────────────────────────────────────
# Minimum raw YOLOv8 confidence before the detection is even passed to
# self-labeling. Self-labeling applies CLASS_CONFIDENCE_THRESHOLDS on top.
DETECTOR_CONF_THRESHOLD = 0.35

# Inference resolution — smaller = faster CPU inference.
# 320 → ~200 ms/frame on a modern laptop CPU.
DETECTOR_IMG_SIZE = 320

# YOLO model name — "yolov8n.pt" is the nano variant (~6 MB, good for CPU).
YOLO_MODEL_NAME = "yolov8n.pt"

# ── Live loop ─────────────────────────────────────────────────────────────────
LIVE_FPS_TARGET = 5         # target frames per second in live mode
LIVE_LOOP_MAX_FRAMES = 500  # safety cap: auto-stop after this many frames

# ── Incremental trainer ───────────────────────────────────────────────────────
# Number of SGDClassifier partial_fit passes over the buffer each time retraining fires.
INCREMENTAL_EPOCHS = 10

# Fraction of buffer samples held out for candidate-model validation.
VALIDATION_HOLDOUT_FRAC = 0.20
