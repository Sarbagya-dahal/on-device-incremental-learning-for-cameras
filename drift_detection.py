"""
drift_detection.py
------------------
Three complementary drift signals and their fusion into a composite drift score.

PHASE 2 STATUS
  [REAL] Feature-shift signal now receives real 1280-dim MobileNetV2 embeddings.
  [REAL] Uncertainty signal receives real YOLOv8n confidence scores.
  [REAL] Tracker stability receives real YOLOv8n bounding boxes.
  The formulas are UNCHANGED from Review 1 — they were already generalized
  over arbitrary feature dimensionality and arbitrary confidence streams.

WHAT REMAINS SIMPLIFIED (Phase 2)
  - Feature Shift:   Mean L2-distance from a scalar baseline, not full MMD.
  - Uncertainty:     Sliding-window slope, not Page-Hinkley/ADWIN.
  - Tracker:         Per-class IoU consistency, not a real SORT/ByteTrack tracker.

WHAT THIS REPLACES IN THE REAL SYSTEM
  - Feature Shift:   Maximum Mean Discrepancy (MMD) between baseline and live
                     feature distributions collected over a sliding window.
  - Uncertainty:     Page-Hinkley test or ADWIN on the per-frame confidence stream.
  - Tracker:         A real multi-object tracker (SORT, ByteTrack) providing
                     track-level ID consistency over time.

Phase 2 simplifications are clearly labelled with # [P2-SIMPLIFIED] comments.
"""

import numpy as np
from collections import deque
import config


# ── 1. Feature-Shift Signal ────────────────────────────────────────────────────

def compute_feature_shift(current_features: list[list],
                          baseline_mean: float = config.BASELINE_FEATURE_MEAN,
                          baseline_std:  float = config.BASELINE_FEATURE_STD,
                          clip_at: float = 4.0) -> float:
    """
    Compute a normalized feature-shift score in [0, 1].

    [REAL for Phase 2] Input is real 1280-dim MobileNetV2 embeddings.
    [P2-SIMPLIFIED] Still uses mean L2-distance from scalar baseline mean.
    Future: Replace with MMD or KL-divergence between a sliding window of live
            feature embeddings and the stored baseline feature distribution.

    Parameters
    ----------
    current_features : list of feature vectors (each FEATURE_DIM floats)
    baseline_mean    : expected mean of each feature dimension at baseline
    baseline_std     : expected std of each feature dimension at baseline
    clip_at          : number of standard deviations considered "maximum shift"
                       (distances beyond this are clipped to 1.0)

    Returns
    -------
    float in [0, 1]   0 = no shift, 1 = very large shift
    """
    if not current_features:
        return 0.0

    arr = np.array(current_features)          # shape (N, D)
    D = arr.shape[1]   # actual dimensionality of the input vectors
    # Standardize each feature dimension against the baseline distribution
    standardized = (arr - baseline_mean) / (baseline_std + 1e-8)
    # Mean L2 distance per sample, then average over the window.
    # Divide by sqrt(D) so the score is dimensionality-agnostic:
    # works for 16-dim synthetic vectors and 1280-dim real embeddings alike.
    per_sample_dist = np.linalg.norm(standardized, axis=1) / np.sqrt(D)
    mean_dist = float(np.mean(per_sample_dist))
    # Normalize: 0 → no shift, clip_at sigmas → 1.0
    score = np.clip(mean_dist / clip_at, 0.0, 1.0)
    return round(float(score), 4)


# ── 2. Uncertainty-Trend Signal ────────────────────────────────────────────────

class UncertaintyTracker:
    """
    Tracks model uncertainty via a sliding window of confidence scores.

    [REAL for Phase 2] Receives real YOLOv8n per-detection confidence scores.
    [P2-SIMPLIFIED] Combines low mean confidence and a downward trend slope.
    Future: Replace with Page-Hinkley cumulative sum or ADWIN adaptive windowing,
            which can detect subtle drift with formal statistical guarantees.
    """

    def __init__(self, window: int = config.UNCERTAINTY_WINDOW):
        self.window = window
        self._conf_window: deque[float] = deque(maxlen=window)

    def update(self, confidence: float):
        self._conf_window.append(confidence)

    def score(self) -> float:
        """
        Returns uncertainty in [0, 1].

        0 = high confidence, stable or rising trend  (no uncertainty)
        1 = very low confidence and/or steep downward trend (high uncertainty)
        """
        if len(self._conf_window) < 3:
            return 0.0

        confs = np.array(self._conf_window)

        # Component A: low mean confidence
        mean_conf   = float(np.mean(confs))
        # Map mean_conf from [0.5, 1.0] → uncertainty [1.0, 0.0]
        conf_score  = np.clip((1.0 - mean_conf) / 0.5, 0.0, 1.0)

        # Component B: downward trend (negative slope of confidence over window)
        # Fit a simple linear regression; a negative slope means falling confidence.
        x = np.arange(len(confs))
        slope = np.polyfit(x, confs, 1)[0]
        # Scale slope: a drop of 0.005/frame over the window counts as "maximum trend"
        trend_score = np.clip(-slope / 0.005, 0.0, 1.0)

        # Combine: weight mean slightly more than trend
        combined = 0.6 * conf_score + 0.4 * trend_score
        return round(float(np.clip(combined, 0.0, 1.0)), 4)

    def average_confidence(self) -> float:
        if not self._conf_window:
            return 1.0
        return round(float(np.mean(self._conf_window)), 4)


# ── 3. Tracker-Stability Signal ────────────────────────────────────────────────

def _iou(bbox_a: list, bbox_b: list) -> float:
    """
    Compute Intersection-over-Union between two [x, y, w, h] bounding boxes.
    Returns 0 if boxes are disjoint, 1 if identical.
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    # Convert to [x1, y1, x2, y2]
    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a_area = aw * ah
    b_area = bw * bh
    union_area = a_area + b_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class TrackerStabilityMonitor:
    """
    Monitors bounding-box consistency between consecutive frames for each class.

    [REAL for Phase 2] Receives real YOLOv8n bounding boxes.
    [P2-SIMPLIFIED] Compares per-class bounding boxes between frame t-1 and t.
    Future: Replace with a proper multi-object tracker (SORT / ByteTrack) that
            assigns persistent track IDs and computes track-level metrics.
    """

    def __init__(self, window: int = 10):
        self._prev_bboxes: dict[str, list]       = {}   # class → last bbox
        self._iou_history: deque[float]           = deque(maxlen=window)

    def update(self, detections: list[dict]) -> float:
        """
        Update with detections from the current frame.
        Returns current tracker stability in [0, 1].
        """
        current_bboxes: dict[str, list] = {}
        ious = []

        for det in detections:
            cls  = det["object"]
            bbox = det["bbox"]
            current_bboxes[cls] = bbox

            if cls in self._prev_bboxes:
                ious.append(_iou(self._prev_bboxes[cls], bbox))

        if ious:
            self._iou_history.append(float(np.mean(ious)))

        self._prev_bboxes = current_bboxes
        return self.stability()

    def stability(self) -> float:
        """
        Mean IoU over the recent history window.
        0 = bounding boxes jump randomly (unstable)
        1 = bounding boxes are perfectly consistent (stable)
        """
        if not self._iou_history:
            return 1.0   # assume stable until we have evidence
        return round(float(np.mean(self._iou_history)), 4)

    def tracker_drift(self) -> float:
        """tracker_drift = 1 − stability  (0 = stable, 1 = drifting)"""
        return round(1.0 - self.stability(), 4)


# ── 4. Multi-Signal Fusion ─────────────────────────────────────────────────────

def compute_composite_score(feature_shift: float,
                             uncertainty:   float,
                             tracker_drift: float,
                             w_fs: float = config.WEIGHT_FEATURE_SHIFT,
                             w_uc: float = config.WEIGHT_UNCERTAINTY,
                             w_td: float = config.WEIGHT_TRACKER_DRIFT) -> float:
    """
    Combine the three drift signals into a single composite drift score.

    Formula:
        composite = w_fs × feature_shift
                  + w_uc × uncertainty
                  + w_td × tracker_drift

    Weights should sum to 1.0; result is clipped to [0, 1].
    """
    composite = w_fs * feature_shift + w_uc * uncertainty + w_td * tracker_drift
    return round(float(np.clip(composite, 0.0, 1.0)), 4)


# ── 5. Sustained Drift Trigger ─────────────────────────────────────────────────

class SustainedDriftChecker:
    """
    Prevents spurious retraining by requiring the composite score to exceed
    DRIFT_THRESHOLD for at least SUSTAINED_FRAMES consecutive frames.

    This mirrors the "sustained drift" requirement described in the proposal.
    """

    def __init__(self,
                 threshold:       float = config.DRIFT_THRESHOLD,
                 sustained_frames: int  = config.SUSTAINED_FRAMES):
        self.threshold        = threshold
        self.sustained_frames = sustained_frames
        self._consecutive     = 0
        self.history: list[float] = []

    def update(self, composite_score: float) -> tuple[bool, int]:
        """
        Feed the latest composite score.

        Returns
        -------
        (threshold_exceeded_sustainably, consecutive_count)
        """
        self.history.append(composite_score)

        if composite_score >= self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0   # reset on any frame below threshold

        triggered = self._consecutive >= self.sustained_frames
        return triggered, self._consecutive

    def reset(self):
        self._consecutive = 0
        # Keep history for graph display but reset the streak counter
