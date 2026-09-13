"""
self_labeling.py
----------------
Confidence-based pseudo-label filtering and bounded pseudo-label buffer.

PHASE 2 STATUS
  [REAL] Receives real YOLOv8n detector confidences and real MobileNetV2
         feature embeddings from object_detector.py.
  [REAL] Per-class confidence thresholds gate pseudo-label acceptance as designed.
  [REAL] FIFO buffer stores real (embedding, class) pairs for IncrementalTrainer.

WHAT THIS REPLACES IN THE REAL SYSTEM
  - Real self-labeling: compare detector confidence against adaptive class thresholds
    that may be learned from past validation performance.
  - Real buffer: memory-mapped storage on the device's eMMC/SD card.

Phase 2 simplifications:
  - Thresholds are fixed (configurable, but not learned online).
  - Buffer uses simple FIFO eviction; no reservoir sampling yet.
  - Diversity tracking is class-count only (no embedding-space diversity).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import config


@dataclass
class PseudoLabel:
    """One accepted pseudo-label entry in the buffer."""
    frame:      int
    obj_class:  str
    confidence: float
    bbox:       list
    features:   list


class PseudoLabelBuffer:
    """
    Bounded FIFO buffer for pseudo-labels.

    Only high-confidence detections are admitted.
    Retraining is blocked until the buffer reaches MIN_BUFFER_SIZE.

    Review 1: Simple FIFO. Future: Replace with diversity-aware reservoir
              sampling to avoid class collapse and redundant near-duplicate samples.
    """

    def __init__(self,
                 max_size: int = config.MAX_BUFFER_SIZE,
                 min_size: int = config.MIN_BUFFER_SIZE,
                 thresholds: Optional[dict] = None):
        self.max_size   = max_size
        self.min_size   = min_size
        self.thresholds = thresholds or dict(config.CLASS_CONFIDENCE_THRESHOLDS)
        self._buffer: deque[PseudoLabel] = deque(maxlen=max_size)

        # Running counters (reset when buffer resets)
        self.total_accepted = 0
        self.total_rejected = 0
        self._accepted_classes: dict[str, int] = {c: 0 for c in config.OBJECT_CLASSES}

    # ── Core API ───────────────────────────────────────────────────────────────

    def try_add(self, frame: int, detection: dict) -> tuple[bool, float]:
        """
        Attempt to add a detection to the buffer.

        Returns
        -------
        (accepted, threshold_used)
        """
        obj_class  = detection["object"]
        confidence = detection["confidence"]
        threshold  = self.thresholds.get(obj_class,
                                         config.CLASS_CONFIDENCE_THRESHOLDS.get(obj_class, 0.80))

        if confidence >= threshold:
            entry = PseudoLabel(
                frame=frame,
                obj_class=obj_class,
                confidence=confidence,
                bbox=detection["bbox"],
                features=detection["features"],
            )
            self._buffer.append(entry)
            self.total_accepted += 1
            self._accepted_classes[obj_class] = self._accepted_classes.get(obj_class, 0) + 1
            return True, threshold
        else:
            self.total_rejected += 1
            return False, threshold

    def reset(self):
        """Clear buffer and counters. Called when retraining completes."""
        self._buffer.clear()
        self.total_accepted = 0
        self.total_rejected = 0
        self._accepted_classes = {c: 0 for c in config.OBJECT_CLASSES}

    # ── Status queries ─────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_qualified(self) -> bool:
        """True when the buffer meets the minimum size for retraining."""
        return self.size >= self.min_size

    @property
    def acceptance_rate(self) -> float:
        total = self.total_accepted + self.total_rejected
        return self.total_accepted / total if total > 0 else 0.0

    def get_class_counts(self) -> dict[str, int]:
        """Count of each class currently in the buffer (not running total)."""
        counts = {c: 0 for c in config.OBJECT_CLASSES}
        for entry in self._buffer:
            counts[entry.obj_class] = counts.get(entry.obj_class, 0) + 1
        return counts

    def get_all_features(self) -> list[list]:
        """Return all feature vectors in the buffer (for drift calculation)."""
        return [entry.features for entry in self._buffer]

    def get_recent_labels(self, n: int = 20) -> list[dict]:
        """Return the n most recent entries as plain dicts for display."""
        items = list(self._buffer)[-n:]
        return [
            {
                "frame":      e.frame,
                "object":     e.obj_class,
                "confidence": e.confidence,
                "threshold":  self.thresholds.get(e.obj_class, 0.80),
                "status":     "ACCEPTED",
            }
            for e in reversed(items)
        ]

    def get_stats(self) -> dict:
        return {
            "buffer_size":     self.size,
            "max_size":        self.max_size,
            "min_size":        self.min_size,
            "is_qualified":    self.is_qualified,
            "total_accepted":  self.total_accepted,
            "total_rejected":  self.total_rejected,
            "acceptance_rate": round(self.acceptance_rate * 100, 1),
            "class_counts":    self.get_class_counts(),
        }
