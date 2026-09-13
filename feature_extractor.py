"""
feature_extractor.py
--------------------
Real CNN feature-vector extraction from detected object crops.

WHAT IS REAL HERE (Phase 2)
  - A frozen MobileNetV2 backbone (ImageNet pretrained) is used to extract
    1280-dimensional penultimate-layer embeddings from each detected crop.
  - This replaces the 16-dim Gaussian synthetic vector from Review 1.

WHAT THIS REPLACES IN THE REAL SYSTEM
  - Review 1 [R1-SIMPLIFIED]: feature vectors were np.random.normal(0, 1, 16).
    The drift formula was correct; only the input was fake.
  - Phase 2 [REAL]: MobileNetV2 penultimate layer (features + GlobalAvgPool2d)
    produces a semantically meaningful 1280-dim embedding per crop.
  - Future: Replace with the detector backbone's own penultimate layer
    (YOLOv8's SPPF / C2f) to share computation. Currently uses a separate
    MobileNetV2 for simplicity.

[REAL for Phase 2] CNN-based feature extraction from real image crops.
[P2-SIMPLIFIED]   Separate backbone (not shared with the detector).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2

import config


# ── Preprocessing transform — ImageNet normalization ──────────────────────────
_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_MIN_CROP_PX = 10   # ignore tiny crops; return zero vector instead


class FeatureExtractor:
    """
    Extracts a 1280-dim feature vector from an image crop using MobileNetV2.

    The model is loaded once and kept in evaluation mode. All inference runs
    inside torch.no_grad() to avoid accumulating a computation graph.

    Parameters
    ----------
    device : str
        Torch device — 'cpu' always for the laptop prototype.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model  = self._build_model()

    def _build_model(self) -> nn.Module:
        """
        Build MobileNetV2 backbone with the classifier head removed.

        Output shape after GlobalAvgPool2d: (1, 1280, 1, 1) → squeezed to (1280,).
        """
        weights = models.MobileNet_V2_Weights.DEFAULT
        backbone = models.mobilenet_v2(weights=weights)

        # Remove the classifier head; keep features + adaptive avg pool
        extractor = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        extractor.eval()
        extractor.to(self._device)
        return extractor

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self,
                bgr_frame: np.ndarray,
                bbox: list) -> list:
        """
        Extract a 1280-dim feature vector from a detected crop.

        Parameters
        ----------
        bgr_frame : np.ndarray
            Full camera frame in BGR uint8 format (as returned by cv2).
        bbox : list
            [x, y, w, h] bounding box in pixel coordinates (may contain floats).

        Returns
        -------
        list of 1280 floats.
            [P2-SIMPLIFIED] Returns zero vector if the crop is too small (<10×10).
        """
        x, y, w, h = (int(v) for v in bbox)

        # Clamp to frame boundaries
        fh, fw = bgr_frame.shape[:2]
        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < _MIN_CROP_PX or crop_h < _MIN_CROP_PX:
            # [P2-SIMPLIFIED] Crop too small; return zero embedding
            return [0.0] * config.FEATURE_DIM

        crop_bgr = bgr_frame[y1:y2, x1:x2]
        # OpenCV is BGR; torchvision expects RGB
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        tensor = _TRANSFORM(crop_rgb).unsqueeze(0).to(self._device)  # (1, 3, 224, 224)

        with torch.no_grad():
            feat = self._model(tensor)   # (1, 1280, 1, 1)
            feat = feat.squeeze()        # (1280,)

        return feat.cpu().numpy().tolist()

    def extract_batch(self,
                      bgr_frame: np.ndarray,
                      bboxes: list[list]) -> list[list]:
        """
        Extract feature vectors for multiple bounding boxes in one call.
        Returns a list of 1280-dim lists, one per bbox.
        """
        return [self.extract(bgr_frame, bbox) for bbox in bboxes]
