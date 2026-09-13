"""
camera_capture.py
-----------------
Live webcam capture and replay/mock frame source.

WHAT IS REAL HERE (Phase 2)
  - Live mode: cv2.VideoCapture(CAMERA_INDEX) reads real BGR frames from the
    built-in laptop webcam in real time.
  - Replay mode (video file): replays a saved .avi/.mp4 file — used for offline
    demos and CI where no camera is present.
  - Replay mode (frame directory): reads sorted PNG/JPEG files from a folder,
    cycling back to frame 0 when the sequence ends.

WHAT THIS REPLACES IN THE REAL SYSTEM
  - Review 1: simulator.py's generate_frames() returned synthetic numpy arrays.
    There was no real I/O at all.
  - Phase 2: real OpenCV VideoCapture replaces the synthetic frame generator.
    The rest of the pipeline sees the same stream interface regardless of source.

[REAL for Phase 2] Live capture via OpenCV VideoCapture.
[REPLAY] Replay mode is fully deterministic — used in CI and offline demos.
"""

import os
import time
import glob
from pathlib import Path
from typing import Union

import cv2
import numpy as np

import config


class CameraCapture:
    """
    Unified frame source — live camera OR deterministic replay.

    Parameters
    ----------
    source : int | str | Path
        int           → live mode (webcam device index, e.g. 0)
        str / Path    → replay mode:
                          - path to a video file (.avi, .mp4, …)
                          - path to a directory of image frames (PNG/JPEG)

    fps_target : float
        Target frame rate for live mode. Replay mode ignores this.
    """

    def __init__(self,
                 source: Union[int, str, Path] = config.CAMERA_INDEX,
                 fps_target: float = config.LIVE_FPS_TARGET):
        self._source     = source
        self._fps_target = fps_target
        self._cap        = None          # cv2.VideoCapture (live / video replay)
        self._frame_paths: list[Path] = []   # for directory replay
        self._frame_idx  = 0             # cursor for directory replay
        self._last_read  = 0.0           # for FPS limiting
        self._opened     = False

        # Detect mode
        if isinstance(source, int):
            self._mode = "live"
        elif isinstance(source, (str, Path)) and os.path.isdir(str(source)):
            self._mode = "dir_replay"
        else:
            self._mode = "video_replay"

    # ── Public API ─────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open the capture source.

        Returns True on success, False if the device/file cannot be opened.
        """
        if self._mode in ("live", "video_replay"):
            self._cap = cv2.VideoCapture(self._source
                                         if self._mode == "live"
                                         else str(self._source))
            if not self._cap.isOpened():
                return False

        elif self._mode == "dir_replay":
            exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
            paths = []
            for ext in exts:
                paths.extend(glob.glob(os.path.join(str(self._source), ext)))
            self._frame_paths = sorted(Path(p) for p in paths)
            if not self._frame_paths:
                return False
            self._frame_idx = 0

        self._opened = True
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        """
        Grab the next frame.

        Returns
        -------
        (ok, frame_bgr)
            ok    : False if the source is exhausted or unavailable
            frame : BGR uint8 numpy array, or an empty array if ok=False
        """
        if not self._opened:
            return False, np.empty(0)

        if self._mode == "live":
            # FPS rate limiting
            elapsed = time.time() - self._last_read
            target_interval = 1.0 / max(self._fps_target, 1)
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            self._last_read = time.time()
            return self._cap.read()

        elif self._mode == "video_replay":
            ok, frame = self._cap.read()
            if not ok:
                # Loop back to the start
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            return ok, frame

        elif self._mode == "dir_replay":
            path = self._frame_paths[self._frame_idx % len(self._frame_paths)]
            self._frame_idx += 1
            frame = cv2.imread(str(path))
            if frame is None:
                return False, np.empty(0)
            return True, frame

        return False, np.empty(0)

    def release(self):
        """Release the underlying capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._opened = False

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        """True only when reading from a real webcam (not replay)."""
        return self._mode == "live"

    @property
    def mode(self) -> str:
        """One of 'live', 'video_replay', 'dir_replay'."""
        return self._mode

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of the capture. (0, 0) if not opened."""
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        if self._frame_paths:
            frame = cv2.imread(str(self._frame_paths[0]))
            if frame is not None:
                h, w = frame.shape[:2]
                return w, h
        return 0, 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    def __repr__(self):
        return f"CameraCapture(mode={self._mode!r}, source={self._source!r})"
