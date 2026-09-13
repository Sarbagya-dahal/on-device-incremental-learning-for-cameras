"""
test_all.py — Phase 2 comprehensive test script
Run with: py -3 test_all.py

Tests:
  1. config.py
  2. simulator.py
  3. self_labeling.py + buffer
  4. drift_detection.py
  5. training_simulator.py (synthetic path)
  6. Buffer gate
  7. camera_capture.py (replay mode — no physical camera required)
  8. feature_extractor.py (synthetic frame — no webcam required)
  9. object_detector.py (MockDetector — no YOLOv8 download required)
  10. incremental_trainer.py (real SGDClassifier on synthetic buffer)
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import config
from simulator import generate_frames
from self_labeling import PseudoLabelBuffer
from drift_detection import (
    compute_feature_shift, UncertaintyTracker,
    TrackerStabilityMonitor, compute_composite_score, SustainedDriftChecker
)
from training_simulator import simulate_training

errors = []

def check(cond, msg):
    if cond:
        print(f"  [OK]  {msg}")
    else:
        print(f"  [FAIL] {msg}")
        errors.append(msg)

print("=" * 60)
print("  On-Device Learning — Phase 2 Test Suite")
print("=" * 60)

# ── 1. Config ──────────────────────────────────────────────────────────────────
print("\n[1] config.py")
check(len(config.SCENARIOS) == 4,            "4 scenarios defined")
check(len(config.OBJECT_CLASSES) == 5,       "5 indoor object classes (Phase 2)")
check(config.FEATURE_DIM == 1280,            "FEATURE_DIM = 1280 (MobileNetV2)")
check(abs(config.WEIGHT_FEATURE_SHIFT + config.WEIGHT_UNCERTAINTY + config.WEIGHT_TRACKER_DRIFT - 1.0) < 1e-6,
      "Default fusion weights sum to 1.0 (unchanged)")
check(hasattr(config, "CAMERA_INDEX"),       "CAMERA_INDEX defined")
check(hasattr(config, "YOLO_MODEL_NAME"),    "YOLO_MODEL_NAME defined")
check(hasattr(config, "INCREMENTAL_EPOCHS"), "INCREMENTAL_EPOCHS defined")
check(hasattr(config, "VALIDATION_HOLDOUT_FRAC"), "VALIDATION_HOLDOUT_FRAC defined")
for cls in config.OBJECT_CLASSES:
    check(cls in config.CLASS_CONFIDENCE_THRESHOLDS,
          f"Threshold defined for class '{cls}'")

# ── 2. Simulator (still used in Synthetic Demo mode) ──────────────────────────
print("\n[2] simulator.py (Synthetic Demo)")
for scenario in config.SCENARIOS:
    frames = generate_frames(scenario, 100, 42)
    check(len(frames) == 100,                    f"{scenario}: 100 frames generated")
    check(all('detections' in f for f in frames), f"{scenario}: all frames have detections key")
    check(all(len(f['detections']) > 0 for f in frames), f"{scenario}: all frames have >=1 detection")
    sample_feat = frames[0]['detections'][0]['features']
    # Simulator uses its own internal 16-dim synthetic vectors (not config.FEATURE_DIM=1280)
    SIM_DIM = len(sample_feat)   # should be 16
    check(SIM_DIM == 16, f"{scenario}: simulator produces 16-dim synthetic features (internal dim)")

# ── 3. Self-labeling & buffer ──────────────────────────────────────────────────
print("\n[3] self_labeling.py")
buf = PseudoLabelBuffer()
normal_frames = generate_frames('Normal Environment', 50, 42)
for f in normal_frames:
    for det in f['detections']:
        buf.try_add(f['frame'], det)
stats = buf.get_stats()
check(stats['buffer_size'] <= config.MAX_BUFFER_SIZE, "Buffer never exceeds max size")
check(stats['total_accepted'] > 0,           "Some samples accepted")
check(stats['acceptance_rate'] > 0,          "Acceptance rate > 0")
check(buf.get_recent_labels(5).__len__() <= 5, "get_recent_labels respects n")

buf2 = PseudoLabelBuffer(max_size=10)
long_frames = generate_frames('Normal Environment', 200, 1)
for f in long_frames:
    for det in f['detections']:
        buf2.try_add(f['frame'], det)
check(buf2.size <= 10, "Buffer maxlen enforced (FIFO eviction)")

# ── 4. Drift detection ─────────────────────────────────────────────────────────
print("\n[4] drift_detection.py")
baseline_feats = []
for bf in generate_frames('Normal Environment', 20, 42):
    for det in bf['detections']:
        baseline_feats.extend(det['features'])
bl_mean = float(np.mean(baseline_feats))
bl_std  = float(np.std(baseline_feats)) if float(np.std(baseline_feats)) > 0 else 1.0

def run_scenario(scenario_name, seed=42):
    frames = generate_frames(scenario_name, 100, seed)
    fw = []; unc = UncertaintyTracker(); trk = TrackerStabilityMonitor()
    composites = []
    for f in frames:
        for det in f['detections']:
            unc.update(det['confidence'])
            fw.extend(det['features'])
        if len(fw) > 240:
            fw = fw[-240:]
        chunked = [fw[i:i+16] for i in range(0, len(fw), 16)
                   if len(fw[i:i+16]) == 16]
        fs = compute_feature_shift(chunked, bl_mean, bl_std)
        ts = trk.update(f['detections'])
        td = 1.0 - ts
        uc = unc.score()
        cp = compute_composite_score(fs, uc, td)
        composites.append(cp)
    return composites

c_normal = run_scenario('Normal Environment')
c_light  = run_scenario('Lighting Change')
c_severe = run_scenario('Severe Distribution Shift')

avg_normal = float(np.mean(c_normal[-20:]))
avg_light  = float(np.mean(c_light[-20:]))
avg_severe = float(np.mean(c_severe[-20:]))

check(avg_severe > avg_normal,  f"Severe ({avg_severe:.3f}) > Normal ({avg_normal:.3f})")
check(avg_light  > avg_normal,  f"Lighting ({avg_light:.3f}) > Normal ({avg_normal:.3f})")
check(avg_normal < 0.65,        f"Normal avg composite {avg_normal:.3f} stays below threshold")
check(all(0.0 <= v <= 1.0 for v in c_severe), "All composite scores in [0,1]")

dc_severe = SustainedDriftChecker(threshold=0.65, sustained_frames=5)
triggered_severe = any(dc_severe.update(cp)[0] for cp in c_severe)
check(triggered_severe, "Severe scenario triggers sustained drift")

dc_normal = SustainedDriftChecker(threshold=0.65, sustained_frames=5)
triggered_normal = any(dc_normal.update(cp)[0] for cp in c_normal)
check(not triggered_normal, "Normal scenario does NOT trigger sustained drift")

# ── 5. training_simulator.py (synthetic fallback path) ────────────────────────
print("\n[5] training_simulator.py (synthetic fallback)")
result = simulate_training(
    buffer_size=50, current_version='v1.0',
    current_accuracy=0.862, seed=42
)
check(result['candidate_version'] == 'v1.1',      "Candidate version incremented correctly")
check(0.0 < result['candidate_accuracy'] < 1.0,   "Candidate accuracy in valid range")
check(len(result['epoch_losses']) == 5,            "5 epochs simulated by default")
check(result['accepted'] == (result['candidate_accuracy'] > result['current_accuracy']),
      "Accepted iff candidate > current")

result_rb = simulate_training(
    buffer_size=50, current_version='v1.0',
    current_accuracy=0.862, seed=42, force_rollback=True
)
check(not result_rb['accepted'],                   "Force-rollback: candidate rejected")

result2 = simulate_training(
    buffer_size=50, current_version='v1.1',
    current_accuracy=0.88, seed=10
)
check(result2['candidate_version'] == 'v1.2', "Version chains correctly (v1.1 -> v1.2)")

# ── 6. Buffer gate ─────────────────────────────────────────────────────────────
print("\n[6] Buffer gate")
small_buf = PseudoLabelBuffer(min_size=30)
tiny_frames = generate_frames('Severe Distribution Shift', 5, 42)
for f in tiny_frames:
    for det in f['detections']:
        small_buf.try_add(f['frame'], det)
check(not small_buf.is_qualified, f"Buffer with {small_buf.size} samples is NOT qualified (min={small_buf.min_size})")

large_buf = PseudoLabelBuffer(min_size=30)
many_frames = generate_frames('Normal Environment', 200, 42)
for f in many_frames:
    for det in f['detections']:
        large_buf.try_add(f['frame'], det)
check(large_buf.is_qualified, f"Buffer with {large_buf.size} samples IS qualified (min={large_buf.min_size})")

# ── 7. camera_capture.py (replay mode — no physical camera) ───────────────────
print("\n[7] camera_capture.py (replay mode)")
FIXTURE = "tests/fixtures/replay_clip.avi"
if not __import__('os').path.exists(FIXTURE):
    print("  [SKIP] Fixture not found — run: py -3 scripts/create_test_fixture.py")
else:
    try:
        from camera_capture import CameraCapture
        cap = CameraCapture(source=FIXTURE)
        opened = cap.open()
        check(opened, "CameraCapture opens replay video")
        if opened:
            ok, frame = cap.read()
            check(ok, "CameraCapture.read() returns ok=True")
            check(frame is not None and frame.ndim == 3, "Frame is H×W×3 array")
            check(cap.is_live == False, "is_live=False in replay mode")
            check(cap.mode == "video_replay", "mode='video_replay'")
            cap.release()
    except ImportError as e:
        print(f"  [SKIP] camera_capture import failed (cv2 missing?): {e}")

# ── 8. feature_extractor.py (synthetic crop — no webcam) ─────────────────────
print("\n[8] feature_extractor.py")
try:
    import cv2 as _cv2
    import numpy as _np
    from feature_extractor import FeatureExtractor
    fe = FeatureExtractor()
    # Synthetic 480×640 frame + bounding box
    fake_frame = _np.zeros((480, 640, 3), dtype=_np.uint8)
    fake_frame[100:300, 200:400] = 128   # grey rectangle
    bbox = [200, 100, 200, 200]
    feat = fe.extract(fake_frame, bbox)
    check(isinstance(feat, list), "extract() returns a list")
    check(len(feat) == config.FEATURE_DIM, f"Feature dim = {config.FEATURE_DIM}")
    check(all(isinstance(v, float) for v in feat), "All feature values are floats")
except ImportError as e:
    print(f"  [SKIP] feature_extractor unavailable (torch/torchvision missing?): {e}")

# ── 9. object_detector.py (MockDetector — no YOLOv8 download) ─────────────────
print("\n[9] object_detector.py (MockDetector)")
try:
    from object_detector import MockDetector, draw_detections
    mock = MockDetector(scenario='Severe Distribution Shift', num_frames=50, seed=42)
    dets = mock.detect()   # bgr_frame is optional for MockDetector
    check(isinstance(dets, list), "MockDetector.detect() returns a list")
    if dets:
        d = dets[0]
        check("object"     in d, "Detection has 'object' key")
        check("confidence" in d, "Detection has 'confidence' key")
        check("bbox"       in d, "Detection has 'bbox' key")
        check("features"   in d, "Detection has 'features' key")
        check(len(d["bbox"]) == 4,                "bbox has 4 elements [x,y,w,h]")
        check(len(d["features"]) == config.FEATURE_DIM,
              f"MockDetector pads features to FEATURE_DIM ({config.FEATURE_DIM})")
        check(0.0 <= d["confidence"] <= 1.0, "confidence in [0,1]")

    # draw_detections
    import numpy as _np
    blank = _np.zeros((480, 640, 3), dtype=_np.uint8)
    annotated = draw_detections(blank, dets)
    check(annotated.shape == blank.shape, "draw_detections returns same-shape frame")
except ImportError as e:
    print(f"  [SKIP] object_detector import failed: {e}")

# ── 10. incremental_trainer.py (real SGDClassifier on synthetic buffer) ────────
print("\n[10] incremental_trainer.py (real SGDClassifier)")
try:
    from incremental_trainer import IncrementalTrainer

    # Build a buffer with enough diverse samples
    train_buf = PseudoLabelBuffer(max_size=200, min_size=20)
    fill_frames = generate_frames('Normal Environment', 100, 42)
    for f in fill_frames:
        for det in f['detections']:
            # Pad features to FEATURE_DIM before inserting
            det2 = dict(det)
            feats = det2["features"]
            if len(feats) < config.FEATURE_DIM:
                rng2 = np.random.default_rng(42)
                det2["features"] = feats + rng2.normal(0, 0.1, config.FEATURE_DIM - len(feats)).tolist()
            # Need at least 2 classes — map objects to indoor classes
            det2["object"] = np.random.choice(config.OBJECT_CLASSES)
            train_buf.try_add(f['frame'], det2)

    trainer = IncrementalTrainer(classes=config.OBJECT_CLASSES)
    result = trainer.fit(
        buffer=train_buf,
        current_version='v1.0',
        current_accuracy=0.50,
        seed=42,
        num_epochs=3,
    )

    check("candidate_version"  in result, "result has 'candidate_version'")
    check("current_accuracy"   in result, "result has 'current_accuracy'")
    check("candidate_accuracy" in result, "result has 'candidate_accuracy'")
    check("accepted"           in result, "result has 'accepted'")
    check("logs"               in result, "result has 'logs'")
    check("epoch_losses"       in result, "result has 'epoch_losses'")
    check(0.0 <= result["candidate_accuracy"] <= 1.0, "candidate_accuracy in [0,1]")
    check(isinstance(result["accepted"], bool), "accepted is bool")
    check(len(result["logs"]) > 0, "Training produced log entries")

    # Force-rollback path
    result_rb = trainer.fit(
        buffer=train_buf,
        current_version='v1.0',
        current_accuracy=0.50,
        seed=42,
        force_rollback=True,
        num_epochs=3,
    )
    check(not result_rb["accepted"], "Force-rollback: candidate rejected")

except ImportError as e:
    print(f"  [SKIP] incremental_trainer import failed (sklearn missing?): {e}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"  FAILED: {len(errors)} test(s):")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  ALL TESTS PASSED")
print("=" * 60)
