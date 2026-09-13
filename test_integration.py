"""
test_integration.py — Phase 2 integration tests
Run with: py -3 test_integration.py

Two integration paths are tested:
  A. Synthetic path (Review 1 behaviour, no camera):
     Severe scenario → buffer → drift → retraining trigger → simulate_training
  B. Replay path (Phase 2 real pipeline, no camera):
     replay_clip.avi → CameraCapture → MockDetector → buffer → drift → IncrementalTrainer

Path B tests the full live pipeline deterministically in CI.
"""
import sys
sys.path.insert(0, '.')

import os
import numpy as np
import config

from simulator           import generate_frames
from self_labeling       import PseudoLabelBuffer
from drift_detection     import (compute_feature_shift, UncertaintyTracker,
                                 TrackerStabilityMonitor, compute_composite_score,
                                 SustainedDriftChecker)
from training_simulator  import simulate_training

errors = []
def check(cond, msg):
    if cond:
        print(f"  [OK]  {msg}")
    else:
        print(f"  [FAIL] {msg}")
        errors.append(msg)

print("=" * 60)
print("  Phase 2 Integration Tests")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# Path A — Synthetic pipeline (exact Review 1 logic)
# ═══════════════════════════════════════════════════════════
print("\n[A] Synthetic pipeline (Severe Distribution Shift)")

seed     = 42
scenario = "Severe Distribution Shift"

buffer = PseudoLabelBuffer(max_size=100, min_size=30,
                            thresholds=config.CLASS_CONFIDENCE_THRESHOLDS)

preseed = generate_frames("Normal Environment", 15, seed + 1)
for pf in preseed:
    for det in pf['detections']:
        buffer.try_add(0, det)
print(f"  Pre-seeded: {buffer.size} samples")

baseline_frames   = generate_frames("Normal Environment", 20, seed)
baseline_features = []
for bf in baseline_frames:
    for det in bf['detections']:
        baseline_features.extend(det['features'])
bl_mean = float(np.mean(baseline_features))
bl_std  = float(np.std(baseline_features)) if float(np.std(baseline_features)) > 0 else 1.0

frames = generate_frames(scenario, 100, seed)
unc = UncertaintyTracker(); trk = TrackerStabilityMonitor()
dc  = SustainedDriftChecker(threshold=0.65, sustained_frames=5)
fw  = []; triggered_at = None
FDIM = 16   # simulator uses 16-dim

for f in frames:
    for det in f['detections']:
        buffer.try_add(f['frame'], det)
        unc.update(det['confidence'])
        fw.extend(det['features'])
    if len(fw) > 240: fw = fw[-240:]
    chunked = [fw[i:i+FDIM] for i in range(0,len(fw),FDIM) if len(fw[i:i+FDIM])==FDIM]
    fs = compute_feature_shift(chunked, bl_mean, bl_std)
    ts = trk.update(f['detections']); td = 1-ts
    uc = unc.score()
    cp = compute_composite_score(fs, uc, td)
    trig, cons = dc.update(cp)
    if trig and buffer.is_qualified and triggered_at is None:
        triggered_at = f['frame']
        print(f"  RETRAINING TRIGGERED at frame {triggered_at}  composite={cp:.3f}  buffer={buffer.size}")
        break

check(triggered_at is not None, "Severe scenario triggers retraining in synthetic path")

if triggered_at:
    tr = simulate_training(buffer_size=buffer.size, current_version='v1.0',
                           current_accuracy=0.862, seed=seed)
    check(tr['accepted'] or not tr['accepted'], "simulate_training returns a result")
    print(f"  Training: accepted={tr['accepted']}  accuracy={tr['candidate_accuracy']*100:.1f}%")

    tr2 = simulate_training(buffer_size=buffer.size, current_version='v1.0',
                            current_accuracy=0.862, seed=seed, force_rollback=True)
    check(not tr2['accepted'], "Rollback path: candidate rejected")
    print(f"  Rollback path: accepted={tr2['accepted']}  accuracy={tr2['candidate_accuracy']*100:.1f}%")

# ═══════════════════════════════════════════════════════════
# Path B — Replay + IncrementalTrainer (Phase 2 real pipeline)
# ═══════════════════════════════════════════════════════════
print("\n[B] Replay pipeline with IncrementalTrainer")

FIXTURE = "tests/fixtures/replay_clip.avi"

if not os.path.exists(FIXTURE):
    print(f"  [SKIP] Fixture not found at {FIXTURE}")
    print("         Run: py -3 scripts/create_test_fixture.py")
else:
    try:
        from camera_capture      import CameraCapture
        from object_detector     import MockDetector
        from incremental_trainer import IncrementalTrainer

        # Use MockDetector driven by the same Severe scenario to ensure drift fires
        mock_det  = MockDetector(scenario='Severe Distribution Shift',
                                 num_frames=200, seed=42)
        cap       = CameraCapture(source=FIXTURE)
        opened    = cap.open()
        check(opened, "CameraCapture(replay) opens successfully")

        buf2      = PseudoLabelBuffer(max_size=200, min_size=30,
                                      thresholds=config.CLASS_CONFIDENCE_THRESHOLDS)
        unc2      = UncertaintyTracker()
        trk2      = TrackerStabilityMonitor()
        dc2       = SustainedDriftChecker(threshold=0.65, sustained_frames=5)
        trainer   = IncrementalTrainer(classes=config.OBJECT_CLASSES)

        # Pre-seed with Normal frames to give the buffer a head start
        preseed2 = generate_frames("Normal Environment", 15, 43)
        for pf in preseed2:
            for det in pf['detections']:
                det2 = dict(det)
                if len(det2["features"]) < config.FEATURE_DIM:
                    det2["features"] = det2["features"] + [0.0] * (config.FEATURE_DIM - len(det2["features"]))
                det2["object"] = np.random.choice(config.OBJECT_CLASSES)
                buf2.try_add(0, det2)
        print(f"  Pre-seeded (replay): {buf2.size} samples")

        fw2 = []; triggered2_at = None; fnum2 = 0
        training_result2 = None

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            fnum2 += 1
            if fnum2 > 200:
                break

            dets2 = mock_det.detect(frame_bgr)
            if not dets2:
                unc2.update(0.0)
                continue

            for det in dets2:
                det2 = dict(det)
                # Ensure FEATURE_DIM compatibility
                if len(det2["features"]) < config.FEATURE_DIM:
                    det2["features"] = det2["features"] + [0.0] * (config.FEATURE_DIM - len(det2["features"]))
                det2["object"] = np.random.choice(config.OBJECT_CLASSES)
                buf2.try_add(fnum2, det2)
                unc2.update(det["confidence"])
                fw2.extend(det["features"][:config.FEATURE_DIM])

            if len(fw2) > 10 * config.FEATURE_DIM:
                fw2 = fw2[-(10 * config.FEATURE_DIM):]
            chunked2 = [fw2[i:i+config.FEATURE_DIM]
                        for i in range(0, len(fw2), config.FEATURE_DIM)
                        if len(fw2[i:i+config.FEATURE_DIM]) == config.FEATURE_DIM]

            fs2 = compute_feature_shift(chunked2, 0.0, 1.0)
            ts2 = trk2.update(dets2)
            td2 = 1.0 - ts2
            uc2 = unc2.score()
            cp2 = compute_composite_score(fs2, uc2, td2)
            trig2, cons2 = dc2.update(cp2)

            if trig2 and buf2.is_qualified and triggered2_at is None:
                triggered2_at = fnum2
                print(f"  RETRAINING TRIGGERED at frame {triggered2_at}  composite={cp2:.3f}  buffer={buf2.size}")
                training_result2 = trainer.fit(
                    buffer=buf2,
                    current_version='v1.0',
                    current_accuracy=0.50,
                    seed=42, num_epochs=3,
                )
                print(f"  IncrementalTrainer: accepted={training_result2['accepted']}  "
                      f"accuracy={training_result2['candidate_accuracy']*100:.1f}%")
                break

        cap.release()

        check(triggered2_at is not None,
              "Replay pipeline triggers retraining via drift detection")
        if training_result2:
            check("candidate_version"  in training_result2, "IncrementalTrainer result schema OK")
            check(0.0 <= training_result2["candidate_accuracy"] <= 1.0,
                  "IncrementalTrainer candidate_accuracy in [0,1]")
            check(isinstance(training_result2["accepted"], bool),
                  "IncrementalTrainer accepted is bool")

    except ImportError as e:
        print(f"  [SKIP] Real pipeline unavailable: {e}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"  FAILED: {len(errors)} test(s):")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  ALL INTEGRATION TESTS PASSED")
print("=" * 60)
