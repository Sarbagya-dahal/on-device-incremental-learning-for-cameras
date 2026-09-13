# Sentinel Edge Monitor

Sentinel Edge Monitor is a Streamlit prototype for autonomous edge-camera monitoring. It watches detector confidence, feature distribution shift, and tracker stability, then decides when the deployed model should retrain from a bounded pseudo-label buffer.

The app is designed for offline evaluation on a laptop or edge workstation. Synthetic, replay, and live-camera modes share the same drift engine and session state.

## What It Does

- Runs a live, replayed, or synthetic camera session.
- Detects indoor desk objects with YOLOv8n when real video is available.
- Extracts MobileNetV2 embeddings for feature-shift monitoring.
- Filters pseudo-labels through class-specific confidence thresholds.
- Combines three drift signals into a composite retraining score.
- Triggers incremental training only when drift is sustained and the buffer is qualified.
- Promotes or rolls back candidate models based on validation accuracy.

## App Structure

- **Live Monitor**: camera signal, current detections, composite drift score, and live operating posture.
- **Drift Analytics**: signal timelines, threshold crossings, and fusion-rule reference.
- **Buffer & Training**: pseudo-label intake, class distribution, training loss, validation result, and system logs.
- **Settings**: source mode, scenario, replay path, thresholds, fusion weights, buffer gate, and rollback testing.

Streamlit's native sidebar navigation is used for page switching. Run/reset controls remain in the sidebar, while all tuning controls live on the Settings page so they do not crowd the live dashboard.

## Pipeline

```text
Camera source
  -> YOLOv8n detector
  -> MobileNetV2 feature extractor
  -> Confidence self-labeling gate
  -> Pseudo-label buffer
  -> Drift engine
       0.40 * feature shift
     + 0.35 * uncertainty
     + 0.25 * tracker drift
  -> Sustained threshold check
  -> Incremental trainer
  -> Validation
  -> Promote or rollback
```

## Install

```bash
git clone https://github.com/Sarbagya-dahal/on-device-incremental-learning-for-cameras.git
cd on-device-incremental-learning-for-cameras
pip install -r requirements.txt
```

YOLOv8n is downloaded automatically on the first real detector run. After install, synthetic and replay workflows can run offline.

## Run

```bash
py -m streamlit run app.py
```

Open Settings first if you want to change the source mode or thresholds. Then use the sidebar run button from any page.

## Source Modes

**Synthetic Demo**

No camera is required. Choose a scenario such as `Severe Distribution Shift`, set the frame count, and run the simulation.

**Live Camera**

Uses OpenCV camera index `0` and processes up to `LIVE_LOOP_MAX_FRAMES` frames. During a live demo, you can induce drift by partially covering the lens, changing the background, dimming the room, or moving quickly.

**Replay Recorded**

Uses a video file or directory of frames. Generate the bundled replay fixture with:

```bash
py -3 scripts/create_test_fixture.py
```

Then set the replay source to:

```text
tests/fixtures/replay_clip.avi
```

## Tests

```bash
py -3 test_all.py
py -3 test_integration.py
```

The tests skip optional heavy dependencies when they are not installed, so the synthetic path remains usable without camera or model downloads.

## Tunable Classes

| Class | Default confidence threshold |
| --- | ---: |
| `person` | 0.70 |
| `cup` | 0.60 |
| `cell phone` | 0.65 |
| `laptop` | 0.65 |
| `bottle` | 0.60 |

## Current Prototype Boundaries

- Feature shift uses scalar baseline statistics rather than full MMD.
- Uncertainty uses a sliding-window confidence trend rather than ADWIN or Page-Hinkley.
- Tracker drift uses per-class IoU rather than a full multi-object tracker.
- The incremental head is an SGDClassifier rather than end-to-end adapter tuning.
- Validation is measured against held-out pseudo-labels, not human-verified ground truth.
- CPU inference is expected to be slower than a tuned NPU or DSP deployment.

## Repository Layout

```text
app.py                         Live Monitor page
pages/
  2_Drift_Analytics.py         Drift charting page
  3_Buffer_Training.py         Buffer, training, and logs page
  4_Settings.py                Runtime settings page
sentinel_engine.py             Shared session state and pipeline execution
sentinel_pages.py              Shared Streamlit page helpers
sentinel_style.py              Shared product styling
config.py                      Defaults and model/runtime constants
camera_capture.py              OpenCV live/replay frame source
object_detector.py             YOLOv8n detector and fallback detector
feature_extractor.py           MobileNetV2 embeddings
drift_detection.py             Drift-signal and fusion logic
self_labeling.py               Confidence gate and pseudo-label buffer
incremental_trainer.py         Incremental training and rollback
simulator.py                   Synthetic camera stream
training_simulator.py          Synthetic training fallback
tests/fixtures/                Replay fixture assets
```
