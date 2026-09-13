"""
sentinel_engine.py
------------------
Shared session state, pipeline engines, and sidebar renderer for Sentinel.

Every page imports from here:
    from sentinel_engine import init_state, render_sidebar, run_pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

import config
from simulator          import generate_frames
from self_labeling      import PseudoLabelBuffer
from drift_detection    import (
    compute_feature_shift, UncertaintyTracker,
    TrackerStabilityMonitor, compute_composite_score, SustainedDriftChecker,
)
from training_simulator import simulate_training


# ── Lazy import of heavy Phase 2 modules ──────────────────────────────────────

def _get_real_pipeline():
    from object_detector    import ObjectDetector, draw_detections
    from camera_capture     import CameraCapture
    from incremental_trainer import IncrementalTrainer
    return ObjectDetector, draw_detections, CameraCapture, IncrementalTrainer


# ── Default settings ───────────────────────────────────────────────────────────

_DEFAULTS = dict(
    # Mode / source
    app_mode        = "Synthetic Demo",
    scenario        = config.DEFAULT_SCENARIO,
    num_frames      = config.DEFAULT_NUM_FRAMES,
    seed            = config.DEFAULT_SEED,
    replay_source   = config.REPLAY_VIDEO_PATH,
    # Thresholds
    class_thresholds = config.CLASS_CONFIDENCE_THRESHOLDS.copy(),
    # Drift detection
    drift_threshold  = config.DRIFT_THRESHOLD,
    sustained_frames = config.SUSTAINED_FRAMES,
    min_buffer       = config.MIN_BUFFER_SIZE,
    # Fusion weights (raw; normalised at run-time)
    w_fs = config.WEIGHT_FEATURE_SHIFT,
    w_uc = config.WEIGHT_UNCERTAINTY,
    w_td = config.WEIGHT_TRACKER_DRIFT,
    # Options
    force_rollback  = False,
    # Runtime state
    results         = None,
    system_logs     = [],
    model_version   = "v1.0",
    model_accuracy  = 0.862,
    live_trainer    = None,
)


def init_state():
    """Initialise session state with defaults (idempotent — won't overwrite)."""
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_state():
    """Hard-reset all runtime state (keep settings)."""
    for k in ("results", "system_logs", "model_version", "model_accuracy", "live_trainer"):
        st.session_state[k] = _DEFAULTS[k]  # type: ignore


# ── Sidebar renderer ───────────────────────────────────────────────────────────

def render_sidebar():
    """
    Render the Sentinel wordmark, current status, and Run/Reset buttons in
    the sidebar.  Call this inside every page's `with st.sidebar:` block.
    Shared settings are read from (and written to) st.session_state.
    """
    from sentinel_style import state_badge

    results = st.session_state.get("results")
    ver     = st.session_state.get("model_version", "v1.0")
    acc     = st.session_state.get("model_accuracy", 0.862)
    mode    = st.session_state.get("app_mode", "Synthetic Demo")

    # Wordmark
    st.markdown(f"""
<div class='sentinel-wordmark'>
    <span class='sw-name'>Sentinel</span>
    <span class='sw-tag'>edge monitor / prototype</span>
</div>""", unsafe_allow_html=True)

    # Current model status
    if results:
        state  = results.get("final_state", "MODEL STABLE")
        badge  = state_badge(state)
        cp_val = results.get("df_frames", pd.DataFrame())
        cp     = float(cp_val["composite"].iloc[-1]) if not cp_val.empty and "composite" in cp_val else 0.0
        st.markdown(f"""
<div class='sidebar-status'>
    {badge} &nbsp;
    <span class='ss-model'>{ver}</span> &nbsp;
    <span class='ss-acc'>{acc*100:.1f}%</span>
    <div style='margin-top:4px;font-size:10px;'>composite: <span style='color:#C8D8E8;'>{cp:.3f}</span></div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class='sidebar-status'>
    <span class='ss-model'>No run yet</span>
    <div style='margin-top:4px;font-size:10px;'>configure in Settings, then run</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='padding: 12px 20px 0 20px;'>", unsafe_allow_html=True)

    # Mode indicator
    mode_labels = {
        "Synthetic Demo":  ("Synthetic",   "mp-synth"),
        "Live Camera":     ("Live Camera", "mp-live"),
        "Replay Recorded": ("Replay",      "mp-replay"),
    }
    ml, mc = mode_labels.get(mode, ("Unknown", "mp-synth"))
    st.markdown(f"<span class='mode-pill {mc}'>{ml}</span>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Action buttons
    if mode == "Live Camera":
        lbl = "Start Live Feed"
    elif mode == "Replay Recorded":
        lbl = "Run Replay"
    else:
        lbl = "Run Simulation"

    run_btn   = st.button(lbl,  type="primary",   use_container_width=True)
    reset_btn = st.button("Reset", type="secondary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Use Settings to tune source, thresholds, and retraining gates.")

    with st.expander("Pipeline diagram"):
        st.markdown("""<div class='arch-box'>Camera [live/replay/synth]
    |
YOLOv8n Detector
    |
MobileNetV2 Features
    |
Self-Labeling gate
    |
Pseudo-Label Buffer
    |
+-- Drift Engine ------+
| Feature Shift  x0.40 |
| Uncertainty    x0.35 |
| Tracker Drift  x0.25 |
+----------+-----------+
           |
    Composite Score
    Sustained >= 5f
    Buffer >= 30
           |
      [Trigger?]
      /        \\
    YES         NO
     |          |
SGDClassifier  Stable
  partial_fit
     |
  Validation
  /      \\
OK    Regress
 |        |
Update  Rollback</div>""", unsafe_allow_html=True)

    return run_btn, reset_btn


# ── Normalise fusion weights ───────────────────────────────────────────────────

def get_weights():
    """Return normalised (w_fs, w_uc, w_td) from session_state."""
    w_fs = st.session_state.get("w_fs", config.WEIGHT_FEATURE_SHIFT)
    w_uc = st.session_state.get("w_uc", config.WEIGHT_UNCERTAINTY)
    w_td = st.session_state.get("w_td", config.WEIGHT_TRACKER_DRIFT)
    s = w_fs + w_uc + w_td
    if s > 0:
        w_fs /= s; w_uc /= s; w_td /= s
    return w_fs, w_uc, w_td


# ── Run dispatcher ─────────────────────────────────────────────────────────────

def run_pipeline(ui_image_ph=None, ui_status_ph=None):
    """
    Read all settings from session_state and execute the appropriate engine.
    Stores results back into session_state.
    """
    mode            = st.session_state["app_mode"]
    class_thresholds = st.session_state["class_thresholds"]
    drift_threshold  = st.session_state["drift_threshold"]
    sustained_frames = st.session_state["sustained_frames"]
    min_buffer       = st.session_state["min_buffer"]
    force_rollback   = st.session_state["force_rollback"]
    current_version  = st.session_state["model_version"]
    current_accuracy = st.session_state["model_accuracy"]
    w_fs, w_uc, w_td = get_weights()

    if mode == "Synthetic Demo":
        results = _run_simulation(
            scenario        = st.session_state["scenario"],
            num_frames      = st.session_state["num_frames"],
            seed            = st.session_state["seed"],
            class_thresholds = class_thresholds,
            drift_threshold  = drift_threshold,
            sustained_frames = sustained_frames,
            min_buffer       = min_buffer,
            w_fs=w_fs, w_uc=w_uc, w_td=w_td,
            force_rollback   = force_rollback,
            current_model_version  = current_version,
            current_model_accuracy = current_accuracy,
        )
    else:
        source = (config.CAMERA_INDEX if mode == "Live Camera"
                  else st.session_state.get("replay_source", config.REPLAY_VIDEO_PATH))
        results = _run_real_pipeline(
            source          = source,
            class_thresholds = class_thresholds,
            drift_threshold  = drift_threshold,
            sustained_frames = sustained_frames,
            min_buffer       = min_buffer,
            w_fs=w_fs, w_uc=w_uc, w_td=w_td,
            force_rollback   = force_rollback,
            current_model_version  = current_version,
            current_model_accuracy = current_accuracy,
            max_frames       = config.LIVE_LOOP_MAX_FRAMES,
            ui_image_placeholder  = ui_image_ph,
            ui_status_placeholder = ui_status_ph,
        )

    st.session_state["results"]     = results
    st.session_state["system_logs"] = results["logs"]
    tr = results.get("training_result")
    if tr and tr["accepted"]:
        st.session_state["model_version"]  = tr["candidate_version"]
        st.session_state["model_accuracy"] = tr["candidate_accuracy"]


# ── Synthetic pipeline engine ──────────────────────────────────────────────────

def _run_simulation(scenario, num_frames, seed,
                    class_thresholds, drift_threshold, sustained_frames,
                    min_buffer, w_fs, w_uc, w_td,
                    force_rollback, current_model_version, current_model_accuracy):
    logs = []
    def log(msg): logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    buffer      = PseudoLabelBuffer(config.MAX_BUFFER_SIZE, min_buffer, class_thresholds)
    unc_tracker = UncertaintyTracker(window=config.UNCERTAINTY_WINDOW)
    trk_monitor = TrackerStabilityMonitor(window=10)
    dc          = SustainedDriftChecker(drift_threshold, sustained_frames)

    bl_feats = []
    for bf in generate_frames("Normal Environment", 20, seed):
        for det in bf["detections"]: bl_feats.extend(det["features"])
    bl_mean = float(np.mean(bl_feats))
    bl_std  = float(np.std(bl_feats)) or 1.0

    if scenario in ("Lighting Change", "Background Change", "Severe Distribution Shift"):
        cnt = 0
        for pf in generate_frames("Normal Environment", 15, seed + 1):
            for det in pf["detections"]:
                ok, _ = buffer.try_add(0, det)
                if ok: cnt += 1
        log(f"Buffer pre-seeded: {cnt} samples from normal conditions")

    frames = generate_frames(scenario, num_frames, seed)
    log(f"Scenario '{scenario}' — {num_frames} frames")

    frame_records, pseudo_rows = [], []
    retraining_triggered_at = None
    system_state = "MODEL STABLE"
    training_result = None
    fw: list = []
    FW = 15

    for fd in frames:
        fnum = fd["frame"]
        dets = fd["detections"]
        ff = []
        for det in dets:
            accepted, thr_used = buffer.try_add(fnum, det)
            ss = "ACCEPTED" if accepted else "REJECTED"
            pseudo_rows.append({"Frame": fnum, "Object": det["object"],
                                 "Confidence": round(det["confidence"], 3),
                                 "Threshold": round(thr_used, 2), "Status": ss})
            if accepted:
                log(f"pseudo-label  {det['object']}  conf={det['confidence']:.3f}")
            ff.extend(det["features"])

        mean_conf = float(np.mean([d["confidence"] for d in dets]))
        unc_tracker.update(mean_conf)

        fw.extend(ff)
        if len(fw) > FW * config.FEATURE_DIM: fw = fw[-(FW * config.FEATURE_DIM):]
        chunked = [fw[i:i+config.FEATURE_DIM]
                   for i in range(0, len(fw), config.FEATURE_DIM)
                   if len(fw[i:i+config.FEATURE_DIM]) == config.FEATURE_DIM]

        fs = compute_feature_shift(chunked, bl_mean, bl_std)
        ts = trk_monitor.update(dets)
        td = trk_monitor.tracker_drift()
        uc = unc_tracker.score()
        cp = compute_composite_score(fs, uc, td, w_fs, w_uc, w_td)
        triggered, consecutive = dc.update(cp)

        if training_result and training_result["accepted"]:
            system_state = "MODEL UPDATED"
        elif training_result and not training_result["accepted"]:
            system_state = "ROLLBACK"
        elif triggered and buffer.is_qualified and retraining_triggered_at is None:
            system_state = "RETRAINING TRIGGERED"
            retraining_triggered_at = fnum
            log(f"DRIFT THRESHOLD EXCEEDED  composite={cp:.3f}")
            log(f"Sustained {consecutive} frames above threshold")
            log(f"Buffer qualified: {buffer.size} samples")
        elif cp >= drift_threshold and consecutive > 0:
            system_state = "DRIFT DETECTED"
        elif cp >= drift_threshold * 0.8:
            system_state = "DRIFT WARNING"
        elif not training_result:
            system_state = "MODEL STABLE"

        log(f"f{fnum:03d}  FS={fs:.3f} UC={uc:.3f} TD={td:.3f} CP={cp:.3f}  {system_state}")
        frame_records.append({
            "frame": fnum, "mean_conf": mean_conf,
            "feature_shift": fs, "uncertainty": uc,
            "tracker_stability": ts, "tracker_drift": td,
            "composite": cp, "consecutive": consecutive,
            "system_state": system_state, "buffer_size": buffer.size,
        })

    if retraining_triggered_at is not None and buffer.is_qualified:
        training_result = simulate_training(
            buffer_size=buffer.size, current_version=current_model_version,
            current_accuracy=current_model_accuracy, seed=int(seed),
            force_rollback=force_rollback,
        )
        for tl in training_result["logs"]: logs.append(tl)
        if training_result["accepted"]:
            log(f"Model promoted: {current_model_version} → {training_result['candidate_version']}")
        else:
            log(f"Rollback: {current_model_version} retained")
    else:
        training_result = None
        if retraining_triggered_at and not buffer.is_qualified:
            log(f"Buffer gate blocked retraining: {buffer.size}/{min_buffer}")

    return {
        "df_frames": pd.DataFrame(frame_records),
        "df_pseudo": pd.DataFrame(pseudo_rows),
        "buffer_stats": buffer.get_stats(),
        "retraining_triggered_at": retraining_triggered_at,
        "training_result": training_result,
        "final_state": system_state,
        "drift_threshold": drift_threshold,
        "sustained_frames": sustained_frames,
        "min_buffer": min_buffer,
        "w_fs": w_fs, "w_uc": w_uc, "w_td": w_td,
        "mode": "synthetic",
        "scenario": scenario,
        "seed": seed,
        "logs": logs,
    }


# ── Real pipeline engine ───────────────────────────────────────────────────────

def _run_real_pipeline(source, class_thresholds, drift_threshold, sustained_frames,
                       min_buffer, w_fs, w_uc, w_td,
                       force_rollback, current_model_version, current_model_accuracy,
                       max_frames=config.LIVE_LOOP_MAX_FRAMES,
                       ui_image_placeholder=None, ui_status_placeholder=None):
    ObjectDetector, draw_detections, CameraCapture, IncrementalTrainer = _get_real_pipeline()

    logs = []
    def log(msg): logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    detector = ObjectDetector()
    cap      = CameraCapture(source=source, fps_target=config.LIVE_FPS_TARGET)

    trainer = st.session_state.get("live_trainer")
    if trainer is None:
        trainer = IncrementalTrainer(classes=config.OBJECT_CLASSES)
        st.session_state["live_trainer"] = trainer

    buffer      = PseudoLabelBuffer(config.MAX_BUFFER_SIZE, min_buffer, class_thresholds)
    unc_tracker = UncertaintyTracker(window=config.UNCERTAINTY_WINDOW)
    trk_monitor = TrackerStabilityMonitor(window=10)
    dc          = SustainedDriftChecker(drift_threshold, sustained_frames)

    if not cap.open():
        log("ERROR: Could not open camera/video source.")
        return _empty_result(drift_threshold, sustained_frames, min_buffer,
                             w_fs, w_uc, w_td, logs)

    log(f"Source opened: {cap.mode}  max={max_frames} frames")

    frame_records, pseudo_rows = [], []
    retraining_triggered_at    = None
    system_state               = "MODEL STABLE"
    training_result            = None
    fw: list = []
    FW = 10
    baseline_feats: list = []
    bl_mean = config.BASELINE_FEATURE_MEAN
    bl_std  = config.BASELINE_FEATURE_STD
    WARMUP  = 20
    fnum    = 0

    try:
        while fnum < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                log("Frame source exhausted.")
                break
            fnum += 1

            dets = detector.detect(frame_bgr)

            if ui_image_placeholder is not None:
                import cv2
                annotated = draw_detections(frame_bgr, dets)
                ui_image_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption=f"Frame {fnum}  ·  {len(dets)} detection(s)",
                )

            if not dets:
                unc_tracker.update(0.0)
                continue

            ff = []
            for det in dets:
                accepted, thr_used = buffer.try_add(fnum, det)
                ss = "ACCEPTED" if accepted else "REJECTED"
                pseudo_rows.append({"Frame": fnum, "Object": det["object"],
                                     "Confidence": round(det["confidence"], 3),
                                     "Threshold": round(thr_used, 2), "Status": ss})
                ff.extend(det["features"])

            mean_conf = float(np.mean([d["confidence"] for d in dets]))
            unc_tracker.update(mean_conf)

            if fnum <= WARMUP:
                baseline_feats.extend(ff)
                if fnum == WARMUP and baseline_feats:
                    bl_mean = float(np.mean(baseline_feats))
                    bl_std  = float(np.std(baseline_feats)) or 1.0
                    log(f"Baseline estimated over {WARMUP} frames")

            fw.extend(ff)
            if len(fw) > FW * config.FEATURE_DIM: fw = fw[-(FW * config.FEATURE_DIM):]
            chunked = [fw[i:i+config.FEATURE_DIM]
                       for i in range(0, len(fw), config.FEATURE_DIM)
                       if len(fw[i:i+config.FEATURE_DIM]) == config.FEATURE_DIM]

            fs = compute_feature_shift(chunked, bl_mean, bl_std)
            ts = trk_monitor.update(dets)
            td = trk_monitor.tracker_drift()
            uc = unc_tracker.score()
            cp = compute_composite_score(fs, uc, td, w_fs, w_uc, w_td)
            triggered, consecutive = dc.update(cp)

            if training_result and training_result["accepted"]:
                system_state = "MODEL UPDATED"
            elif training_result and not training_result["accepted"]:
                system_state = "ROLLBACK"
            elif triggered and buffer.is_qualified and retraining_triggered_at is None:
                system_state = "RETRAINING TRIGGERED"
                retraining_triggered_at = fnum
                log(f"DRIFT THRESHOLD EXCEEDED  composite={cp:.3f}")
                training_result = trainer.fit(
                    buffer=buffer,
                    current_version=current_model_version,
                    current_accuracy=current_model_accuracy,
                    force_rollback=force_rollback,
                )
                for tl in training_result["logs"]: logs.append(tl)
                if training_result["accepted"]:
                    log(f"Model promoted → {training_result['candidate_version']}")
                else:
                    log(f"Rollback: {current_model_version} retained")
            elif cp >= drift_threshold and consecutive > 0:
                system_state = "DRIFT DETECTED"
            elif cp >= drift_threshold * 0.8:
                system_state = "DRIFT WARNING"
            elif not training_result:
                system_state = "MODEL STABLE"

            if ui_status_placeholder is not None:
                ui_status_placeholder.markdown(
                    f"<div style='font-family:var(--font-mono,monospace);font-size:12px;"
                    f"color:#6A7E92;padding:4px 0;'>"
                    f"f{fnum:04d} &nbsp; FS={fs:.3f} &nbsp; UC={uc:.3f} &nbsp; "
                    f"TD={td:.3f} &nbsp; CP={cp:.3f} &nbsp; {system_state}</div>",
                    unsafe_allow_html=True,
                )

            log(f"f{fnum:03d}  FS={fs:.3f} UC={uc:.3f} TD={td:.3f} CP={cp:.3f}  {system_state}")
            frame_records.append({
                "frame": fnum, "mean_conf": mean_conf,
                "feature_shift": fs, "uncertainty": uc,
                "tracker_stability": ts, "tracker_drift": td,
                "composite": cp, "consecutive": consecutive,
                "system_state": system_state, "buffer_size": buffer.size,
            })
    finally:
        cap.release()

    return {
        "df_frames": pd.DataFrame(frame_records) if frame_records else pd.DataFrame(),
        "df_pseudo": pd.DataFrame(pseudo_rows)   if pseudo_rows   else pd.DataFrame(),
        "buffer_stats": buffer.get_stats(),
        "retraining_triggered_at": retraining_triggered_at,
        "training_result": training_result,
        "final_state": system_state,
        "drift_threshold": drift_threshold,
        "sustained_frames": sustained_frames,
        "min_buffer": min_buffer,
        "w_fs": w_fs, "w_uc": w_uc, "w_td": w_td,
        "mode": "live",
        "scenario": "—",
        "seed": 0,
        "logs": logs,
    }


def _empty_result(drift_threshold, sustained_frames, min_buffer,
                  w_fs, w_uc, w_td, logs):
    return {
        "df_frames": pd.DataFrame(), "df_pseudo": pd.DataFrame(),
        "buffer_stats": PseudoLabelBuffer().get_stats(),
        "retraining_triggered_at": None, "training_result": None,
        "final_state": "MODEL STABLE",
        "drift_threshold": drift_threshold, "sustained_frames": sustained_frames,
        "min_buffer": min_buffer, "w_fs": w_fs, "w_uc": w_uc, "w_td": w_td,
        "mode": "error", "scenario": "—", "seed": 0, "logs": logs,
    }
