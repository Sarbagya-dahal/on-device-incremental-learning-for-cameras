"""Sentinel Edge Monitor - Settings page."""

import streamlit as st

import config
from sentinel_engine import reset_state
from sentinel_pages import boot_page, render_footer, render_page_start


boot_page("Settings", "T")
render_page_start(
    "Settings",
    "Source selection, confidence gates, drift thresholds, and retraining controls.",
)

st.markdown('<div class="page-body">', unsafe_allow_html=True)

source_col, gate_col = st.columns([1, 1], gap="large")

with source_col:
    st.markdown('<div class="section-title">Source</div>', unsafe_allow_html=True)
    st.session_state["app_mode"] = st.selectbox(
        "Operating mode",
        ["Synthetic Demo", "Live Camera", "Replay Recorded"],
        index=["Synthetic Demo", "Live Camera", "Replay Recorded"].index(st.session_state.get("app_mode", "Synthetic Demo")),
    )

    if st.session_state["app_mode"] == "Synthetic Demo":
        st.session_state["scenario"] = st.selectbox(
            "Scenario",
            config.SCENARIOS,
            index=config.SCENARIOS.index(st.session_state.get("scenario", config.DEFAULT_SCENARIO)),
        )
        st.session_state["num_frames"] = st.slider(
            "Frames",
            30,
            500,
            int(st.session_state.get("num_frames", config.DEFAULT_NUM_FRAMES)),
            step=10,
        )
        st.session_state["seed"] = st.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=int(st.session_state.get("seed", config.DEFAULT_SEED)),
            step=1,
        )
    elif st.session_state["app_mode"] == "Replay Recorded":
        st.session_state["replay_source"] = st.text_input(
            "Replay file or frame directory",
            value=st.session_state.get("replay_source", config.REPLAY_VIDEO_PATH),
            placeholder="tests/fixtures/replay_clip.avi",
        )
    else:
        st.info(f"Live Camera uses OpenCV device index {config.CAMERA_INDEX} and stops after {config.LIVE_LOOP_MAX_FRAMES} frames.")

with gate_col:
    st.markdown('<div class="section-title">Drift gates</div>', unsafe_allow_html=True)
    st.session_state["drift_threshold"] = st.slider(
        "Composite drift threshold",
        0.10,
        0.95,
        float(st.session_state.get("drift_threshold", config.DRIFT_THRESHOLD)),
        step=0.01,
    )
    st.session_state["sustained_frames"] = st.slider(
        "Sustained frames required",
        1,
        30,
        int(st.session_state.get("sustained_frames", config.SUSTAINED_FRAMES)),
        step=1,
    )
    st.session_state["min_buffer"] = st.slider(
        "Minimum buffer size",
        1,
        config.MAX_BUFFER_SIZE,
        int(st.session_state.get("min_buffer", config.MIN_BUFFER_SIZE)),
        step=1,
    )
    st.session_state["force_rollback"] = st.checkbox(
        "Force rollback on next retraining event",
        value=bool(st.session_state.get("force_rollback", False)),
    )

weight_col, threshold_col = st.columns([1, 1], gap="large")

with weight_col:
    st.markdown('<div class="section-title">Fusion weights</div>', unsafe_allow_html=True)
    st.session_state["w_fs"] = st.slider(
        "Feature shift weight",
        0.0,
        1.0,
        float(st.session_state.get("w_fs", config.WEIGHT_FEATURE_SHIFT)),
        step=0.01,
    )
    st.session_state["w_uc"] = st.slider(
        "Uncertainty weight",
        0.0,
        1.0,
        float(st.session_state.get("w_uc", config.WEIGHT_UNCERTAINTY)),
        step=0.01,
    )
    st.session_state["w_td"] = st.slider(
        "Tracker drift weight",
        0.0,
        1.0,
        float(st.session_state.get("w_td", config.WEIGHT_TRACKER_DRIFT)),
        step=0.01,
    )
    total = st.session_state["w_fs"] + st.session_state["w_uc"] + st.session_state["w_td"]
    st.caption(f"Weights are normalized at runtime. Current raw total: {total:.2f}.")

with threshold_col:
    st.markdown('<div class="section-title">Pseudo-label thresholds</div>', unsafe_allow_html=True)
    thresholds = dict(st.session_state.get("class_thresholds", config.CLASS_CONFIDENCE_THRESHOLDS))
    for class_name in config.OBJECT_CLASSES:
        thresholds[class_name] = st.slider(
            f"{class_name} confidence",
            0.10,
            0.99,
            float(thresholds.get(class_name, config.CLASS_CONFIDENCE_THRESHOLDS[class_name])),
            step=0.01,
        )
    st.session_state["class_thresholds"] = thresholds

st.markdown('<div class="section-title">Session controls</div>', unsafe_allow_html=True)
reset_settings, clear_run = st.columns([1, 1], gap="medium")
with reset_settings:
    if st.button("Restore default settings", use_container_width=True):
        st.session_state["app_mode"] = "Synthetic Demo"
        st.session_state["scenario"] = config.DEFAULT_SCENARIO
        st.session_state["num_frames"] = config.DEFAULT_NUM_FRAMES
        st.session_state["seed"] = config.DEFAULT_SEED
        st.session_state["replay_source"] = config.REPLAY_VIDEO_PATH
        st.session_state["class_thresholds"] = config.CLASS_CONFIDENCE_THRESHOLDS.copy()
        st.session_state["drift_threshold"] = config.DRIFT_THRESHOLD
        st.session_state["sustained_frames"] = config.SUSTAINED_FRAMES
        st.session_state["min_buffer"] = config.MIN_BUFFER_SIZE
        st.session_state["w_fs"] = config.WEIGHT_FEATURE_SHIFT
        st.session_state["w_uc"] = config.WEIGHT_UNCERTAINTY
        st.session_state["w_td"] = config.WEIGHT_TRACKER_DRIFT
        st.session_state["force_rollback"] = False
        st.rerun()
with clear_run:
    if st.button("Clear run data", use_container_width=True):
        reset_state()
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
render_footer()
