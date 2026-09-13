"""Shared page rendering helpers for the Sentinel Streamlit app."""

from __future__ import annotations

import html
from typing import Iterable

import pandas as pd
import streamlit as st

import config
from sentinel_engine import init_state, render_sidebar, reset_state, run_pipeline
from sentinel_style import apply_style, score_color, state_badge


PRODUCT_NAME = "Sentinel Edge Monitor"


def boot_page(title: str, icon: str = "S") -> None:
    st.set_page_config(
        page_title=f"Sentinel - {title}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_style()
    init_state()


def mode_meta() -> tuple[str, str]:
    mode = st.session_state.get("app_mode", "Synthetic Demo")
    labels = {
        "Synthetic Demo": ("Synthetic", "mp-synth"),
        "Live Camera": ("Live", "mp-live"),
        "Replay Recorded": ("Replay", "mp-replay"),
    }
    return labels.get(mode, ("Unknown", "mp-synth"))


def render_shell(title: str, subtitle: str) -> None:
    mode_label, mode_class = mode_meta()
    model = st.session_state.get("model_version", "v1.0")
    accuracy = st.session_state.get("model_accuracy", 0.862)
    st.markdown(
        f"""
<div class="page-header">
    <div>
        <div class="ph-title">{html.escape(title)}</div>
        <div class="ph-sub">{html.escape(subtitle)}</div>
    </div>
    <div class="ph-meta">
        <span class="mode-pill {mode_class}">{mode_label}</span><br>
        model {model} / accuracy {accuracy * 100:.1f}%
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_page_start(title: str, subtitle: str) -> None:
    with st.sidebar:
        run_btn, reset_btn = render_sidebar()

    if reset_btn:
        reset_state()
        st.rerun()

    if run_btn:
        mode = st.session_state.get("app_mode", "Synthetic Demo")
        img_ph = st.empty()
        status_ph = st.empty()
        with st.spinner("Processing camera session..."):
            run_pipeline(
                ui_image_ph=img_ph if mode != "Synthetic Demo" else None,
                ui_status_ph=status_ph if mode != "Synthetic Demo" else None,
            )
        st.rerun()

    render_shell(title, subtitle)


def render_idle(title: str = "No telemetry yet", body: str | None = None) -> None:
    body = body or "Configure the source on Settings, then start a run from the sidebar."
    st.markdown(
        f"""
<div class="idle-panel">
    <div class="ip-score">--</div>
    <div class="ip-title">{html.escape(title)}</div>
    <div class="ip-desc">{html.escape(body)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    style = f' style="color:{color};"' if color else ""
    return f"""
<div class="metric-card">
    <div class="metric-label">{html.escape(label)}</div>
    <div class="metric-value"{style}>{value}</div>
    <div class="metric-sub">{html.escape(sub)}</div>
</div>
"""


def stat_strip(items: Iterable[tuple[str, str, str, str | None]]) -> None:
    cells = "".join(metric_card(label, value, sub, color) for label, value, sub, color in items)
    st.markdown(f'<div class="stat-strip">{cells}</div>', unsafe_allow_html=True)


def render_signal_grid(df: pd.DataFrame, threshold: float) -> None:
    last = df.iloc[-1]
    specs = [
        ("Feature Shift", float(last.get("feature_shift", 0)), "embedding distribution", threshold),
        ("Uncertainty", float(last.get("uncertainty", 0)), "confidence trend", threshold),
        ("Tracker Drift", float(last.get("tracker_drift", 0)), "box displacement", threshold),
        ("Tracker Stability", float(last.get("tracker_stability", 1)), "temporal overlap", 1.0),
    ]
    panels = []
    for label, value, desc, local_thr in specs:
        color = score_color(value, local_thr) if label != "Tracker Stability" else "#6EA8D9"
        width = min(max(int(value * 100), 0), 100)
        panels.append(
            f"""
<div class="signal-panel">
    <div class="sp-label">{label}</div>
    <div class="sp-value" style="color:{color};">{value:.3f}</div>
    <div class="sp-desc">{desc}</div>
    <div class="sp-track"><div class="sp-fill" style="width:{width}%;background:{color};"></div></div>
</div>
"""
        )
    st.markdown(f'<div class="signal-grid">{"".join(panels)}</div>', unsafe_allow_html=True)


def render_feed_panel(results: dict) -> None:
    df = results["df_frames"]
    df_pseudo = results["df_pseudo"]
    last_frame = int(df["frame"].iloc[-1]) if not df.empty else 0
    recent = df_pseudo.sort_values("Frame", ascending=False).head(3) if not df_pseudo.empty else pd.DataFrame()
    boxes = []
    positions = [(14, 24, 28, 34), (48, 18, 22, 28), (58, 52, 24, 26)]
    for idx, (_, row) in enumerate(recent.iterrows()):
        left, top, width, height = positions[idx % len(positions)]
        status = str(row.get("Status", ""))
        cls = "bbox alert" if status != "ACCEPTED" else "bbox"
        label = f"{row.get('Object', 'object')} {float(row.get('Confidence', 0)):.2f}"
        boxes.append(
            f'<div class="{cls}" style="left:{left}%;top:{top}%;width:{width}%;height:{height}%;">'
            f"<span>{html.escape(label)}</span></div>"
        )
    if not boxes:
        boxes.append('<div class="feed-reticle"></div>')

    mode = st.session_state.get("app_mode", "Synthetic Demo")
    st.markdown(
        f"""
<div class="feed-panel">
    <div class="feed-toolbar">
        <div class="metric-label">Camera Signal</div>
        <div class="metric-sub">frame {last_frame:04d} / {html.escape(mode)}</div>
    </div>
    <div class="feed-stage">
        <div class="feed-reticle"></div>
        {''.join(boxes)}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    mode_label, _ = mode_meta()
    st.markdown(
        f'<div class="sentinel-footer">{PRODUCT_NAME} / Prototype / {mode_label}</div>',
        unsafe_allow_html=True,
    )
