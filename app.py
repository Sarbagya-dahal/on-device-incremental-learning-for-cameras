"""Sentinel Edge Monitor - Live Monitor page."""

import html

import pandas as pd
import streamlit as st

from sentinel_pages import (
    boot_page,
    render_feed_panel,
    render_footer,
    render_idle,
    render_page_start,
    render_signal_grid,
    stat_strip,
)
from sentinel_style import score_color, state_badge


boot_page("Live Monitor", "S")
render_page_start(
    "Live Monitor",
    "Real-time drift posture, camera signal, and current detections.",
)

results = st.session_state.get("results")
mode = st.session_state.get("app_mode", "Synthetic Demo")

if mode == "Live Camera" and results is None:
    st.markdown(
        """
<div class="page-body">
    <div class="drift-tips">
        <div class="dt-label">Live drift exercises</div>
        <ul>
            <li>Cover part of the lens to stress tracker stability.</li>
            <li>Move to a new background to raise feature shift.</li>
            <li>Dim the room to test uncertainty response.</li>
            <li>Move quickly to combine all three drift signals.</li>
        </ul>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

if results is None:
    render_idle()
    st.stop()

df = results["df_frames"]
df_pseudo = results["df_pseudo"]
buf = results["buffer_stats"]
threshold = float(results["drift_threshold"])

if df.empty:
    render_idle("No frames processed", "Check the selected camera or replay source.")
    st.stop()

last = df.iloc[-1]
composite = float(last.get("composite", 0))
consecutive = int(last.get("consecutive", 0))
state = results.get("final_state", "MODEL STABLE")
score_cls = "sh-alert" if composite >= threshold else "sh-warning" if composite >= threshold * 0.8 else "sh-stable"
score_col = score_color(composite, threshold)
trigger_frame = results.get("retraining_triggered_at")
avg_conf = float(df["mean_conf"].mean()) if "mean_conf" in df else 0.0

st.markdown('<div class="page-body">', unsafe_allow_html=True)
left, right = st.columns([0.92, 1.48], gap="medium")

with left:
    trigger_text = f"triggered at frame {trigger_frame}" if trigger_frame else "trigger not active"
    st.markdown(
        f"""
<div class="score-hero {score_cls}">
    <div class="sh-label">Composite drift score</div>
    <div class="sh-value" style="color:{score_col};">{composite:.3f}</div>
    <div class="sh-thr">threshold {threshold:.2f} / {consecutive} consecutive frames</div>
    <div class="sh-state">{state_badge(state)}<span class="metric-sub">{html.escape(trigger_text)}</span></div>
    <div class="sh-formula">
        {results["w_fs"]:.2f} x feature shift<br>
        {results["w_uc"]:.2f} x uncertainty<br>
        {results["w_td"]:.2f} x tracker drift<br>
        = {composite:.4f}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    render_feed_panel(results)

render_signal_grid(df, threshold)

gate = "Qualified" if buf["is_qualified"] else "Pending"
gate_color = "#66C2A3" if buf["is_qualified"] else "#D08B5B"
stat_strip(
    [
        ("Frames", f"{len(df)}", mode.lower(), None),
        ("Average confidence", f"{avg_conf * 100:.1f}%", "detector output", None),
        ("Buffer", f"{buf['buffer_size']} / {buf['max_size']}", f"minimum {buf['min_size']}", None),
        ("Acceptance", f"{buf['acceptance_rate']:.1f}%", "pseudo-label gate", None),
        ("Training gate", gate, "ready to retrain" if buf["is_qualified"] else "collecting labels", gate_color),
    ]
)

st.markdown('<div class="section-title">Recent detections</div>', unsafe_allow_html=True)
if not df_pseudo.empty:
    show = df_pseudo.sort_values("Frame", ascending=False).head(14).reset_index(drop=True)
    rows = []
    for _, row in show.iterrows():
        status = str(row["Status"])
        status_cls = "accepted" if status == "ACCEPTED" else "rejected"
        rows.append(
            f"""
<tr>
    <td>{int(row["Frame"])}</td>
    <td>{html.escape(str(row["Object"]))}</td>
    <td>{float(row["Confidence"]):.3f}</td>
    <td>{float(row["Threshold"]):.2f}</td>
    <td class="{status_cls}">{html.escape(status.title())}</td>
</tr>
"""
        )
    st.markdown(
        f"""
<table class="det-table">
    <thead><tr><th>Frame</th><th>Object</th><th>Confidence</th><th>Threshold</th><th>Status</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
</table>
""",
        unsafe_allow_html=True,
    )
else:
    st.info("No detections recorded for this run.")

st.markdown("</div>", unsafe_allow_html=True)
render_footer()
