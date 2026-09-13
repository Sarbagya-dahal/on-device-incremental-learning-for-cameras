"""Sentinel Edge Monitor - Buffer and Training page."""

import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import config
from sentinel_pages import boot_page, render_footer, render_idle, render_page_start, stat_strip
from sentinel_style import MPL_THEME


boot_page("Buffer & Training", "B")
render_page_start(
    "Buffer & Training",
    "Pseudo-label quality, retraining gate status, and model promotion events.",
)

results = st.session_state.get("results")
if results is None:
    render_idle("No buffer activity", "Start a run to inspect label intake and training decisions.")
    st.stop()

df = results["df_frames"]
df_pseudo = results["df_pseudo"]
buffer_stats = results["buffer_stats"]
training = results.get("training_result")
logs = st.session_state.get("system_logs", [])

st.markdown('<div class="page-body">', unsafe_allow_html=True)

gate_label = "Qualified" if buffer_stats["is_qualified"] else "Collecting"
gate_color = "#66C2A3" if buffer_stats["is_qualified"] else "#D08B5B"
stat_strip(
    [
        ("Buffer fill", f"{buffer_stats['buffer_size']} / {buffer_stats['max_size']}", f"minimum {buffer_stats['min_size']}", None),
        ("Accepted", f"{buffer_stats['total_accepted']}", "pseudo-labels", "#66C2A3"),
        ("Rejected", f"{buffer_stats['total_rejected']}", "below confidence gate", "#D16060"),
        ("Acceptance", f"{buffer_stats['acceptance_rate']:.1f}%", "session rate", None),
        ("Training gate", gate_label, "ready" if buffer_stats["is_qualified"] else "not ready", gate_color),
    ]
)

left, right = st.columns([1.05, 0.95], gap="medium")

with left:
    st.markdown('<div class="section-title">Class distribution</div>', unsafe_allow_html=True)
    counts = buffer_stats.get("class_counts", {})
    max_count = max(max(counts.values()), 1) if counts else 1
    bars = []
    for name in config.OBJECT_CLASSES:
        value = int(counts.get(name, 0))
        width = int((value / max_count) * 100) if max_count else 0
        bars.append(
            f"""
<div class="metric-card" style="margin-bottom:10px;">
    <div style="display:flex;justify-content:space-between;gap:12px;">
        <span class="metric-label">{html.escape(name)}</span>
        <span class="metric-sub">{value}</span>
    </div>
    <div class="bar-track" style="margin-top:8px;"><div class="bar-fill" style="width:{width}%;background:#6EA8D9;"></div></div>
</div>
"""
        )
    st.markdown("".join(bars), unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Model validation</div>', unsafe_allow_html=True)
    if training:
        verdict = "Promoted" if training["accepted"] else "Rolled back"
        verdict_color = "#66C2A3" if training["accepted"] else "#D16060"
        st.markdown(
            f"""
<div class="val-panel">
    <table>
        <tr><td>Active before run</td><td>{training['current_accuracy'] * 100:.1f}%</td></tr>
        <tr><td>Candidate</td><td>{html.escape(training['candidate_version'])}</td></tr>
        <tr><td>Candidate accuracy</td><td>{training['candidate_accuracy'] * 100:.1f}%</td></tr>
    </table>
    <div class="metric-value" style="color:{verdict_color};font-size:22px;">{verdict}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="panel">
    <div class="metric-label">No retraining event</div>
    <div class="metric-sub" style="margin-top:8px;">The run has not met both the sustained drift and buffer gates.</div>
</div>
""",
            unsafe_allow_html=True,
        )

if training and training.get("epoch_losses"):
    st.markdown('<div class="section-title">Training loss</div>', unsafe_allow_html=True)
    epochs = list(range(1, len(training["epoch_losses"]) + 1))
    fig, ax = plt.subplots(figsize=(12, 2.8))
    fig.patch.set_facecolor(MPL_THEME["bg"])
    ax.set_facecolor(MPL_THEME["surf"])
    for spine in ax.spines.values():
        spine.set_edgecolor(MPL_THEME["bord"])
    ax.tick_params(colors=MPL_THEME["text_dim"], labelsize=9)
    ax.grid(axis="y", color=MPL_THEME["bord"], linewidth=0.5)
    ax.plot(epochs, training["epoch_losses"], color=MPL_THEME["stable"], linewidth=1.8, marker="o", markersize=4)
    ax.set_xlabel("Epoch", color=MPL_THEME["text_dim"], fontsize=9)
    ax.set_ylabel("Loss", color=MPL_THEME["text_dim"], fontsize=9)
    fig.tight_layout(pad=0.7)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown('<div class="section-title">Pseudo-label ledger</div>', unsafe_allow_html=True)
if not df_pseudo.empty:
    show = df_pseudo.sort_values("Frame", ascending=False).head(20).reset_index(drop=True)
    rows = []
    for _, row in show.iterrows():
        status = str(row["Status"])
        status_cls = "accepted" if status == "ACCEPTED" else "rejected"
        rows.append(
            f"""
<tr>
    <td>{int(row['Frame'])}</td>
    <td>{html.escape(str(row['Object']))}</td>
    <td>{float(row['Confidence']):.3f}</td>
    <td>{float(row['Threshold']):.2f}</td>
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
    st.info("No pseudo-label decisions were recorded.")

st.markdown('<div class="section-title">System log</div>', unsafe_allow_html=True)
if logs:
    st.markdown(f'<div class="log-console">{html.escape(chr(10).join(logs[-120:]))}</div>', unsafe_allow_html=True)
else:
    st.info("No events logged yet.")

st.markdown("</div>", unsafe_allow_html=True)
render_footer()
