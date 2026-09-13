"""Sentinel Edge Monitor - Drift Analytics page."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from sentinel_pages import boot_page, render_footer, render_idle, render_page_start, stat_strip
from sentinel_style import MPL_THEME


boot_page("Drift Analytics", "D")
render_page_start(
    "Drift Analytics",
    "Signal history, threshold crossings, and fusion math for the current run.",
)

results = st.session_state.get("results")
if results is None or results["df_frames"].empty:
    render_idle("No drift history", "Start a run to inspect signal timelines.")
    st.stop()

df = results["df_frames"]
threshold = float(results["drift_threshold"])
trigger_frame = results.get("retraining_triggered_at")
x = df["frame"].values

st.markdown('<div class="page-body">', unsafe_allow_html=True)

final = df.iloc[-1]
above = int((df["composite"] >= threshold).sum())
peak = float(df["composite"].max())
stat_strip(
    [
        ("Peak composite", f"{peak:.3f}", "maximum observed", None),
        ("Final composite", f"{float(final['composite']):.3f}", "latest frame", None),
        ("Above threshold", f"{above}", "frames", None),
        ("Trigger frame", f"{trigger_frame or '--'}", "sustained gate", None),
        ("Window", f"{len(df)}", "processed frames", None),
    ]
)


def apply_axis(ax):
    ax.set_facecolor(MPL_THEME["surf"])
    for spine in ax.spines.values():
        spine.set_edgecolor(MPL_THEME["bord"])
    ax.tick_params(colors=MPL_THEME["text_dim"], labelsize=9)
    ax.grid(axis="y", color=MPL_THEME["bord"], linewidth=0.5)
    ax.grid(axis="x", color=MPL_THEME["bord"], linewidth=0.3, linestyle=":")


st.markdown('<div class="section-title">Composite drift score over time</div>', unsafe_allow_html=True)
fig1, ax1 = plt.subplots(figsize=(14, 4.2))
fig1.patch.set_facecolor(MPL_THEME["bg"])
apply_axis(ax1)
ax1.plot(x, df["feature_shift"], color=MPL_THEME["telemetry"], linewidth=0.9, alpha=0.58, linestyle="--", label=f"Feature shift x{results['w_fs']:.2f}")
ax1.plot(x, df["uncertainty"], color=MPL_THEME["warning"], linewidth=0.9, alpha=0.58, linestyle="--", label=f"Uncertainty x{results['w_uc']:.2f}")
ax1.plot(x, df["tracker_drift"], color=MPL_THEME["text_sec"], linewidth=0.9, alpha=0.58, linestyle=":", label=f"Tracker drift x{results['w_td']:.2f}")
ax1.plot(x, df["composite"], color=MPL_THEME["alert"], linewidth=2.1, label="Composite")
ax1.axhline(threshold, color=MPL_THEME["stable"], linewidth=1.0, linestyle="--", label=f"Threshold {threshold:.2f}")
ax1.fill_between(x, threshold, df["composite"].values, where=(df["composite"].values >= threshold), alpha=0.10, color=MPL_THEME["alert"])
if trigger_frame:
    ax1.axvline(trigger_frame, color=MPL_THEME["stable"], linewidth=1.0, linestyle=":")
    ax1.text(trigger_frame + 1, min(threshold + 0.08, 0.96), f"retrain {trigger_frame}", color=MPL_THEME["stable"], fontsize=8, fontfamily="monospace")
ax1.set_xlabel("Frame", color=MPL_THEME["text_dim"], fontsize=9)
ax1.set_ylabel("Score", color=MPL_THEME["text_dim"], fontsize=9)
ax1.set_xlim(x[0], x[-1])
ax1.set_ylim(-0.02, 1.05)
ax1.legend(fontsize=8, loc="upper left", facecolor=MPL_THEME["surf"], edgecolor=MPL_THEME["bord"], labelcolor=MPL_THEME["text_sec"], framealpha=1)
fig1.tight_layout(pad=0.8)
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

st.markdown('<div class="section-title">Signal breakdown</div>', unsafe_allow_html=True)
fig2, axes = plt.subplots(1, 3, figsize=(14, 3.1), sharey=True)
fig2.patch.set_facecolor(MPL_THEME["bg"])
specs = [
    ("Feature shift", "feature_shift", MPL_THEME["telemetry"]),
    ("Uncertainty", "uncertainty", MPL_THEME["warning"]),
    ("Tracker drift", "tracker_drift", MPL_THEME["text_sec"]),
]
for ax, (title, col, color) in zip(axes, specs):
    apply_axis(ax)
    ax.plot(x, df[col], color=color, linewidth=1.35)
    ax.fill_between(x, df[col], alpha=0.08, color=color)
    ax.axhline(threshold, color=MPL_THEME["stable"], linewidth=0.8, linestyle="--", alpha=0.75)
    if trigger_frame:
        ax.axvline(trigger_frame, color=MPL_THEME["stable"], linewidth=0.8, linestyle=":", alpha=0.75)
    ax.set_title(title, color=MPL_THEME["text_sec"], fontsize=10, loc="left", fontfamily="monospace")
    ax.set_xlabel("Frame", color=MPL_THEME["text_dim"], fontsize=8)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-0.02, 1.05)
axes[0].set_ylabel("Score", color=MPL_THEME["text_dim"], fontsize=8)
fig2.tight_layout(pad=0.6)
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

st.markdown('<div class="section-title">Sustained threshold gate</div>', unsafe_allow_html=True)
fig3, ax3 = plt.subplots(figsize=(14, 2.5))
fig3.patch.set_facecolor(MPL_THEME["bg"])
apply_axis(ax3)
ax3.step(x, df["consecutive"], color=MPL_THEME["alert"], linewidth=1.3, where="post")
ax3.fill_between(x, df["consecutive"], alpha=0.09, color=MPL_THEME["alert"], step="post")
ax3.axhline(results["sustained_frames"], color=MPL_THEME["stable"], linewidth=1.0, linestyle="--", label=f"Trigger gate {int(results['sustained_frames'])} frames")
ax3.set_xlabel("Frame", color=MPL_THEME["text_dim"], fontsize=8)
ax3.set_ylabel("Consecutive", color=MPL_THEME["text_dim"], fontsize=8)
ax3.set_xlim(x[0], x[-1])
ax3.legend(fontsize=8, facecolor=MPL_THEME["surf"], edgecolor=MPL_THEME["bord"], labelcolor=MPL_THEME["text_sec"], framealpha=1)
fig3.tight_layout(pad=0.6)
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

with st.expander("Fusion rule"):
    st.code(
        f"""composite = (
    {results['w_fs']:.2f} * feature_shift
  + {results['w_uc']:.2f} * uncertainty
  + {results['w_td']:.2f} * tracker_drift
)

trigger = (
    composite >= {threshold:.2f}
    and consecutive_frames >= {int(results['sustained_frames'])}
    and buffer_size >= {int(results['min_buffer'])}
)""",
        language="text",
    )

st.markdown("</div>", unsafe_allow_html=True)
render_footer()
