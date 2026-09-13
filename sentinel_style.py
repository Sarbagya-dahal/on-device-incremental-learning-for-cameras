"""
Shared visual system for Sentinel Edge Monitor.

The CSS styles Streamlit content and navigation without hiding the native
header or collapsed-sidebar control.
"""

import streamlit as st

BG = "#0A0E14"
SURF = "#111824"
BORD = "#253449"

MPL_THEME = {
    "bg": BG,
    "surf": SURF,
    "bord": BORD,
    "text_dim": "#73869B",
    "text_sec": "#A8B8C8",
    "stable": "#66C2A3",
    "warning": "#D08B5B",
    "alert": "#D16060",
    "telemetry": "#6EA8D9",
}


def apply_style():
    st.markdown(_CSS, unsafe_allow_html=True)


def score_color(value: float, threshold: float) -> str:
    if value >= threshold:
        return MPL_THEME["alert"]
    if value >= threshold * 0.8:
        return MPL_THEME["warning"]
    return MPL_THEME["stable"]


def state_badge(state: str) -> str:
    cfg = {
        "MODEL STABLE": ("Stable", "b-stable"),
        "DRIFT WARNING": ("Watch", "b-warning"),
        "DRIFT DETECTED": ("Drift", "b-alert"),
        "RETRAINING TRIGGERED": ("Retraining", "b-training"),
        "MODEL UPDATED": ("Updated", "b-updated"),
        "ROLLBACK": ("Rolled Back", "b-rollback"),
    }
    label, cls = cfg.get(state, (state.title(), "b-stable"))
    return f"<span class='badge {cls}'>{label}</span>"


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --base: #0A0E14;
    --surface: #111824;
    --surface-hi: #172233;
    --panel-deep: #0D131C;
    --border: #253449;
    --border-hi: #3B5572;
    --text-primary: #D7E3EF;
    --text-sec: #A8B8C8;
    --text-dim: #73869B;
    --stable: #66C2A3;
    --warning: #D08B5B;
    --alert: #D16060;
    --telemetry: #6EA8D9;
    --font-ui: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'IBM Plex Mono', 'Cascadia Mono', Consolas, monospace;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--base);
    color: var(--text-primary);
    font-family: var(--font-ui);
    font-size: 15px;
    line-height: 1.55;
}

[data-testid="stHeader"] {
    background: rgba(10, 14, 20, 0.86);
    border-bottom: 1px solid rgba(37, 52, 73, 0.65);
}

[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    visibility: hidden !important;
}

.block-container {
    max-width: 100% !important;
    padding: 0 0 44px 0 !important;
}

section[data-testid="stSidebar"] {
    background: #0C121B;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebarNav"] {
    border-bottom: 1px solid var(--border);
    padding-top: 4px;
}

[data-testid="stSidebarNav"] ul {
    padding: 4px 0 8px;
}

[data-testid="stSidebarNav"] a {
    color: var(--text-sec) !important;
    font-family: var(--font-ui);
    font-size: 13px;
    font-weight: 500;
    border-left: 3px solid transparent;
    padding: 9px 22px;
}

[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] a:focus-visible {
    color: var(--text-primary) !important;
    background: rgba(110, 168, 217, 0.08);
    outline: 2px solid var(--border-hi);
    outline-offset: -2px;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    color: var(--telemetry) !important;
    border-left-color: var(--telemetry);
    background: rgba(110, 168, 217, 0.10);
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--text-sec) !important;
    font-family: var(--font-ui);
}

[data-testid="stSidebar"] hr, hr {
    border-color: var(--border) !important;
}

.sentinel-wordmark {
    padding: 18px 22px 14px;
    border-bottom: 1px solid var(--border);
}

.sw-name {
    display: block;
    color: var(--text-primary);
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 0;
}

.sw-tag {
    display: block;
    margin-top: 2px;
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 11px;
}

.sidebar-status {
    padding: 14px 22px;
    border-bottom: 1px solid var(--border);
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 12px;
}

.ss-model { color: var(--text-primary); }
.ss-acc { color: var(--telemetry); }

.page-header {
    min-height: 70px;
    padding: 16px 30px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #111824 0%, #0D131C 100%);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.ph-title {
    color: var(--text-primary);
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0;
}

.ph-sub {
    margin-top: 2px;
    color: var(--text-dim);
    font-size: 13px;
}

.ph-meta {
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 12px;
    text-align: right;
}

.page-body {
    padding: 24px 30px 0;
}

.mode-pill, .badge {
    display: inline-flex;
    align-items: center;
    border-radius: 4px;
    padding: 4px 9px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0;
    border: 1px solid transparent;
}

.mode-pill { color: var(--telemetry); background: rgba(110, 168, 217, 0.12); }
.mp-live { color: var(--stable); background: rgba(102, 194, 163, 0.13); }
.mp-replay { color: var(--warning); background: rgba(208, 139, 91, 0.14); }
.mp-synth { color: var(--telemetry); background: rgba(110, 168, 217, 0.12); }

.b-stable, .b-updated { color: var(--stable); background: rgba(102, 194, 163, 0.12); border-color: rgba(102, 194, 163, 0.30); }
.b-warning { color: var(--warning); background: rgba(208, 139, 91, 0.13); border-color: rgba(208, 139, 91, 0.32); }
.b-alert, .b-rollback { color: var(--alert); background: rgba(209, 96, 96, 0.12); border-color: rgba(209, 96, 96, 0.34); }
.b-training { color: var(--telemetry); background: rgba(110, 168, 217, 0.13); border-color: rgba(110, 168, 217, 0.32); }

.score-hero, .signal-panel, .feed-panel, .panel, .idle-panel, .val-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
}

.score-hero {
    min-height: 372px;
    padding: 26px;
    border-left: 5px solid var(--stable);
}

.score-hero.sh-warning { border-left-color: var(--warning); }
.score-hero.sh-alert { border-left-color: var(--alert); }

.sh-label, .sp-label, .metric-label, .section-title {
    color: var(--text-sec);
    font-size: 12px;
    font-weight: 600;
}

.section-title {
    margin: 26px 0 12px;
}

.sh-value {
    margin: 14px 0 8px;
    font-family: var(--font-mono);
    font-size: 68px;
    font-weight: 600;
    line-height: 1;
    transition: color 260ms ease;
}

.sh-thr, .sh-formula, .metric-sub, .sp-desc {
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 12px;
}

.sh-state { margin-top: 16px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.sh-formula { margin-top: 22px; padding-top: 16px; border-top: 1px solid var(--border); line-height: 1.8; }

.feed-panel {
    min-height: 372px;
    overflow: hidden;
    position: relative;
}

.feed-toolbar {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}

.feed-stage {
    min-height: 318px;
    position: relative;
    background:
        linear-gradient(rgba(110,168,217,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(110,168,217,0.06) 1px, transparent 1px),
        #080D13;
    background-size: 34px 34px;
}

.feed-reticle {
    position: absolute;
    inset: 34px;
    border: 1px solid rgba(110,168,217,0.28);
}

.bbox {
    position: absolute;
    border: 2px solid var(--stable);
    background: rgba(102,194,163,0.06);
}

.bbox.alert { border-color: var(--warning); background: rgba(208,139,91,0.08); }
.bbox span {
    position: absolute;
    top: -25px;
    left: -2px;
    background: #0D131C;
    border: 1px solid currentColor;
    color: inherit;
    padding: 2px 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    white-space: nowrap;
}

.signal-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-top: 14px;
}

.signal-panel { padding: 16px; }
.sp-value, .metric-value {
    margin: 6px 0;
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 28px;
    font-weight: 600;
    line-height: 1;
}
.sp-track, .bar-track { height: 4px; background: #0A0E14; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.sp-fill, .bar-fill { height: 100%; background: var(--telemetry); transition: width 360ms ease; }

.stat-strip, .buffer-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
}

.metric-card, .panel {
    padding: 16px;
}

.metric-card {
    background: var(--panel-deep);
    border: 1px solid var(--border);
    border-radius: 6px;
}

.det-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--panel-deep);
    border: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 12px;
}
.det-table th, .det-table td { padding: 9px 12px; border-bottom: 1px solid var(--border); text-align: left; }
.det-table th { color: var(--text-sec); background: var(--surface-hi); font-family: var(--font-ui); font-size: 12px; }
.det-table td { color: var(--text-sec); }
.accepted { color: var(--stable) !important; font-weight: 600; }
.rejected { color: var(--alert) !important; font-weight: 600; }

.idle-panel { margin: 26px 30px 0; padding: 48px 28px; text-align: center; }
.ip-score { color: var(--text-dim); font-family: var(--font-mono); font-size: 58px; font-weight: 600; }
.ip-title { color: var(--text-primary); font-size: 18px; font-weight: 700; }
.ip-desc { color: var(--text-sec); margin-top: 6px; }

.log-console {
    max-height: 360px;
    overflow-y: auto;
    white-space: pre-wrap;
    background: #070B10;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--stable);
    padding: 14px 16px;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1.75;
}

.arch-box {
    overflow-x: auto;
    white-space: pre;
    background: #070B10;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-dim);
    padding: 12px;
    font-family: var(--font-mono);
    font-size: 10px;
    line-height: 1.6;
}

.drift-tips {
    margin-bottom: 16px;
    border: 1px solid rgba(208,139,91,0.36);
    border-left: 5px solid var(--warning);
    border-radius: 6px;
    background: rgba(208,139,91,0.08);
    padding: 14px 16px;
}
.drift-tips .dt-label { color: var(--warning); font-weight: 700; margin-bottom: 6px; }
.drift-tips li { color: var(--text-sec); margin: 4px 0; }

.sentinel-footer {
    margin-top: 34px;
    padding: 14px 30px;
    border-top: 1px solid var(--border);
    color: var(--text-dim);
    font-family: var(--font-mono);
    font-size: 11px;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 5px !important;
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
}
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="slider"]:focus-visible {
    outline: 2px solid var(--telemetry) !important;
    outline-offset: 2px !important;
}

.stSlider, .stSelectbox, .stTextInput, .stNumberInput, .stCheckbox {
    color: var(--text-primary);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
}

@media (max-width: 980px) {
    .page-header { align-items: flex-start; flex-direction: column; }
    .ph-meta { text-align: left; }
    .page-body { padding: 18px 16px 0; }
    .signal-grid, .stat-strip, .buffer-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .sh-value { font-size: 52px; }
    .score-hero, .feed-panel { min-height: auto; }
}

@media (max-width: 620px) {
    .signal-grid, .stat-strip, .buffer-grid { grid-template-columns: 1fr; }
    .page-header { padding: 14px 16px; }
    .idle-panel { margin-left: 16px; margin-right: 16px; }
    .det-table { font-size: 11px; }
    .det-table th, .det-table td { padding: 7px 8px; }
}
</style>
"""
