"""
SignalTrust Dashboard  -  Step 4 (v2)
======================================
Usage (from project root):
    streamlit run app/app.py -- --oof outputs/phase2/oof_predictions.csv
    streamlit run app/app.py -- --oof outputs/phase2/oof_predictions.csv --wesad-root <S2..S17 dir>
"""

import argparse, sys, time, pickle, os
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

FS_BVP        = 64
WIN_S         = 8
DEFAULT_THRESH = 0.70

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignalTrust Dashboard",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg:      #F4F3EF;
  --card:    #FFFFFF;
  --border:  #E6E4DD;
  --text:    #18181B;
  --muted:   #71717A;
  --violet:  #8B5CF6;
  --sky:     #0EA5E9;
  --mint:    #10B981;
  --rose:    #F43F5E;
  --amber:   #F59E0B;
  --shadow:  0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
  --radius:  14px;
}

/* --- App shell --- */
.stApp { background: var(--bg) !important; font-family: 'Inter', sans-serif; color: var(--text); }
[data-testid="stSidebar"] { background: var(--card) !important; border-right: 1px solid var(--border) !important; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; max-width: 1300px; }

/* --- Cards --- */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.1rem 1.3rem;
  height: 100%;
  box-sizing: border-box;
}

/* --- Metric cards (fixed height for uniformity) --- */
.mcard {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.15rem 1.25rem;
  min-height: 130px;
  display: flex; flex-direction: column; justify-content: space-between;
}
.mcard .lbl {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 4px;
}
.mcard .val { font-size: 1.85rem; font-weight: 800; line-height: 1.05; }
.mcard .sub { font-size: 0.73rem; color: var(--muted); margin-top: 4px; }
.mcard .explain {
  font-size: 0.71rem; color: #A1A1AA;
  border-top: 1px solid var(--border); padding-top: 7px; margin-top: 8px;
}

/* --- HR main card --- */
.hr-main {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 1.15rem 1.25rem;
  min-height: 130px;
  display: flex; flex-direction: column; justify-content: space-between;
}
.hr-badge {
  border-radius: 10px; padding: 0.7rem 0.9rem;
  text-align: center; font-family: 'Inter', sans-serif;
}
.hr-badge .bpm { font-size: 2.2rem; font-weight: 800; line-height: 1; display: block; }
.hr-badge .tag { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; display: block; }
.badge-g { background: #F0FDF4; border: 1.5px solid #86EFAC; color: #166534; }
.badge-r { background: #FFF1F2; border: 1.5px solid #FCA5A5; color: #9F1239; }
.badge-n { background: #F4F4F5; border: 1.5px solid #D4D4D8; color: #71717A; }

/* --- Quality bar --- */
.qbar-bg { width:100%; height:8px; border-radius:999px; background:#E4E4E7; margin: 5px 0 3px; }
.qbar-fill { height:100%; border-radius:999px; }

/* --- Section header --- */
.sec-hdr {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 0.45rem;
}

/* --- Pills --- */
.pill { display:inline-block; border-radius:999px; padding:2px 10px; font-size:0.72rem; font-weight:700; }
.pill-g { background:#DCFCE7; color:#166534; }
.pill-r { background:#FFE4E6; color:#9F1239; }

/* --- Disclaimer --- */
.disc {
  font-size: 0.68rem; color: var(--muted);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 0.5rem 0.75rem; background: #FAFAFA; line-height: 1.5;
}

/* --- Sidebar logo --- */
.logo-row { display:flex; align-items:center; gap:9px; margin-bottom:1.2rem; }
.logo-row .name { font-size:1rem; font-weight:700; color:var(--text); }
.logo-row .tagline { font-size:0.68rem; color:var(--muted); font-weight:500; }

/* Progress bar */
.prog-wrap { height:6px; border-radius:999px; background:#E4E4E7; margin: 6px 0; }
.prog-fill  { height:100%; border-radius:999px; background: linear-gradient(90deg,#8B5CF6,#0EA5E9); }

/* --- Hide Streamlit chrome --- */
[data-testid="stToolbar"] { display:none !important; }
footer { visibility: hidden !important; }
div[data-testid="stSlider"] label { font-size:0.78rem !important; font-weight:600 !important; }
div[data-testid="stSelectbox"] label { font-size:0.78rem !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)


# ── CLI args / path resolution ───────────────────────────────────────────────
# On Streamlit Community Cloud there are no CLI args. We probe several
# candidate paths so the app works both locally and in the cloud.
_OOF_CANDIDATES = [
    "data/oof_predictions.csv",               # committed data/ folder (cloud)
    "outputs/phase2/oof_predictions.csv",     # local standard layout
    "out/oof_predictions.csv",                # legacy local layout
]

def get_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--oof",        default=None)
    p.add_argument("--wesad-root", default=None)
    try:
        idx = sys.argv.index("--")
        args, _ = p.parse_known_args(sys.argv[idx + 1:])
    except ValueError:
        args, _ = p.parse_known_args([])
    # auto-discover OOF path if not given
    if args.oof is None:
        for cand in _OOF_CANDIDATES:
            if os.path.exists(cand):
                args.oof = cand
                break
        else:
            args.oof = _OOF_CANDIDATES[0]  # will give a clear error below
    return args

args = get_args()



# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading predictions…")
def load_oof(path):
    df = pd.read_csv(path)
    return df.sort_values(["subject","t0"]).reset_index(drop=True)


def bp(x, lo=0.5, hi=8, fs=64, order=3):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)


@st.cache_resource(show_spinner=False)
def load_bvp(wesad_root, subject):
    if not wesad_root:
        return None
    pat = os.path.join(wesad_root, subject, f"{subject}.pkl")
    if not os.path.exists(pat):
        return None
    try:
        with open(pat, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        raw = np.asarray(d["signal"]["wrist"]["BVP"]).ravel().astype(float)
        return bp(raw)
    except Exception:
        return None


def bvp_seg(arr, t0):
    if arr is None:
        return None
    s, e = int(t0*FS_BVP), int(t0*FS_BVP) + WIN_S*FS_BVP
    return arr[s:e] if e <= len(arr) else None


def running_stats(sub_df, thresh, idx):
    past  = sub_df.iloc[:idx+1]
    valid = past[past.err.notna()]
    u_mae = valid.err.mean() if len(valid) else float("nan")
    gated = valid[valid.p_cal >= thresh]
    g_mae = gated.err.mean()  if len(gated) else float("nan")
    cov   = len(gated)/len(valid) if len(valid) else float("nan")
    return u_mae, g_mae, cov


def qcolor(p):
    return "#10B981" if p >= 0.70 else ("#F59E0B" if p >= 0.50 else "#F43F5E")


def nan_str(v, fmt=".1f"):
    return format(v, fmt) if not np.isnan(float(v)) else "—"


# ── Charts ────────────────────────────────────────────────────────────────────
def chart_layout(fig, h=220):
    fig.update_layout(
        height=h,
        margin=dict(l=4,r=4,t=6,b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11, color="#18181B"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04,
            xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E6E4DD", borderwidth=1,
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(gridcolor="#F0EEE8", zeroline=False, showline=False)
    fig.update_yaxes(gridcolor="#F0EEE8", zeroline=False, showline=False)


def fig_mae_curve(sub_df):
    ev = sub_df[sub_df.err.notna()].copy()
    if len(ev) < 5:
        return None
    fracs = np.linspace(0.1, 1.0, 10)

    def curve(score):
        e = ev.err.values[np.argsort(-np.asarray(score), kind="stable")]
        return [e[:max(1, int(round(f*len(e))))].mean() for f in fracs]

    rng = np.random.default_rng(42)
    fig = go.Figure()
    data = [
        ("Model (calibrated)",  ev.p_cal.values,        "#8B5CF6", 3, "solid"),
        ("SNR gate",            ev.snr.values,           "#0EA5E9", 2, "dash"),
        ("Random baseline",     rng.random(len(ev)),    "#D4D4D8", 1.5, "dot"),
    ]
    for name, score, color, w, dash in data:
        fig.add_trace(go.Scatter(
            x=fracs*100, y=curve(score), name=name, mode="lines+markers",
            line=dict(color=color, width=w, dash=dash),
            marker=dict(size=5, color=color),
        ))
    # annotate the model@50% point
    m50 = curve(ev.p_cal.values)[4]
    fig.add_annotation(
        x=50, y=m50, text=f"Model {m50:.1f} bpm", showarrow=True,
        arrowhead=2, arrowcolor="#8B5CF6", font=dict(size=10, color="#8B5CF6"),
        bgcolor="#F5F3FF", bordercolor="#8B5CF6", borderwidth=1, ay=-28,
    )
    chart_layout(fig, h=250)
    fig.update_xaxes(title="Coverage kept (% of windows reported)")
    fig.update_yaxes(title="HR error — MAE (bpm)")
    return fig


def fig_quality_timeline(sub_df, idx, thresh):
    past = sub_df.iloc[:idx+1]
    t, p = past.t0.values, past.p_cal.values
    acc  = p >= thresh

    fig = go.Figure()
    # threshold band (fill below)
    fig.add_hrect(y0=0, y1=thresh, fillcolor="rgba(244,63,94,0.04)", line_width=0)
    fig.add_hrect(y0=thresh, y1=1.05, fillcolor="rgba(16,185,129,0.04)", line_width=0)
    fig.add_hline(y=thresh, line=dict(color="#71717A", width=1.2, dash="dash"),
                  annotation_text=f"Gate threshold {thresh:.2f}",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#71717A"))
    if len(t) > 1:
        fig.add_trace(go.Scatter(
            x=t, y=p, mode="lines",
            line=dict(color="#C4B5FD", width=1.5),
            showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=t[acc], y=p[acc], mode="markers", name="HR reported",
        marker=dict(color="#10B981", size=7, line=dict(color="#fff", width=1.2)),
    ))
    fig.add_trace(go.Scatter(
        x=t[~acc], y=p[~acc], mode="markers", name="HR withheld",
        marker=dict(color="#F43F5E", size=7, line=dict(color="#fff", width=1.2)),
    ))
    if len(t):
        fig.add_vline(x=t[-1], line=dict(color="#F59E0B", width=2, dash="dot"),
                      annotation_text="now", annotation_position="top",
                      annotation_font=dict(size=9, color="#F59E0B"))
    chart_layout(fig, h=220)
    fig.update_xaxes(title="Time (seconds into recording)")
    fig.update_yaxes(title="Quality score", range=[0, 1.08])
    return fig


def fig_bvp(seg, t0, gate_pass):
    t = np.linspace(t0, t0+WIN_S, len(seg))
    color = "#10B981" if gate_pass else "#F43F5E"
    fill  = "rgba(16,185,129,0.08)" if gate_pass else "rgba(244,63,94,0.08)"
    mn, mx = seg.min(), seg.max()
    pad    = (mx - mn)*0.12 + 1e-6

    fig = go.Figure()
    fig.add_shape(type="rect", x0=t[0], x1=t[-1], y0=mn-pad, y1=mx+pad,
                  fillcolor=fill, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=t, y=seg, mode="lines",
                             line=dict(color=color, width=2)))
    chart_layout(fig, h=190)
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="BVP (filtered)", showticklabels=False)
    fig.update_layout(showlegend=False)
    return fig


def fig_feature_bars(row, columns):
    """Show signal quality features available in oof_predictions.csv.
    Only uses columns that are actually present: snr, acc_std, p_cal.
    """
    def safe(col, default=0.0):
        return float(row[col]) if col in columns and not pd.isna(row[col]) else default

    snr_v = safe("snr", 0.0)
    acc_v = safe("acc_std", 0.0)
    pcal_v = safe("p_cal", 0.0)

    # Build only the features we have
    feats = {}
    feats["Quality score (p_cal)"] = (pcal_v, 0, 1)
    feats["SNR (spectral)"]        = (snr_v,  0, 1)
    feats["Acc calm (1-norm)"]     = (max(0.0, 1.0 - acc_v / 3.0), 0, 1)

    # Add optional features if present
    for col, label, lo, hi, default in [
        ("tmpl_corr", "Template corr",    0, 1, 0.0),
        ("ibi_cv",   "IBI regularity",    0, 1, 0.0),
    ]:
        if col in columns:
            v = safe(col, default)
            if col == "ibi_cv":
                v = max(0.0, 1.0 - v)  # invert: low CV = regular
            feats[label] = (v, lo, hi)

    labels, vals, colors = [], [], []
    for name, (v, lo, hi) in feats.items():
        labels.append(name)
        normed = max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-9)))
        vals.append(round(normed, 3))
        colors.append("#10B981" if normed > 0.6 else ("#F59E0B" if normed > 0.35 else "#F43F5E"))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in vals],
        textposition="outside",
    ))
    fig.update_xaxes(range=[0, 1.18], showticklabels=False, showgrid=False)
    fig.update_yaxes(showgrid=False)
    chart_layout(fig, h=max(130, len(feats) * 48))
    fig.update_layout(showlegend=False, bargap=0.35)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    # ── resolve OOF path ────────────────────────────────────────────────────
    candidates = [
        args.oof,
        "outputs/phase2/oof_predictions.csv",
        "out/oof_predictions.csv",
        "../out/oof_predictions.csv",
    ]
    oof_path = next((c for c in candidates if os.path.exists(c)), None)
    if oof_path is None:
        st.error("Cannot find `oof_predictions.csv`. Run:\n\n"
                 "```\nstreamlit run app/app.py -- --oof outputs/phase2/oof_predictions.csv\n```")
        st.stop()

    df_all = load_oof(oof_path)

    # ── init session state (BEFORE sidebar reads it) ─────────────────────────
    if "replay_idx" not in st.session_state:
        st.session_state.replay_idx     = 0
        st.session_state.replay_running = False
        st.session_state.replay_sub     = None

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="logo-row"><span style="font-size:1.7rem;">💓</span>'
            '<div><div class="name">SignalTrust</div>'
            '<div class="tagline">Quality-Aware HR Monitor</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Subject**")
        subjects = sorted(df_all.subject.unique())
        subject  = st.selectbox("Subject", subjects, label_visibility="collapsed")

        # reset on subject switch
        if st.session_state.replay_sub != subject:
            st.session_state.replay_idx     = 0
            st.session_state.replay_running = False
            st.session_state.replay_sub     = subject

        sub_df   = df_all[df_all.subject == subject].reset_index(drop=True)
        n_win    = len(sub_df)

        st.markdown("**Quality gate threshold**")
        thresh = st.slider(
            "thresh", 0.30, 0.95, DEFAULT_THRESH, 0.01,
            label_visibility="collapsed",
            help="Windows with quality score ≥ threshold get a HR reading. Below → withheld.",
        )
        st.caption(f"Recommended: **0.70** (false-accept ≤ 5%) · Current: **{thresh:.2f}**")

        st.markdown("**Replay speed**")
        speed = st.selectbox(
            "speed", ["0.25 s", "0.5 s", "1 s", "2 s", "Manual"],
            index=2, label_visibility="collapsed",
        )
        delay = {"0.25 s": 0.25, "0.5 s": 0.5, "1 s": 1.0, "2 s": 2.0, "Manual": None}[speed]

        st.divider()

        ca, cb, cc = st.columns(3)
        with ca:
            play_clicked = st.button("▶ Play", use_container_width=True, type="primary")
        with cb:
            pause_clicked = st.button("⏸ Pause", use_container_width=True)
        with cc:
            reset_clicked = st.button("↺ Reset", use_container_width=True)

        if play_clicked:
            st.session_state.replay_running = True
        if pause_clicked:
            st.session_state.replay_running = False
        if reset_clicked:
            st.session_state.replay_idx     = 0
            st.session_state.replay_running = False

        if delay is None:
            st.session_state.replay_idx = st.slider(
                "Window", 0, n_win-1, st.session_state.replay_idx,
                key="man_slider",
            )

        # replay progress
        prog_pct = int(100 * st.session_state.replay_idx / max(n_win-1, 1))
        st.markdown(
            f'<div style="font-size:0.72rem;color:#71717A;margin-top:6px;">'
            f'Window {st.session_state.replay_idx+1} / {n_win}'
            f' &nbsp;·&nbsp; {"▶ Playing" if st.session_state.replay_running else "⏸ Paused"}</div>'
            f'<div class="prog-wrap"><div class="prog-fill" style="width:{prog_pct}%;"></div></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # how to read legend
        st.markdown("""
        <div style="font-size:0.72rem;color:#71717A;line-height:1.7;">
          <strong style="color:#18181B;">How to read</strong><br>
          🟢 <b>Quality ≥ threshold</b> → HR is reported<br>
          🔴 <b>Quality &lt; threshold</b> → HR withheld (abstain)<br>
          <b>MAE curve</b>: lower = better. At 50% coverage the model keeps only the cleanest half of windows.
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown(
            '<div class="disc">⚠️ <b>Research prototype</b>. Replay of pre-computed '
            'WESAD LOSO scores. Not for clinical use. '
            '<a href="https://github.com/solitude-Yugen/signaltrust" '
            'style="color:#8B5CF6;">GitHub</a></div>',
            unsafe_allow_html=True,
        )

    # ── current window ────────────────────────────────────────────────────────
    idx       = st.session_state.replay_idx
    row       = sub_df.iloc[idx]
    t0        = float(row.t0)
    p_cal     = float(row.p_cal)
    hr_ppg    = float(row.hr_ppg) if not pd.isna(row.hr_ppg) else float("nan")
    hr_ecg    = float(row.hr_ecg) if not pd.isna(row.hr_ecg) else float("nan")
    gate_pass = p_cal >= thresh
    has_hr    = not np.isnan(hr_ppg)

    u_mae, g_mae, cov = running_stats(sub_df, thresh, idx)

    bvp_arr = load_bvp(args.wesad_root, subject)
    seg     = bvp_seg(bvp_arr, t0)

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        f'margin-bottom:0.8rem;">'
        f'<div>'
        f'<h2 style="margin:0;font-size:1.4rem;font-weight:800;letter-spacing:-0.02em;">'
        f'💓 SignalTrust  <span style="font-size:0.85rem;font-weight:500;color:#71717A;">'
        f'— {subject}</span></h2>'
        f'<p style="margin:4px 0 0;font-size:0.78rem;color:#71717A;">'
        f'Window <b>{idx+1}/{n_win}</b> &nbsp;·&nbsp; '
        f't = {int(t0)}–{int(t0)+WIN_S} s of recording &nbsp;·&nbsp; '
        f'8 s windows, 2 s stride &nbsp;·&nbsp; WESAD LOSO</p></div>'
        f'<div style="text-align:right;font-size:0.7rem;color:#A1A1AA;line-height:1.6;">'
        f'Gate threshold: <b style="color:#18181B;">{thresh:.2f}</b><br>'
        f'AUROC (WESAD): <b style="color:#8B5CF6;">0.864</b></div></div>',
        unsafe_allow_html=True,
    )

    # ── metric row (5 uniform cards) ──────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)

    # card 1 – HR output
    if gate_pass and has_hr:
        bcls, bpm_s, btag = "badge-g", f"{hr_ppg:.0f}", "TRUSTED ✓"
    elif not gate_pass:
        bcls, bpm_s, btag = "badge-r", "—", "WITHHELD ✕"
    else:
        bcls, bpm_s, btag = "badge-n", "—", "NO ESTIMATE"
    ecg_s = nan_str(hr_ecg) + " bpm" if not np.isnan(hr_ecg) else "—"

    with m1:
        st.markdown(
            f'<div class="mcard">'
            f'<div class="lbl">Heart Rate Output</div>'
            f'<div class="hr-badge {bcls}">'
            f'<span class="bpm">{bpm_s}</span>'
            f'<span class="tag">{btag}</span></div>'
            f'<div class="explain">ECG reference: <b>{ecg_s}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # card 2 – quality score
    qc   = qcolor(p_cal)
    qpct = int(p_cal * 100)
    with m2:
        st.markdown(
            f'<div class="mcard">'
            f'<div class="lbl">Quality Score</div>'
            f'<div class="val" style="color:{qc};">{p_cal:.2f}</div>'
            f'<div class="qbar-bg"><div class="qbar-fill" style="width:{qpct}%;background:{qc};"></div></div>'
            f'<div class="sub">Gate: {"PASS ✓" if gate_pass else "FAIL ✕"} (thresh {thresh:.2f})</div>'
            f'<div class="explain">Calibrated model probability that PPG HR is within 5 bpm of ECG truth</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # card 3 – gated MAE
    gm_s = nan_str(g_mae) + " bpm" if not np.isnan(g_mae) else "—"
    um_s = nan_str(u_mae) + " bpm" if not np.isnan(u_mae) else "—"
    improvement = u_mae - g_mae if (not np.isnan(g_mae) and not np.isnan(u_mae)) else float("nan")
    with m3:
        st.markdown(
            f'<div class="mcard">'
            f'<div class="lbl">Gated MAE (so far)</div>'
            f'<div class="val" style="color:#8B5CF6;">{gm_s}</div>'
            f'<div class="sub">vs ungated {um_s}</div>'
            f'<div class="explain">{"↓ {:.1f} bpm improvement from gating".format(improvement) if not np.isnan(improvement) else "No gated windows yet"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # card 4 – coverage
    cov_s  = f"{cov:.0%}" if not np.isnan(cov) else "—"
    n_rep  = int(sub_df.iloc[:idx+1][sub_df.iloc[:idx+1].p_cal >= thresh].shape[0])
    with m4:
        st.markdown(
            f'<div class="mcard">'
            f'<div class="lbl">Coverage (so far)</div>'
            f'<div class="val" style="color:#0EA5E9;">{cov_s}</div>'
            f'<div class="sub">{n_rep} / {idx+1} windows reported</div>'
            f'<div class="explain">Fraction of windows where HR is output (not withheld)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # card 5 – subject overview
    full_un  = sub_df[sub_df.err.notna()].err.mean()
    full_ga  = sub_df[(sub_df.p_cal >= thresh) & sub_df.err.notna()].err.mean()
    usable_r = sub_df.usable.mean()
    full_un_s = f"{full_un:.1f}" if not np.isnan(float(full_un)) else "—"
    full_ga_s = f"{full_ga:.1f}" if not np.isnan(float(full_ga)) else "—"
    with m5:
        st.markdown(
            f'<div class="mcard">'
            f'<div class="lbl">Subject Overview</div>'
            f'<div class="val" style="color:#F59E0B;">{usable_r:.0%}</div>'
            f'<div class="sub">true usable rate (ECG label)</div>'
            f'<div class="explain">Full-session: ungated {full_un_s} bpm → gated {full_ga_s} bpm (p≥{thresh:.2f})</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── BVP waveform  |  Quality timeline ────────────────────────────────────
    pill_cls = "pill-g" if gate_pass else "pill-r"
    pill_txt = "✓ Gate PASS — HR reported" if gate_pass else "✕ Gate FAIL — HR withheld"

    col_bvp, col_tl = st.columns([1, 1])

    with col_bvp:
        st.markdown(
            f'<div class="card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
            f'<span class="sec-hdr">Signal window — t = {int(t0)}–{int(t0+WIN_S)} s</span>'
            f'<span class="pill {pill_cls}">{pill_txt}</span></div>',
            unsafe_allow_html=True,
        )
        if seg is not None:
            st.plotly_chart(fig_bvp(seg, t0, gate_pass),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            # No raw BVP — show feature quality bars instead
            st.markdown(
                '<div style="font-size:0.72rem;color:#71717A;margin-bottom:4px;">'
                'BVP waveform not loaded (pass <code>--wesad-root</code>). '
                'Showing pre-computed signal features instead:</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_feature_bars(row, sub_df.columns),
                            use_container_width=True, config={"displayModeBar": False})
            snr_disp = f'{row["snr"]:.2f}' if "snr" in sub_df.columns else "—"
            acc_disp = f'{row["acc_std"]:.2f}' if "acc_std" in sub_df.columns else "—"
            st.markdown(
                f'<div style="font-size:0.72rem;color:#71717A;margin-top:2px;">'
                f'SNR = {snr_disp} &nbsp;·&nbsp; Acc std = {acc_disp} &nbsp;·&nbsp; '
                f'Quality score (p_cal) = {p_cal:.2f}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_tl:
        st.markdown(
            '<div class="card">'
            '<div class="sec-hdr" style="margin-bottom:6px;">Quality score over time</div>'
            '<div style="font-size:0.71rem;color:#71717A;margin-bottom:6px;">'
            '🟢 green dots = HR was reported &nbsp;|&nbsp; '
            '🔴 red dots = HR withheld &nbsp;|&nbsp; '
            'dashed line = gate threshold &nbsp;|&nbsp; '
            'amber line = current window</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_quality_timeline(sub_df, idx, thresh),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── MAE vs Coverage ───────────────────────────────────────────────────────
    st.markdown(
        '<div class="card" style="margin-top:0;">'
        '<div class="sec-hdr">MAE vs. Coverage — lower is better; rightward = more windows reported</div>'
        '<div style="font-size:0.71rem;color:#71717A;margin-bottom:6px;">'
        'X-axis: % of this subject\'s windows kept (sorted best→worst by the gate score). '
        'Y-axis: mean absolute HR error (bpm) on those windows. '
        'At 50% coverage, the model keeps only the cleanest half. '
        'A good gate should have <b>low MAE even at high coverage</b>.</div>',
        unsafe_allow_html=True,
    )
    mae_fig = fig_mae_curve(sub_df)
    if mae_fig:
        st.plotly_chart(mae_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("Not enough windows to plot.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recent windows table ──────────────────────────────────────────────────
    with st.expander("🔍 Recent window log (last 15 windows)"):
        st.markdown(
            '<div style="font-size:0.72rem;color:#71717A;margin-bottom:6px;">'
            'Each row = one 8 s window. Gate = PASS if p_cal ≥ threshold. '
            'HR output is shown only when gate passes AND estimator returns a value.</div>',
            unsafe_allow_html=True,
        )
        recent = sub_df.iloc[max(0, idx-14):idx+1].copy()
        recent["Gate"]      = recent.p_cal.apply(lambda p: "✓ PASS" if p >= thresh else "✕ FAIL")
        recent["HR output"] = recent.apply(
            lambda r: f"{r.hr_ppg:.1f}" if (r.p_cal >= thresh and not pd.isna(r.hr_ppg)) else "—", axis=1
        )
        recent["ECG ref"]   = recent.hr_ecg.apply(lambda v: f"{v:.1f}" if not pd.isna(v) else "—")
        recent["Err (bpm)"] = recent.err.apply(lambda v: f"{v:.1f}" if not pd.isna(v) else "—")
        # only include columns that actually exist in the loaded CSV
        avail = list(sub_df.columns)
        show_cols = [c for c in ["t0","p_cal","Gate","HR output","ECG ref","Err (bpm)","snr","acc_std"] if c in avail or c in ["Gate","HR output","ECG ref","Err (bpm)"]]
        rename_map = {"t0":"t₀ (s)","snr":"SNR","acc_std":"Acc std"}
        st.dataframe(
            recent[show_cols].rename(columns=rename_map)
                             .iloc[::-1].reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

    # ── replay advance (at the very end, after all rendering) ─────────────────
    if st.session_state.replay_running and delay is not None:
        if idx < n_win - 1:
            time.sleep(delay)
            st.session_state.replay_idx = idx + 1
            st.rerun()
        else:
            st.session_state.replay_running = False
            st.rerun()


if __name__ == "__main__":
    main()
