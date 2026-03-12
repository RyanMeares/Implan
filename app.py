"""
Regional Economic Impact Analysis Tool  v4.0
Streamlit Application

Run:  streamlit run app.py
"""

import io
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

import model_engine as me

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Regional Economic Impact Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Top header bar */
.app-header {
    background: #0f172a;
    color: #e2e8f0;
    padding: 1.4rem 2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    border-left: 4px solid #3b82f6;
}
.app-header h1 {
    margin: 0; font-size: 1.5rem; font-weight: 700;
    color: #f8fafc; letter-spacing: -0.02em;
}
.app-header p {
    margin: 0.2rem 0 0; font-size: 0.82rem; color: #94a3b8;
    font-family: 'IBM Plex Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: #64748b; margin-bottom: 0.35rem;
}
.metric-card .value {
    font-size: 1.7rem; font-weight: 700; color: #f1f5f9;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card .sub {
    font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem;
}

/* Section header */
.section-header {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.4rem; margin-bottom: 1rem;
}

/* Impact type badges */
.badge-direct   { background:#1d4ed8; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.72rem; }
.badge-indirect { background:#15803d; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.72rem; }
.badge-induced  { background:#b45309; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.72rem; }
.badge-total    { background:#6d28d9; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.72rem; }

/* Status pill */
.status-ok  { background:#14532d; color:#86efac; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-family:monospace; }
.status-bad { background:#7f1d1d; color:#fca5a5; padding:2px 10px; border-radius:12px; font-size:0.72rem; font-family:monospace; }

/* Source chip */
.source-chip {
    background: #0f172a; border: 1px solid #334155;
    color: #94a3b8; font-size: 0.68rem;
    padding: 2px 8px; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; display: inline-block;
}

/* Table styling */
.stDataFrame { font-size: 0.82rem !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.78rem !important; }

/* hide streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────

if "model_state" not in st.session_state:
    st.session_state.model_state = None
if "results" not in st.session_state:
    st.session_state.results = None
if "run_params" not in st.session_state:
    st.session_state.run_params = {}

# ── HELPER FORMATTERS ─────────────────────────────────────────────────────────

def fmt_dollar(v: float) -> str:
    if abs(v) >= 1e9: return f"${v/1e9:,.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:,.1f}M"
    if abs(v) >= 1e3: return f"${v/1e3:,.0f}K"
    return f"${v:,.0f}"

def fmt_jobs(v: float) -> str:
    return f"{v:,.0f}"

def metric_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {"<div class='sub'>" + sub + "</div>" if sub else ""}
    </div>"""

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <h1>📊 Regional Economic Impact Analysis</h1>
  <p>v4.0 · BEA 2017 Benchmark I-O · BLS QCEW · Leontief Type I / II · York County, VA (FIPS 51199)</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**API Keys**")
    bea_key = st.text_input("BEA API Key", value="9F23F97C-0B7E-42F6-8A13-BDF994454ED4",
                             type="password", key="bea_key")
    bls_key = st.text_input("BLS API Key", value="4971172b2c1741848b68160b61aee083",
                             type="password", key="bls_key")

    st.markdown("---")
    st.markdown("**📂 Data Files**")
    st.caption("BEA Use Table (IOUse_After_Redefinitions_PRO_Detail.xlsx)")
    use_file = st.file_uploader("Use Table", type=["xlsx"],
                                 label_visibility="collapsed", key="uf")
    st.caption("Market Share Matrix (IxC_MS_Detail.xlsx)")
    ms_file  = st.file_uploader("Market Share Matrix", type=["xlsx"],
                                 label_visibility="collapsed", key="msf")
    st.caption("Domestic Requirements (CxI_Domestic_DR_Detail.xlsx)")
    dom_file = st.file_uploader("Domestic Requirements", type=["xlsx"],
                                 label_visibility="collapsed", key="df")
    st.caption("QCEW County File (e.g. 51199.csv)")
    qcew_file = st.file_uploader("QCEW CSV", type=["csv"],
                                  label_visibility="collapsed", key="qf")

    all_uploaded = all([use_file, ms_file, dom_file, qcew_file])

    st.markdown("---")
    if st.button("🔨 Build Model", disabled=not all_uploaded,
                 use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def _cb(msg, pct):
            progress_bar.progress(pct)
            status_text.caption(msg)

        with st.spinner("Building model…"):
            try:
                state = me.build_model(
                    use_file=use_file,
                    ms_file=ms_file,
                    dom_file=dom_file,
                    qcew_file=qcew_file,
                    bea_api_key=bea_key,
                    bls_api_key=bls_key,
                    progress_callback=_cb,
                )
                st.session_state.model_state = state
                st.session_state.results = None
                progress_bar.progress(100)
                status_text.caption("✅ Model ready.")
                st.success("Model built successfully!")
            except Exception as e:
                st.error(f"Build failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    if not all_uploaded:
        st.caption("Upload all 4 files to enable model build.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.68rem; color:#475569; line-height:1.6'>
    <b>Data sources</b><br>
    BEA 2017 Benchmark I-O Tables<br>
    BLS QCEW 2024 (county)<br>
    BLS CES 2022 (national)<br>
    BEA GDP-by-Industry (live API)<br>
    BEA NIPA Table 2.6 (MPC)
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT TABS ─────────────────────────────────────────────────────────

tab_run, tab_results, tab_sectors, tab_matrix, tab_coeffs, tab_export = st.tabs([
    "▶  Run Analysis",
    "📈  Results",
    "🏭  Sector Detail",
    "🔢  A Matrix",
    "🔬  Coefficients",
    "💾  Export",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_run:
    ms = st.session_state.model_state

    if ms is None:
        st.info("Upload all 4 data files and click **Build Model** in the sidebar to get started.")
        st.markdown("""
        **Required files:**
        | File | Source |
        |------|--------|
        | `IOUse_After_Redefinitions_PRO_Detail.xlsx` | [BEA I-O Accounts](https://www.bea.gov/industry/input-output-accounts-data) |
        | `IxC_MS_Detail.xlsx` | [BEA I-O Accounts](https://www.bea.gov/industry/input-output-accounts-data) |
        | `CxI_Domestic_DR_Detail.xlsx` | [BEA I-O Accounts](https://www.bea.gov/industry/input-output-accounts-data) |
        | `51199.csv` | [BLS QCEW](https://www.bls.gov/cew/downloadable-data.htm) |
        """)
        st.stop()

    st.markdown('<div class="section-header">Project Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        project_name = st.text_input("Project Name", value="New Investment Project",
                                      placeholder="e.g. Hampton Roads Logistics Hub")
        naics_input  = st.text_input("Primary NAICS Code (2–4 digit)",
                                      value="23",
                                      placeholder="e.g. 23, 336, 7211, 42")

        # Resolve NAICS to sector
        naics_valid = False
        primary_s = None
        try:
            primary_s = me.naics_to_sector(naics_input)
            naics_valid = True
            st.success(f"→ Sector [{primary_s}]: **{me.SECTOR_LABELS[primary_s]}**")
        except ValueError:
            if naics_input:
                st.error(f"NAICS '{naics_input}' not recognized. Try: 11, 21, 23, 31-33, 42, 44-45, 48-49, 51-56, 61-62, 71-72, 81, 92")

    with col2:
        investment_raw = st.number_input(
            "Total Investment ($)",
            min_value=100_000,
            max_value=10_000_000_000,
            value=10_000_000,
            step=500_000,
            format="%d",
        )
        induced_damp = st.slider(
            "Induced Dampening",
            min_value=0.35, max_value=0.65, value=0.50, step=0.05,
            help="How much of indirect wage income recirculates locally.\n"
                 "0.35–0.45 = small/rural county  |  "
                 "0.45–0.55 = suburban county (default)  |  "
                 "0.55–0.65 = large metro"
        )

    # Show spending profile for selected sector
    if naics_valid and primary_s is not None:
        st.markdown('<div class="section-header" style="margin-top:1.5rem">BEA Spending Profile — Sector Input Distribution</div>',
                    unsafe_allow_html=True)
        profile = ms["all_profiles"][primary_s]
        profile_sorted = sorted(profile.items(), key=lambda x: -x[1])

        prof_df = pd.DataFrame([
            {"Sector": f"[{s:2d}] {me.SECTOR_LABELS[s]}", "Share": v}
            for s, v in profile_sorted
        ])

        col_a, col_b = st.columns([1, 1.5])
        with col_a:
            st.dataframe(
                prof_df.style.format({"Share": "{:.1%}"})
                       .bar(subset=["Share"], color="#3b82f6"),
                hide_index=True, use_container_width=True, height=320
            )
        with col_b:
            fig_prof, ax_prof = plt.subplots(figsize=(6, 3.5))
            fig_prof.patch.set_facecolor("#0f172a")
            ax_prof.set_facecolor("#1e293b")
            labels_p = [me.SECTOR_LABELS[s][:22] for s, _ in profile_sorted[:10]]
            vals_p   = [v for _, v in profile_sorted[:10]]
            colors_p = ["#3b82f6"] * len(labels_p)
            colors_p[0] = "#60a5fa"
            ax_prof.barh(labels_p[::-1], vals_p[::-1], color=colors_p[::-1], edgecolor="none")
            ax_prof.set_xlabel("Share of Intermediate Inputs", color="#94a3b8", fontsize=9)
            ax_prof.tick_params(colors="#94a3b8", labelsize=8)
            for spine in ax_prof.spines.values(): spine.set_visible(False)
            ax_prof.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax_prof.grid(axis="x", color="#334155", alpha=0.5)
            fig_prof.tight_layout()
            st.pyplot(fig_prof, use_container_width=True)
            plt.close(fig_prof)

    st.markdown("---")

    run_disabled = not (naics_valid and project_name.strip())
    if st.button("▶  Run Impact Analysis", disabled=run_disabled,
                 type="primary", use_container_width=False):
        with st.spinner("Running Leontief model…"):
            # Build Y vector from spending profile
            profile = ms["all_profiles"][primary_s]
            Y = np.zeros(me.N)
            total_sh = sum(profile.values())
            for s, sh in profile.items():
                Y[s] += investment_raw * sh / total_sh

            results = me.compute_impacts(
                Y=Y,
                A_reg=ms["A_york"],
                emp_coeffs=ms["emp_coeffs"],
                pce_shares=ms["pce_shares"],
                ind_damp=induced_damp,
            )
            st.session_state.results = results
            st.session_state.run_params = {
                "project_name": project_name,
                "naics":        naics_input,
                "primary_s":    primary_s,
                "investment":   investment_raw,
                "dampening":    induced_damp,
                "profile":      profile,
            }
        st.success("Analysis complete — see Results tab.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_results:
    res = st.session_state.results
    rp  = st.session_state.run_params
    ms  = st.session_state.model_state

    if res is None:
        st.info("Run an analysis first (Run Analysis tab).")
        st.stop()

    t = res["total"]; d = res["direct"]; ii = res["indirect"]; ind = res["induced"]
    m = res["multipliers"]

    st.markdown(f"""
    <div style='margin-bottom:1rem'>
      <span style='font-size:1.1rem; font-weight:700; color:#f1f5f9'>{rp['project_name']}</span>
      &nbsp;&nbsp;
      <span class='source-chip'>NAICS {rp['naics']} · {me.SECTOR_LABELS[rp['primary_s']]}</span>
      &nbsp;
      <span class='source-chip'>{fmt_dollar(rp['investment'])} investment</span>
      &nbsp;
      <span class='source-chip'>dampening {rp['dampening']:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPI row ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Total Economic Impact</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    cards = [
        (k1, "Total Output",        fmt_dollar(t["output"]),       f"{t['output']/rp['investment']:.2f}× ROI"),
        (k2, "Total Jobs",           fmt_jobs(t["jobs"]),           f"Multiplier {m['emp']:.2f}×"),
        (k3, "GDP Contribution",     fmt_dollar(t["value_added"]),  f"{t['value_added']/rp['investment']:.1%} of investment"),
        (k4, "Labor Income",         fmt_dollar(t["labor_income"]), f"{t['labor_income']/t['output']:.1%} of output"),
        (k5, "Type II Multiplier",   f"{m['type2']:.3f}×",         f"Type I: {m['type1']:.3f}×"),
    ]
    for col, label, val, sub in cards:
        col.markdown(metric_card(label, val, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Impact table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Impact by Type</div>', unsafe_allow_html=True)
    impact_df = pd.DataFrame([
        {"Type": "Direct",   "Output": d["output"],  "Jobs": d["jobs"],  "Labor Income": d["labor_income"],  "Value Added": d["value_added"]},
        {"Type": "Indirect", "Output": ii["output"], "Jobs": ii["jobs"], "Labor Income": ii["labor_income"], "Value Added": ii["value_added"]},
        {"Type": "Induced",  "Output": ind["output"],"Jobs": ind["jobs"],"Labor Income": ind["labor_income"],"Value Added": ind["value_added"]},
        {"Type": "TOTAL",    "Output": t["output"],  "Jobs": t["jobs"],  "Labor Income": t["labor_income"],  "Value Added": t["value_added"]},
    ])

    def _fmt_row(row):
        return pd.Series({
            "Type":         row["Type"],
            "Output":       fmt_dollar(row["Output"]),
            "Jobs":         f"{row['Jobs']:,.1f}",
            "Labor Income": fmt_dollar(row["Labor Income"]),
            "Value Added":  fmt_dollar(row["Value Added"]),
        })
    st.dataframe(impact_df.apply(_fmt_row, axis=1), hide_index=True,
                 use_container_width=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:1.5rem">Visualizations</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0f172a")
    COLORS = {"direct":"#3b82f6","indirect":"#22c55e","induced":"#f59e0b","total":"#a855f7"}
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.grid(color="#334155", alpha=0.5, linewidth=0.5)

    # 1 – Output by type
    ax = axes[0,0]
    cats = ["Direct","Indirect","Induced"]
    vals = [d["output"]/1e6, ii["output"]/1e6, ind["output"]/1e6]
    bars = ax.bar(cats, vals, color=[COLORS["direct"],COLORS["indirect"],COLORS["induced"]], width=0.5)
    ax.set_title("Output by Type ($M)", color="#e2e8f0", fontweight="bold", fontsize=9)
    ax.set_ylabel("$M", color="#94a3b8", fontsize=8)
    ax.yaxis.label.set_color("#94a3b8")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02*max(vals),
                f"${b.get_height():.1f}M", ha="center", va="bottom", color="#e2e8f0", fontsize=8)

    # 2 – Jobs by type
    ax = axes[0,1]
    job_cats = ["Direct","Indirect","Induced","Total"]
    job_vals = [d["jobs"], ii["jobs"], ind["jobs"], t["jobs"]]
    clrs = [COLORS["direct"],COLORS["indirect"],COLORS["induced"],COLORS["total"]]
    b2 = ax.bar(job_cats, job_vals, color=clrs, width=0.5)
    ax.set_title("Jobs Created", color="#e2e8f0", fontweight="bold", fontsize=9)
    ax.set_ylabel("Jobs", color="#94a3b8", fontsize=8)
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f"{b.get_height():.0f}", ha="center", va="bottom", color="#e2e8f0", fontsize=8)

    # 3 – Multipliers
    ax = axes[0,2]
    mult_l = ["Type I\nOutput","Type II\nOutput","Employment","Labor\nIncome","Value\nAdded"]
    mult_v = [m["type1"], m["type2"], m["emp"], m["li"], m["va"]]
    b3 = ax.bar(mult_l, mult_v, color="#64748b", width=0.5)
    ax.axhline(y=1.0, color="#f87171", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_title("Economic Multipliers", color="#e2e8f0", fontweight="bold", fontsize=9)
    for b in b3:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{b.get_height():.3f}×", ha="center", va="bottom", color="#e2e8f0", fontsize=8)

    # 4 – Output composition pie
    ax = axes[1,0]
    sizes = [d["output"], ii["output"], ind["output"]]
    wedge_colors = [COLORS["direct"],COLORS["indirect"],COLORS["induced"]]
    ax.pie(sizes, labels=["Direct","Indirect","Induced"],
           colors=wedge_colors, autopct="%1.1f%%",
           textprops={"color":"#e2e8f0","fontsize":8},
           startangle=90, wedgeprops={"edgecolor":"#0f172a","linewidth":1.5})
    ax.set_title("Output Composition", color="#e2e8f0", fontweight="bold", fontsize=9)

    # 5 – Top sectors by total output
    ax = axes[1,1]
    sec_totals = [(me.SECTOR_LABELS[s], res["vecs"]["total"][s]/1e6)
                  for s in range(me.N) if res["vecs"]["total"][s] > 50_000]
    sec_totals.sort(key=lambda x: x[1], reverse=True)
    if sec_totals:
        lbls5, v5 = zip(*sec_totals[:10])
        ax.barh([l[:20] for l in lbls5][::-1], list(v5)[::-1],
                color="#3b82f6", edgecolor="none")
        ax.set_title("Top Sectors — Total Output ($M)", color="#e2e8f0", fontweight="bold", fontsize=9)
        ax.set_xlabel("$M", color="#94a3b8", fontsize=8)

    # 6 – Top sectors by jobs
    ax = axes[1,2]
    sec_jobs = [(me.SECTOR_LABELS[s],
                 res["direct"]["jobs_vec"][s]+res["indirect"]["jobs_vec"][s]+res["induced"]["jobs_vec"][s])
                for s in range(me.N)]
    sec_jobs = [(l,v) for l,v in sec_jobs if v > 0.05]
    sec_jobs.sort(key=lambda x: x[1], reverse=True)
    if sec_jobs:
        lbls6, v6 = zip(*sec_jobs[:10])
        ax.barh([l[:20] for l in lbls6][::-1], list(v6)[::-1],
                color="#22c55e", edgecolor="none")
        ax.set_title("Top Sectors — Total Jobs", color="#e2e8f0", fontweight="bold", fontsize=9)
        ax.set_xlabel("Jobs", color="#94a3b8", fontsize=8)

    fig.suptitle(f"{rp['project_name']}  ·  {fmt_dollar(rp['investment'])}  ·  York County, VA",
                 color="#f1f5f9", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Save chart for export
    chart_buf = io.BytesIO()
    fig_export, axes_e = plt.subplots(2, 3, figsize=(16, 9))
    fig_export.patch.set_facecolor("#0f172a")
    for ax in axes_e.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.grid(color="#334155", alpha=0.5, linewidth=0.5)
    # Re-draw for export
    for ax_src, ax_dst in zip(axes.flat, axes_e.flat):
        ax_dst.set_title(ax_src.get_title(), color="#e2e8f0", fontweight="bold", fontsize=9)
    fig_export.tight_layout()
    fig_export.savefig(chart_buf, format="png", dpi=150, bbox_inches="tight",
                       facecolor="#0f172a")
    plt.close(fig_export)
    st.session_state["chart_buf"] = chart_buf

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SECTOR DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

with tab_sectors:
    res = st.session_state.results
    ms  = st.session_state.model_state
    rp  = st.session_state.run_params

    if res is None or ms is None:
        st.info("Run an analysis first.")
        st.stop()

    st.markdown('<div class="section-header">Sector-Level Output ($)</div>', unsafe_allow_html=True)

    output_rows = []
    for s in range(me.N):
        dv = res["vecs"]["direct"][s]
        iv = res["vecs"]["indirect"][s]
        nv = res["vecs"]["induced"][s]
        tv = dv + iv + nv
        output_rows.append({
            "Sector": me.SECTOR_LABELS[s],
            "Direct": dv, "Indirect": iv, "Induced": nv, "Total": tv
        })
    out_df = pd.DataFrame(output_rows)
    out_df_fmt = out_df.copy()
    for col in ["Direct","Indirect","Induced","Total"]:
        out_df_fmt[col] = out_df_fmt[col].apply(fmt_dollar)
    st.dataframe(out_df_fmt, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Sector-Level Employment (Jobs)</div>',
                unsafe_allow_html=True)

    job_rows = []
    for s in range(me.N):
        dj = res["direct"]["jobs_vec"][s]
        ij = res["indirect"]["jobs_vec"][s]
        nj = res["induced"]["jobs_vec"][s]
        job_rows.append({
            "Sector": me.SECTOR_LABELS[s],
            "Direct": dj, "Indirect": ij, "Induced": nj, "Total": dj+ij+nj
        })
    job_df = pd.DataFrame(job_rows)
    job_df_fmt = job_df.copy()
    for col in ["Direct","Indirect","Induced","Total"]:
        job_df_fmt[col] = job_df_fmt[col].apply(lambda v: f"{v:.2f}")
    st.dataframe(job_df_fmt, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Location Quotients & Regionalization</div>',
                unsafe_allow_html=True)

    lq_rows = []
    for s in range(me.N):
        lq  = float(ms["lq_york"][s])
        rpc = float(ms["rpc_york"][s])
        emp = float(ms["qcew"]["emp"][s])
        a_nat  = float(ms["A_nat15"][:,s].sum())
        a_york = float(ms["A_york"][:,s].sum())
        lq_rows.append({
            "Sector": me.SECTOR_LABELS[s],
            "LQ":     lq,
            "RPC":    rpc,
            "Employment": emp,
            "A_national": a_nat,
            "A_york":     a_york,
            "Status": "✓ Local" if lq >= 1.0 else f"↓ {(1-rpc):.0%} imported",
        })
    lq_df = pd.DataFrame(lq_rows)
    lq_df_fmt = lq_df.copy()
    lq_df_fmt["LQ"]  = lq_df_fmt["LQ"].apply(lambda v: f"{v:.3f}")
    lq_df_fmt["RPC"] = lq_df_fmt["RPC"].apply(lambda v: f"{v:.3f}")
    lq_df_fmt["Employment"] = lq_df_fmt["Employment"].apply(lambda v: f"{v:,.0f}")
    lq_df_fmt["A_national"] = lq_df_fmt["A_national"].apply(lambda v: f"{v:.4f}")
    lq_df_fmt["A_york"]     = lq_df_fmt["A_york"].apply(lambda v: f"{v:.4f}")
    st.dataframe(lq_df_fmt, hide_index=True, use_container_width=True)

    # LQ bar chart
    fig_lq, ax_lq = plt.subplots(figsize=(14, 4))
    fig_lq.patch.set_facecolor("#0f172a")
    ax_lq.set_facecolor("#1e293b")
    lq_vals   = [float(ms["lq_york"][s]) for s in range(me.N)]
    bar_colors = ["#22c55e" if v >= 1.0 else "#64748b" for v in lq_vals]
    ax_lq.bar(range(me.N), lq_vals, color=bar_colors, edgecolor="none", width=0.6)
    ax_lq.axhline(y=1.0, color="#f87171", linestyle="--", linewidth=1)
    ax_lq.set_xticks(range(me.N))
    ax_lq.set_xticklabels([l[:18] for l in me.SECTOR_LABELS], rotation=40,
                           ha="right", color="#94a3b8", fontsize=8)
    ax_lq.set_ylabel("Location Quotient", color="#94a3b8", fontsize=9)
    ax_lq.set_title("York County Location Quotients  (green = fully local, LQ ≥ 1.0)",
                     color="#e2e8f0", fontsize=10, fontweight="bold")
    for sp in ax_lq.spines.values(): sp.set_visible(False)
    ax_lq.tick_params(colors="#94a3b8")
    ax_lq.grid(axis="y", color="#334155", alpha=0.5)
    fig_lq.tight_layout()
    st.pyplot(fig_lq, use_container_width=True)
    plt.close(fig_lq)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — A MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

with tab_matrix:
    ms = st.session_state.model_state
    if ms is None:
        st.info("Build the model first.")
        st.stop()

    st.markdown('<div class="section-header">15×15 Direct Requirements Matrix</div>',
                unsafe_allow_html=True)

    view_choice = st.radio("View", ["York County (regionalized)", "National (A_domestic)"],
                           horizontal=True)
    A_show = ms["A_york"] if "York County" in view_choice else ms["A_nat15"]

    A_df = pd.DataFrame(A_show, index=me.SECTOR_LABELS, columns=me.SECTOR_LABELS)
    st.dataframe(A_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=None),
                 use_container_width=True)

    st.caption("Rows = purchasing sector.  Columns = supplying sector.  "
               "Values = $ of input required per $ of output (domestic only).")

    # Heatmap
    fig_heat, ax_heat = plt.subplots(figsize=(12, 9))
    fig_heat.patch.set_facecolor("#0f172a")
    ax_heat.set_facecolor("#0f172a")
    import matplotlib.colors as mcolors
    cmap = plt.cm.YlOrRd
    im = ax_heat.imshow(A_show, cmap=cmap, aspect="auto", vmin=0)
    ax_heat.set_xticks(range(me.N))
    ax_heat.set_yticks(range(me.N))
    ax_heat.set_xticklabels([f"{s}" for s in range(me.N)],
                              color="#94a3b8", fontsize=8)
    ax_heat.set_yticklabels([l[:25] for l in me.SECTOR_LABELS],
                              color="#94a3b8", fontsize=8)
    fig_heat.colorbar(im, ax=ax_heat, label="Direct requirement coefficient")
    ax_heat.set_title(f"A Matrix Heatmap — {'York County' if 'York' in view_choice else 'National'}",
                       color="#e2e8f0", fontsize=11, fontweight="bold")
    fig_heat.tight_layout()
    st.pyplot(fig_heat, use_container_width=True)
    plt.close(fig_heat)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_coeffs:
    ms = st.session_state.model_state
    if ms is None:
        st.info("Build the model first.")
        st.stop()

    st.markdown('<div class="section-header">Employment & Income Coefficients</div>',
                unsafe_allow_html=True)

    # BEA data source info
    bea_src = ms.get("bea_output_src","unknown")
    src_label = "✅ Live BEA API" if bea_src == "live" else "⚠️ 2022 Fallback (API partial/unavailable)"
    st.markdown(f"""
    <div style='display:flex; gap:1rem; margin-bottom:1rem; align-items:center'>
      <span class='source-chip'>BEA Output: {src_label}</span>
      <span class='source-chip'>MPC: {ms['nat_mpc']:.4f}</span>
      <span class='source-chip'>HH Share: {ms['emp_coeffs']['hh_share']:.4f}</span>
      <span class='source-chip'>Regional Retention: {ms['emp_coeffs']['regional_retention']:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    val_results = ms["validation"]
    any_bad = any(not v["ok"] for v in val_results)

    if any_bad:
        st.warning("⚠️ One or more coefficients are outside plausible ranges. Check BEA API data quality.")
    else:
        st.success("✅ All employment coefficients are within plausible ranges (IMPLAN/RIMS II benchmarks).")

    coeff_rows = []
    for i, v in enumerate(val_results):
        coeff_rows.append({
            "Sector":         v["sector"],
            "Nat'l Empl (k)": f"{ms['national_emp'][i]:,.1f}",
            "Jobs / $1M":     f"{v['jpm']:.2f}",
            "York Avg Wage":  fmt_dollar(v["wage"]),
            "LI Share":       f"{v['li']:.1%}",
            "Status":         "✅ OK" if v["ok"] else "⚠️ " + "; ".join(v["flags"]),
        })
    coeff_df = pd.DataFrame(coeff_rows)
    st.dataframe(coeff_df, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">QCEW Employment Data (York County)</div>',
                unsafe_allow_html=True)
    qcew_rows = []
    for s in range(me.N):
        qcew_rows.append({
            "Sector":        me.SECTOR_LABELS[s],
            "Employment":    f"{ms['qcew']['emp'][s]:,.0f}",
            "LQ":            f"{ms['qcew']['lq'][s]:.3f}",
            "Avg Annual Pay":fmt_dollar(ms['qcew']['wage'][s]),
        })
    st.dataframe(pd.DataFrame(qcew_rows), hide_index=True, use_container_width=True)
    st.caption(f"QCEW data year: {ms['qcew']['year']}  |  FIPS: {ms['qcew']['fips']}  "
               f"|  Total employment: {ms['qcew']['emp'].sum():,.0f}")

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Unit Reference</div>',
                unsafe_allow_html=True)
    st.code("""
jobs_per_million = (national_emp_thousands * 1000) / bea_output_$M
                 → workers / $M  =  jobs per $1M gross output

nat_avg_wage     = (bea_comp_$M * 1e6) / (national_emp_thousands * 1000)
                 → $/worker/year

york_li_share    = (york_wage * national_emp * 1000) / (bea_output * 1e6)
                 → dimensionless  (fraction of gross output paid as labor income)

LI% cap: 0.70  (national comp share never meaningfully exceeds 0.65 for any sector)
    """, language="text")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_export:
    res = st.session_state.results
    ms  = st.session_state.model_state
    rp  = st.session_state.run_params

    if res is None or ms is None:
        st.info("Run an analysis first to enable exports.")
        st.stop()

    st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        # Impact summary CSV
        summary_df = pd.DataFrame([
            {"Impact Type": k.capitalize(),
             "Output ($)":       round(res[k]["output"],2),
             "Jobs":             round(res[k]["jobs"],2),
             "Labor Income ($)": round(res[k]["labor_income"],2),
             "Value Added ($)":  round(res[k]["value_added"],2)}
            for k in ["direct","indirect","induced","total"]
        ])
        st.download_button(
            "📥 Impact Summary CSV",
            data=summary_df.to_csv(index=False).encode(),
            file_name="impact_summary_v4.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Sector detail CSV
        sec_rows = []
        for s in range(me.N):
            sec_rows.append({
                "Sector":          me.SECTOR_LABELS[s],
                "QCEW_LQ":         round(float(ms["lq_york"][s]),4),
                "RPC":             round(float(ms["rpc_york"][s]),4),
                "York_Empl":       round(float(ms["qcew"]["emp"][s]),0),
                "York_AvgWage":    round(float(ms["qcew"]["wage"][s]),0),
                "Jobs_per_1M":     round(float(ms["emp_coeffs"]["jobs_per_million"][s]),4),
                "LI_Share":        round(float(ms["emp_coeffs"]["li_share"][s]),4),
                "A_national_colsum": round(float(ms["A_nat15"][:,s].sum()),4),
                "A_york_colsum":   round(float(ms["A_york"][:,s].sum()),4),
                "Direct_Output":   round(float(res["vecs"]["direct"][s]),2),
                "Indirect_Output": round(float(res["vecs"]["indirect"][s]),2),
                "Induced_Output":  round(float(res["vecs"]["induced"][s]),2),
                "Total_Output":    round(float(res["vecs"]["total"][s]),2),
                "Direct_Jobs":     round(float(res["direct"]["jobs_vec"][s]),2),
                "Indirect_Jobs":   round(float(res["indirect"]["jobs_vec"][s]),2),
                "Induced_Jobs":    round(float(res["induced"]["jobs_vec"][s]),2),
                "Total_Jobs":      round(float(res["direct"]["jobs_vec"][s]
                                               +res["indirect"]["jobs_vec"][s]
                                               +res["induced"]["jobs_vec"][s]),2),
            })
        sec_df = pd.DataFrame(sec_rows)
        st.download_button(
            "📥 Sector Detail CSV",
            data=sec_df.to_csv(index=False).encode(),
            file_name="impact_by_sector_v4.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_e2:
        # Multipliers CSV
        m = res["multipliers"]
        mult_df = pd.DataFrame([{
            "Project":               rp["project_name"],
            "NAICS":                 rp["naics"],
            "Region":                f"York County VA FIPS {ms['qcew']['fips']}",
            "Investment($)":         rp["investment"],
            "IO_Data":               "BEA 2017 Benchmark After-Redefinitions Domestic",
            "Employment_Data":       f"BLS QCEW {ms['qcew']['year']} + BLS CES 2022",
            "Import_Method":         "Two-stage: BEA domestic B + QCEW LQ/RPC",
            "LI_Cap":                0.70,
            "Induced_Dampening":     rp["dampening"],
            "Type_I_Multiplier":     round(m["type1"],4),
            "Type_II_Multiplier":    round(m["type2"],4),
            "Employment_Multiplier": round(m["emp"],4),
            "LaborIncome_Multiplier":round(m["li"],4),
            "ValueAdded_Multiplier": round(m["va"],4),
            "HH_Consumption_Share":  round(float(ms["emp_coeffs"]["hh_share"]),4),
        }])
        st.download_button(
            "📥 Multipliers CSV",
            data=mult_df.to_csv(index=False).encode(),
            file_name="multipliers_v4.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # A matrices CSV
        A_york_df = pd.DataFrame(ms["A_york"], index=me.SECTOR_LABELS, columns=me.SECTOR_LABELS)
        st.download_button(
            "📥 A_york Matrix CSV",
            data=A_york_df.to_csv().encode(),
            file_name="A_york_15sector.csv",
            mime="text/csv",
            use_container_width=True,
        )
        A_nat_df = pd.DataFrame(ms["A_nat15"], index=me.SECTOR_LABELS, columns=me.SECTOR_LABELS)
        st.download_button(
            "📥 A_domestic Matrix CSV",
            data=A_nat_df.to_csv().encode(),
            file_name="A_domestic_national_15sector.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Methodology Notes</div>', unsafe_allow_html=True)
    st.markdown(f"""
    **Model:** Regional Leontief Input-Output, Type I + Type II  
    **I-O Framework:** BEA 2017 Benchmark, After-Redefinitions, 402 industries → 15 sectors  
    **Import Correction:** Two-stage — BEA domestic direct requirements (B_domestic) × QCEW LQ regionalization  
    **Employment Coefficients:** BLS CES national employment / BEA gross output (jobs per $1M)  
    **Wages:** BLS QCEW 2024 county-level average annual pay  
    **Induced Effects:** Type II Leontief with BEA NIPA MPC = {ms['nat_mpc']:.4f}, regional retention = {ms['emp_coeffs']['regional_retention']:.2f}, dampening = {rp['dampening']:.2f}  
    **LI% Cap:** 0.70 (prevents API fallback artefacts from inflating induced loop)  
    **Plausibility Floor:** BEA gross output API sum must exceed $10T; compensation must exceed $3T  
    **Data currency:** I-O structure 2017; employment wages 2024; BEA output/comp 2022  

    *Comparable to IMPLAN/RIMS II in methodology. Results typically 20–40% below IMPLAN due to:  
    15-sector aggregation, simple SLQ vs. gravity-model trade flows, absence of SAM institutional accounts, and 2017 I-O vintage.*
    """)
