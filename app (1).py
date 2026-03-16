"""
Regional Economic Impact Analysis Tool  v5.0
Streamlit Application

Run:  streamlit run app.py
"""

import io
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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.app-header {
    background: #0f172a; color: #e2e8f0;
    padding: 1.4rem 2rem; border-radius: 8px;
    margin-bottom: 1.5rem; border-left: 4px solid #3b82f6;
}
.app-header h1 { margin:0; font-size:1.5rem; font-weight:700; color:#f8fafc; letter-spacing:-0.02em; }
.app-header p  { margin:0.2rem 0 0; font-size:0.82rem; color:#94a3b8; font-family:'IBM Plex Mono',monospace; }

.metric-card { background:#1e293b; border:1px solid #334155; border-radius:8px; padding:1.1rem 1.3rem; text-align:center; }
.metric-card .label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; margin-bottom:0.35rem; }
.metric-card .value { font-size:1.7rem; font-weight:700; color:#f1f5f9; font-family:'IBM Plex Mono',monospace; }
.metric-card .sub   { font-size:0.75rem; color:#94a3b8; margin-top:0.2rem; }

.section-header {
    font-size:0.72rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.1em; color:#64748b;
    border-bottom:1px solid #1e293b; padding-bottom:0.4rem; margin-bottom:1rem;
}
.source-chip {
    background:#0f172a; border:1px solid #334155; color:#94a3b8;
    font-size:0.68rem; padding:2px 8px; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; display:inline-block;
}
.warn-box  { background:#7f1d1d; border:1px solid #dc2626; border-radius:6px; padding:0.7rem 1rem; color:#fca5a5; font-size:0.82rem; margin:0.5rem 0; }
.ok-box    { background:#14532d; border:1px solid #16a34a; border-radius:6px; padding:0.7rem 1rem; color:#86efac; font-size:0.82rem; margin:0.5rem 0; }
.info-box  { background:#1e3a5f; border:1px solid #2563eb; border-radius:6px; padding:0.7rem 1rem; color:#93c5fd; font-size:0.82rem; margin:0.5rem 0; }

#MainMenu {visibility:hidden;} footer {visibility:hidden;}
section[data-testid="stSidebar"] { background:#0f172a !important; }
section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────

for key in ["model_state","results","run_params"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── HELPERS ───────────────────────────────────────────────────────────────────

def fmt_dollar(v):
    if abs(v) >= 1e9: return f"${v/1e9:,.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:,.1f}M"
    if abs(v) >= 1e3: return f"${v/1e3:,.0f}K"
    return f"${v:,.0f}"

def metric_card(label, value, sub=""):
    return (f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            + (f'<div class="sub">{sub}</div>' if sub else "")
            + '</div>')

COLORS = {
    "direct":   "#3b82f6",
    "indirect": "#22c55e",
    "induced":  "#f59e0b",
    "total":    "#a855f7",
}

def _dark_ax(ax):
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(color="#334155", alpha=0.5, linewidth=0.5)

# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
  <h1>📊 Regional Economic Impact Analysis</h1>
  <p>v5.0 · FLQ Regionalization · Jobs/VA Coefficients · Leontief Stability · BEA 2017 Benchmark · BLS QCEW</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**API Keys**")
    bea_key = st.text_input("BEA API Key", value="", type="password", key="bea_key",
                             placeholder="Enter BEA API key")
    bls_key = st.text_input("BLS API Key", value="", type="password", key="bls_key",
                             placeholder="Enter BLS API key")

    st.markdown("---")
    st.markdown("**Model Parameters**")

    flq_delta = st.slider(
        "FLQ Delta (δ)",
        min_value=0.10, max_value=0.35, value=0.25, step=0.05,
        help="Flegg LQ sensitivity. Higher = more import leakage assumed for small regions.\n"
             "Flegg & Webber (2000) recommend 0.25–0.30 for most US counties.\n"
             "0.10 approaches Simple LQ; 0.35 maximizes small-region correction."
    )
    mpc_param = st.slider(
        "MPC (Marginal Propensity to Consume)",
        min_value=0.70, max_value=0.98, value=0.90, step=0.01,
        help="Share of each additional dollar of income that households spend.\n"
             "BEA NIPA 2022 national average ≈ 0.96. Default 0.90 is more conservative."
    )
    rr_param = st.slider(
        "Regional Retention Rate",
        min_value=0.30, max_value=0.70, value=0.50, step=0.05,
        help="Share of household consumption spending that stays in the county.\n"
             "0.30–0.40 = small/rural  |  0.45–0.55 = suburban  |  0.60–0.70 = large metro.\n"
             "York County: 0.45–0.50 recommended (embedded in Hampton Roads MSA)."
    )

    st.markdown(f"""
    <div class='info-box'>
    <b>Induced HH share</b>: MPC × retention<br>
    = {mpc_param:.2f} × {rr_param:.2f} = <b>{mpc_param*rr_param:.4f}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📂 Data Files**")
    st.caption("BEA Use Table")
    use_file  = st.file_uploader("Use Table",   type=["xlsx"], label_visibility="collapsed", key="uf")
    st.caption("Market Share Matrix")
    ms_file   = st.file_uploader("MS Matrix",   type=["xlsx"], label_visibility="collapsed", key="msf")
    st.caption("Domestic Requirements")
    dom_file  = st.file_uploader("Dom Req",     type=["xlsx"], label_visibility="collapsed", key="df")
    st.caption("QCEW County CSV")
    qcew_file = st.file_uploader("QCEW CSV",    type=["csv"],  label_visibility="collapsed", key="qf")

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
                    use_file=use_file, ms_file=ms_file,
                    dom_file=dom_file, qcew_file=qcew_file,
                    bea_api_key=bea_key or "", bls_api_key=bls_key or "",
                    flq_delta=flq_delta, mpc=mpc_param,
                    regional_retention=rr_param,
                    progress_callback=_cb,
                )
                st.session_state.model_state = state
                st.session_state.results = None
                progress_bar.progress(100)
                status_text.caption("✅ Model ready.")

                # Show stability result immediately
                stab = state["stability"]
                if stab["stable"]:
                    st.success(f"✅ Model stable — spectral radius = {stab['spectral_radius']:.4f}")
                else:
                    st.error(f"⚠️ Stability warning: spectral radius = {stab['spectral_radius']:.4f}")
                    for w in stab["warnings"]:
                        st.warning(w)
            except Exception as e:
                st.error(f"Build failed: {e}")
                import traceback; st.code(traceback.format_exc())

    if not all_uploaded:
        st.caption("Upload all 4 files to enable build.")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab_run, tab_results, tab_sectors, tab_matrix, tab_coeffs, tab_diag, tab_export = st.tabs([
    "▶  Run Analysis",
    "📈  Results",
    "🏭  Sector Detail",
    "🔢  A Matrix",
    "🔬  Coefficients",
    "🔍  Diagnostics",
    "💾  Export",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ═══════════════════════════════════════════════════════════════════════════════

with tab_run:
    ms = st.session_state.model_state
    if ms is None:
        st.info("Upload all 4 data files and click **Build Model** in the sidebar.")
        st.markdown("""
        | File | Source |
        |------|--------|
        | `IOUse_After_Redefinitions_PRO_Detail.xlsx` | [BEA I-O Accounts](https://www.bea.gov/industry/input-output-accounts-data) |
        | `IxC_MS_Detail.xlsx` | [BEA I-O Accounts](https://www.bea.gov/industry/input-output-accounts-data) |
        | `CxI_Domestic_DR_Detail.xlsx` | [BEA I-O Accounts](https://www.bea.gov/industry/input-output-accounts-data) |
        | `51199.csv` | [BLS QCEW](https://www.bls.gov/cew/downloadable-data.htm) |
        """)
        st.stop()

    st.markdown('<div class="section-header">Project Parameters</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        project_name = st.text_input("Project Name", value="New Investment Project")
        naics_input  = st.text_input("Primary NAICS Code (2–4 digit)", value="23",
                                      placeholder="e.g. 23, 336, 7211, 42")
        naics_valid = False; primary_s = None
        try:
            primary_s   = me.naics_to_sector(naics_input)
            naics_valid = True
            st.success(f"→ Sector [{primary_s}]: **{me.SECTOR_LABELS[primary_s]}**")
        except ValueError:
            if naics_input:
                st.error(f"NAICS '{naics_input}' not recognized.")

    with col2:
        investment_raw = st.number_input(
            "Total Investment ($)", min_value=100_000, max_value=10_000_000_000,
            value=10_000_000, step=500_000, format="%d")
        st.markdown(f"""
        <div class='info-box' style='margin-top:0.5rem'>
          <b>Active parameters:</b><br>
          FLQ δ={ms['flq_delta']:.2f} · λ={ms['flq_lambda']:.4f}<br>
          MPC={ms['nat_mpc']:.3f} · Retention={ms['regional_retention']:.2f}<br>
          HH share = {ms['emp_coeffs']['hh_share']:.4f}
        </div>
        """, unsafe_allow_html=True)

    if naics_valid and primary_s is not None:
        st.markdown('<div class="section-header" style="margin-top:1.5rem">BEA Spending Profile</div>',
                    unsafe_allow_html=True)
        profile = ms["all_profiles"][primary_s]
        profile_sorted = sorted(profile.items(), key=lambda x: -x[1])

        col_a, col_b = st.columns([1, 1.5])
        with col_a:
            prof_df = pd.DataFrame([
                {"Sector": f"[{s:2d}] {me.SECTOR_LABELS[s]}", "Share": v}
                for s, v in profile_sorted
            ])
            st.dataframe(prof_df.style.format({"Share": "{:.1%}"})
                               .bar(subset=["Share"], color="#3b82f6"),
                         hide_index=True, use_container_width=True, height=320)
        with col_b:
            fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
            fig_p.patch.set_facecolor("#0f172a"); _dark_ax(ax_p)
            lbls = [me.SECTOR_LABELS[s][:22] for s,_ in profile_sorted[:10]]
            vals = [v for _,v in profile_sorted[:10]]
            ax_p.barh(lbls[::-1], vals[::-1], color="#3b82f6", edgecolor="none")
            ax_p.set_xlabel("Share", color="#94a3b8", fontsize=9)
            ax_p.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            fig_p.tight_layout()
            st.pyplot(fig_p, use_container_width=True); plt.close(fig_p)

    st.markdown("---")
    if st.button("▶  Run Impact Analysis",
                 disabled=not (naics_valid and project_name.strip()),
                 type="primary"):
        with st.spinner("Running Leontief model…"):
            profile = ms["all_profiles"][primary_s]
            Y = np.zeros(me.N)
            ts = sum(profile.values())
            for s, sh in profile.items():
                Y[s] += investment_raw * sh / ts

            results = me.compute_impacts(Y, ms["inverses"], ms["emp_coeffs"])
            st.session_state.results = results
            st.session_state.run_params = {
                "project_name": project_name,
                "naics":        naics_input,
                "primary_s":    primary_s,
                "investment":   investment_raw,
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
        st.info("Run an analysis first."); st.stop()

    t = res["total"]; d = res["direct"]; ii = res["indirect"]; ind = res["induced"]
    m = res["multipliers"]

    st.markdown(f"""
    <div style='margin-bottom:1rem'>
      <span style='font-size:1.1rem;font-weight:700;color:#f1f5f9'>{rp['project_name']}</span>
      &nbsp;&nbsp;
      <span class='source-chip'>NAICS {rp['naics']} · {me.SECTOR_LABELS[rp['primary_s']]}</span>
      &nbsp;<span class='source-chip'>{fmt_dollar(rp['investment'])}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Multiplier sanity check alerts ───────────────────────────────────────
    for chk in res["mult_checks"]:
        if not chk["ok"]:
            st.markdown(f"""<div class='warn-box'>
            ⚠️ <b>{chk['metric']}</b> = {chk['value']:.4f}×
            is outside expected range [{chk['low']:.1f}–{chk['high']:.1f}].
            {chk['note']}
            </div>""", unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Total Economic Impact</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    for col, lbl, val, sub in [
        (k1,"Total Output",     fmt_dollar(t["output"]),      f"{t['output']/rp['investment']:.2f}× ROI"),
        (k2,"Total Jobs",       f"{t['jobs']:,.0f}",           f"Mult {m['emp']:.2f}×"),
        (k3,"GDP Contribution", fmt_dollar(t["value_added"]), f"{t['value_added']/rp['investment']:.1%} of invest."),
        (k4,"Labor Income",     fmt_dollar(t["labor_income"]),f"{t['labor_income']/t['output']:.1%} of output"),
        (k5,"Type II Mult.",    f"{m['type2']:.3f}×",         f"Type I: {m['type1']:.3f}×"),
    ]:
        col.markdown(metric_card(lbl, val, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Impact table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Impact by Type</div>', unsafe_allow_html=True)
    impact_df = pd.DataFrame([
        {"Type":"Direct",  "Output":d["output"],  "Jobs":d["jobs"],  "Labor Income":d["labor_income"],  "Value Added":d["value_added"]},
        {"Type":"Indirect","Output":ii["output"], "Jobs":ii["jobs"], "Labor Income":ii["labor_income"], "Value Added":ii["value_added"]},
        {"Type":"Induced", "Output":ind["output"],"Jobs":ind["jobs"],"Labor Income":ind["labor_income"],"Value Added":ind["value_added"]},
        {"Type":"TOTAL",   "Output":t["output"],  "Jobs":t["jobs"],  "Labor Income":t["labor_income"],  "Value Added":t["value_added"]},
    ])
    def _fmt_row(row):
        return pd.Series({"Type": row["Type"], "Output": fmt_dollar(row["Output"]),
                           "Jobs": f"{row['Jobs']:,.1f}",
                           "Labor Income": fmt_dollar(row["Labor Income"]),
                           "Value Added":  fmt_dollar(row["Value Added"])})
    st.dataframe(impact_df.apply(_fmt_row,axis=1), hide_index=True, use_container_width=True)

    # ── Stacked output chart (improvement #8) ────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:1.5rem">Visualizations</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes.flat: _dark_ax(ax)

    # 1 — Stacked output bar (direct/indirect/induced)
    ax = axes[0,0]
    impact_types = ["Direct","Indirect","Induced"]
    d_out   = d["output"]/1e6
    ii_out  = ii["output"]/1e6
    ind_out = ind["output"]/1e6
    x_pos   = [0]
    ax.bar(x_pos, [d_out],   color=COLORS["direct"],   label="Direct",   width=0.5)
    ax.bar(x_pos, [ii_out],  color=COLORS["indirect"], label="Indirect", width=0.5, bottom=[d_out])
    ax.bar(x_pos, [ind_out], color=COLORS["induced"],  label="Induced",  width=0.5, bottom=[d_out+ii_out])
    ax.set_title("Output Stacked ($M)", color="#e2e8f0", fontweight="bold", fontsize=9)
    ax.set_ylabel("$M", color="#94a3b8", fontsize=8)
    ax.set_xticks([0]); ax.set_xticklabels(["Total"], color="#94a3b8")
    ax.legend(fontsize=7, facecolor="#1e293b", labelcolor="#e2e8f0")
    ax.text(0, d_out/2, f"${d_out:.1f}M", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(0, d_out+ii_out/2, f"${ii_out:.1f}M", ha="center", va="center", color="white", fontsize=8)
    ax.text(0, d_out+ii_out+ind_out/2, f"${ind_out:.1f}M", ha="center", va="center", color="white", fontsize=8)

    # 2 — Jobs by type (grouped)
    ax = axes[0,1]
    job_vals = [d["jobs"], ii["jobs"], ind["jobs"], t["jobs"]]
    clrs = [COLORS["direct"],COLORS["indirect"],COLORS["induced"],COLORS["total"]]
    b2 = ax.bar(["Direct","Indirect","Induced","Total"], job_vals, color=clrs, width=0.5)
    ax.set_title("Jobs Created", color="#e2e8f0", fontweight="bold", fontsize=9)
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                f"{b.get_height():.0f}", ha="center", va="bottom", color="#e2e8f0", fontsize=8)

    # 3 — Multipliers
    ax = axes[0,2]
    mult_l = ["Type I\nOutput","Type II\nOutput","Employment","Labor\nIncome","Value\nAdded"]
    mult_v = [m["type1"], m["type2"], m["emp"], m["li"], m["va"]]
    clrs_m = [
        COLORS["indirect"] if (me.MULT_TYPE1_MIN <= m["type1"] <= me.MULT_TYPE1_MAX) else "#ef4444",
        COLORS["total"]    if (me.MULT_TYPE2_MIN <= m["type2"] <= me.MULT_TYPE2_MAX) else "#ef4444",
        "#64748b","#64748b","#64748b"
    ]
    b3 = ax.bar(mult_l, mult_v, color=clrs_m, width=0.5)
    ax.axhline(y=1.0, color="#f87171", linestyle="--", linewidth=1)
    ax.set_title("Economic Multipliers", color="#e2e8f0", fontweight="bold", fontsize=9)
    for b in b3:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{b.get_height():.3f}×", ha="center", va="bottom", color="#e2e8f0", fontsize=8)

    # 4 — Output composition pie
    ax = axes[1,0]
    ax.pie([d["output"], ii["output"], ind["output"]],
           labels=["Direct","Indirect","Induced"],
           colors=[COLORS["direct"],COLORS["indirect"],COLORS["induced"]],
           autopct="%1.1f%%", startangle=90,
           textprops={"color":"#e2e8f0","fontsize":8},
           wedgeprops={"edgecolor":"#0f172a","linewidth":1.5})
    ax.set_title("Output Composition", color="#e2e8f0", fontweight="bold", fontsize=9)

    # 5 — Import leakage rate + HH retention (improvement #8)
    ax = axes[1,1]
    flq_rpc = ms["rpc_york"]
    slq_rpc = ms["rpc_slq"]
    x5 = np.arange(me.N)
    w5 = 0.35
    ax.bar(x5-w5/2, 1-slq_rpc, w5, label="SLQ leakage", color="#64748b", alpha=0.7)
    ax.bar(x5+w5/2, 1-flq_rpc, w5, label="FLQ leakage", color="#ef4444", alpha=0.9)
    ax.set_title("Import Leakage Rate by Sector\n(red=FLQ, grey=SLQ for comparison)",
                 color="#e2e8f0", fontweight="bold", fontsize=8)
    ax.set_ylabel("Leakage %", color="#94a3b8", fontsize=8)
    ax.set_xticks(x5); ax.set_xticklabels([str(s) for s in range(me.N)], color="#94a3b8")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=7, facecolor="#1e293b", labelcolor="#e2e8f0")

    # 6 — HH spending retention rate visualization (improvement #8)
    ax = axes[1,2]
    mpc_v = ms["nat_mpc"]; rr_v = ms["regional_retention"]
    hh_v  = ms["emp_coeffs"]["hh_share"]
    ax.barh(["MPC","× Retention","= HH Share"], [mpc_v, rr_v, hh_v],
            color=[COLORS["direct"], COLORS["indirect"], COLORS["total"]], edgecolor="none")
    ax.set_xlim(0, 1.1)
    ax.set_title("Household Spending Retention",
                 color="#e2e8f0", fontweight="bold", fontsize=9)
    ax.set_xlabel("Rate", color="#94a3b8", fontsize=8)
    for i, v in enumerate([mpc_v, rr_v, hh_v]):
        ax.text(v+0.02, i, f"{v:.3f}", va="center", color="#e2e8f0", fontsize=9, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.suptitle(f"{rp['project_name']}  ·  {fmt_dollar(rp['investment'])}",
                 color="#f1f5f9", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SECTOR DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

with tab_sectors:
    res = st.session_state.results; ms = st.session_state.model_state
    rp  = st.session_state.run_params
    if res is None or ms is None:
        st.info("Run an analysis first."); st.stop()

    st.markdown('<div class="section-header">Sector Output ($)</div>', unsafe_allow_html=True)
    out_rows = []
    for s in range(me.N):
        dv=res["vecs"]["direct"][s]; iv=res["vecs"]["indirect"][s]
        nv=res["vecs"]["induced"][s]; tv=dv+iv+nv
        out_rows.append({"Sector":me.SECTOR_LABELS[s],"Direct":dv,"Indirect":iv,"Induced":nv,"Total":tv})
    out_df = pd.DataFrame(out_rows)
    out_fmt = out_df.copy()
    for c in ["Direct","Indirect","Induced","Total"]:
        out_fmt[c] = out_fmt[c].apply(fmt_dollar)
    st.dataframe(out_fmt, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Sector Employment (Jobs)</div>',
                unsafe_allow_html=True)
    job_rows = []
    for s in range(me.N):
        dj=res["direct"]["jobs_vec"][s]; ij=res["indirect"]["jobs_vec"][s]
        nj=res["induced"]["jobs_vec"][s]
        job_rows.append({"Sector":me.SECTOR_LABELS[s],"Direct":dj,"Indirect":ij,"Induced":nj,"Total":dj+ij+nj})
    job_df = pd.DataFrame(job_rows)
    job_fmt = job_df.copy()
    for c in ["Direct","Indirect","Induced","Total"]:
        job_fmt[c] = job_fmt[c].apply(lambda v: f"{v:.2f}")
    st.dataframe(job_fmt, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Location Quotients & FLQ Regionalization</div>',
                unsafe_allow_html=True)
    lq_rows = []
    for s in range(me.N):
        lq=float(ms["lq_york"][s]); rpc_flq=float(ms["rpc_york"][s]); rpc_slq=float(ms["rpc_slq"][s])
        lq_rows.append({
            "Sector":        me.SECTOR_LABELS[s],
            "LQ":            f"{lq:.3f}",
            "RPC (FLQ)":     f"{rpc_flq:.4f}",
            "RPC (SLQ)":     f"{rpc_slq:.3f}",
            "FLQ Δ leakage": f"+{(rpc_slq-rpc_flq):.3f}",
            "Employment":    f"{ms['qcew']['emp'][s]:,.0f}",
            "A_york":        f"{ms['A_york'][:,s].sum():.4f}",
            "Status":        "✓ Local" if lq>=1.0 else f"↓{(1-rpc_flq):.0%} imported",
        })
    st.dataframe(pd.DataFrame(lq_rows), hide_index=True, use_container_width=True)
    st.caption(f"FLQ lambda = {ms['flq_lambda']:.4f}  (δ={ms['flq_delta']})  "
               f"— all LQs scaled by this factor before capping at 1.0. "
               f"Avg FLQ import leakage: {ms['import_leakage_rate']:.1%}  "
               f"vs SLQ: {ms['import_leakage_rate_slq']:.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — A MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

with tab_matrix:
    ms = st.session_state.model_state
    if ms is None:
        st.info("Build the model first."); st.stop()

    st.markdown('<div class="section-header">15×15 Direct Requirements Matrix</div>',
                unsafe_allow_html=True)
    view_choice = st.radio("View", ["York County (FLQ regionalized)", "National (A_domestic)"],
                           horizontal=True)
    A_show = ms["A_york"] if "York" in view_choice else ms["A_nat15"]

    A_df = pd.DataFrame(A_show, index=me.SECTOR_LABELS, columns=me.SECTOR_LABELS)
    st.dataframe(A_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=None),
                 use_container_width=True)

    fig_h, ax_h = plt.subplots(figsize=(12, 9))
    fig_h.patch.set_facecolor("#0f172a"); ax_h.set_facecolor("#0f172a")
    im = ax_h.imshow(A_show, cmap="YlOrRd", aspect="auto", vmin=0)
    ax_h.set_xticks(range(me.N)); ax_h.set_yticks(range(me.N))
    ax_h.set_xticklabels([f"{s}" for s in range(me.N)], color="#94a3b8", fontsize=8)
    ax_h.set_yticklabels([l[:25] for l in me.SECTOR_LABELS], color="#94a3b8", fontsize=8)
    fig_h.colorbar(im, ax=ax_h, label="Coefficient")
    ax_h.set_title(f"A Matrix Heatmap — {'York County (FLQ)' if 'York' in view_choice else 'National'}",
                   color="#e2e8f0", fontsize=11, fontweight="bold")
    fig_h.tight_layout()
    st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_coeffs:
    ms = st.session_state.model_state
    if ms is None:
        st.info("Build the model first."); st.stop()

    bea_src = ms.get("bea_va_src","unknown")
    src_lbl = {"live":"✅ Live BEA API","csv_cache":"📁 Local CSV Cache","hardcoded_fallback":"⚠️ 2022 Fallback"}.get(bea_src,"⚠️ Fallback")
    st.markdown(f"""
    <div style='display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:1rem'>
      <span class='source-chip'>VA Data: {src_lbl}</span>
      <span class='source-chip'>MPC: {ms['nat_mpc']:.4f}</span>
      <span class='source-chip'>Retention: {ms['regional_retention']:.2f}</span>
      <span class='source-chip'>HH share: {ms['emp_coeffs']['hh_share']:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

    val_results = ms["validation"]
    if any(not v["ok"] for v in val_results):
        st.warning("⚠️ One or more coefficients are outside plausible ranges.")
    else:
        st.success("✅ All employment coefficients are within plausible ranges.")

    coeff_rows = []
    for i, v in enumerate(val_results):
        coeff_rows.append({
            "Sector":           v["sector"],
            "Nat'l Empl (k)":  f"{ms['national_emp'][i]:,.1f}",
            "VA $M":            f"{ms['bea_va'][i]:,.0f}",
            "Jobs/$1M VA":      f"{v['jpm']:.2f}",
            "York Avg Wage":    fmt_dollar(v["wage"]),
            "LI Share":         f"{v['li']:.1%}",
            "Status":           "✅" if v["ok"] else "⚠️ " + "; ".join(v["flags"]),
        })
    st.dataframe(pd.DataFrame(coeff_rows), hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">QCEW Employment</div>',
                unsafe_allow_html=True)
    qr = []
    for s in range(me.N):
        qr.append({"Sector":me.SECTOR_LABELS[s],
                   "Employment":f"{ms['qcew']['emp'][s]:,.0f}",
                   "LQ":f"{ms['qcew']['lq'][s]:.3f}",
                   "FLQ RPC":f"{ms['rpc_york'][s]:.4f}",
                   "Avg Annual Pay":fmt_dollar(ms['qcew']['wage'][s])})
    st.dataframe(pd.DataFrame(qr), hide_index=True, use_container_width=True)
    st.caption(f"QCEW {ms['qcew']['year']}  |  FIPS {ms['qcew']['fips']}  |  Total: {ms['qcew']['emp'].sum():,.0f} workers")

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Unit Reference (v5)</div>',
                unsafe_allow_html=True)
    st.code("""
# Employment intensity — jobs per $1M VALUE ADDED (not gross output)
jobs_per_va = (national_emp_thousands * 1000) / bea_va_$M
            → workers / $M VA

# National average wage
nat_avg_wage = (bea_comp_$M * 1e6) / (national_emp_thousands * 1000)
             → $/worker/year

# York County labor income share (fraction of gross output paid as wages)
york_li_share = (york_wage * national_emp * 1000) / (bea_output_$M * 1e6)
              → dimensionless  [capped at 0.70]

# FLQ regionalization
lambda = log2(1 + county_emp / national_emp) ^ delta
FLQ_s  = lambda * LQ_s
RPC_s  = min(FLQ_s, 1.0)

# Household spending share
hh_share = MPC * regional_retention
    """, language="text")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_diag:
    ms  = st.session_state.model_state
    res = st.session_state.results
    if ms is None:
        st.info("Build the model first."); st.stop()

    st.markdown('<div class="section-header">Leontief Stability</div>', unsafe_allow_html=True)
    stab = ms["stability"]
    if stab["stable"]:
        st.markdown(f"""<div class='ok-box'>
        ✅ <b>Leontief system is stable</b><br>
        Spectral radius = <b>{stab['spectral_radius']:.6f}</b> &lt; 1.0<br>
        Max column sum = {stab['max_col_sum']:.4f}
        (All columns &lt; 1.0 means no sector spends more on inputs than it earns)
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='warn-box'>
        ⚠️ <b>Stability warning</b><br>
        Spectral radius = <b>{stab['spectral_radius']:.6f}</b> ≥ 1.0<br>
        The Leontief inverse may not converge. Check data quality.
        </div>""", unsafe_allow_html=True)
    for w in stab["warnings"]:
        st.warning(w)

    # Eigenvalue plot
    A_show_diag = ms["A_york"]
    eigvals = np.linalg.eigvals(A_show_diag)
    fig_e, ax_e = plt.subplots(figsize=(7, 7))
    fig_e.patch.set_facecolor("#0f172a"); ax_e.set_facecolor("#1e293b")
    theta = np.linspace(0, 2*np.pi, 200)
    ax_e.plot(np.cos(theta), np.sin(theta), color="#ef4444", linewidth=1.5,
              linestyle="--", label="Unit circle (stability boundary)")
    ax_e.scatter(eigvals.real, eigvals.imag, color="#3b82f6", s=40, zorder=5, label="Eigenvalues")
    ax_e.axhline(0, color="#475569", linewidth=0.5)
    ax_e.axvline(0, color="#475569", linewidth=0.5)
    ax_e.set_xlim(-1.5, 1.5); ax_e.set_ylim(-1.5, 1.5)
    ax_e.set_title(f"A_york Eigenvalues  (spectral radius={stab['spectral_radius']:.4f})",
                   color="#e2e8f0", fontweight="bold")
    ax_e.legend(fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0")
    for sp in ax_e.spines.values(): sp.set_visible(False)
    ax_e.tick_params(colors="#94a3b8")
    fig_e.tight_layout()
    st.pyplot(fig_e, use_container_width=True); plt.close(fig_e)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">FLQ vs SLQ Comparison</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | FLQ delta (δ) | {ms['flq_delta']} |
    | FLQ lambda (λ) | {ms['flq_lambda']:.4f} |
    | County employment | {ms['qcew']['total_county_emp']:,.0f} workers |
    | National employment | {ms['national_emp'].sum()*1000:,.0f} workers |
    | Avg RPC — SLQ | {ms['rpc_slq'].mean():.4f} |
    | Avg RPC — FLQ | {ms['rpc_york'].mean():.4f} |
    | Avg import leakage — SLQ | {ms['import_leakage_rate_slq']:.1%} |
    | Avg import leakage — FLQ | {ms['import_leakage_rate']:.1%} |
    """)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Multiplier Checks</div>',
                unsafe_allow_html=True)
    if res:
        for chk in res["mult_checks"]:
            status = "✅" if chk["ok"] else "⚠️"
            st.markdown(
                f"{status} **{chk['metric']}** = {chk['value']:.4f}× "
                f"(range {chk['low']:.1f}–{chk['high']:.1f}) — {chk['note']}"
            )
    else:
        st.info("Run an analysis to see multiplier checks.")

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Data Source Summary</div>',
                unsafe_allow_html=True)
    src_map = {"live":"✅ Live BEA API","csv_cache":"📁 Local CSV Cache","hardcoded_fallback":"⚠️ 2022 Constant"}
    st.markdown(f"""
    | Data | Source | Status |
    |---|---|---|
    | BEA Gross Output (Table 5) | GDPbyIndustry API | {src_map.get(ms['bea_output_src'],'?')} |
    | BEA Compensation (Table 2) | GDPbyIndustry API | {src_map.get(ms['bea_comp_src'],'?')} |
    | BEA Value Added (Table 1) | GDPbyIndustry API | {src_map.get(ms['bea_va_src'],'?')} |
    | MPC | BEA NIPA T20600 | {ms['nat_mpc']:.4f} |
    | National Employment | BLS CES 2022 | {ms['national_emp'].sum()*1000:,.0f} workers |
    | County Employment | BLS QCEW {ms['qcew']['year']} | {ms['qcew']['emp'].sum():,.0f} workers |
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_export:
    res = st.session_state.results
    ms  = st.session_state.model_state
    rp  = st.session_state.run_params
    if res is None or ms is None:
        st.info("Run an analysis first."); st.stop()

    st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        # Impact summary
        summ_df = pd.DataFrame([
            {"Impact Type":k.capitalize(),"Output ($)":round(res[k]["output"],2),
             "Jobs":round(res[k]["jobs"],2),"Labor Income ($)":round(res[k]["labor_income"],2),
             "Value Added ($)":round(res[k]["value_added"],2)}
            for k in ["direct","indirect","induced","total"]
        ])
        st.download_button("📥 Impact Summary CSV",
                           data=summ_df.to_csv(index=False).encode(),
                           file_name="impact_summary_v5.csv", mime="text/csv",
                           use_container_width=True)

        # Sector detail
        sec_rows = []
        for s in range(me.N):
            dj=res["direct"]["jobs_vec"][s]; ij=res["indirect"]["jobs_vec"][s]
            nj=res["induced"]["jobs_vec"][s]
            sec_rows.append({
                "Sector":          me.SECTOR_LABELS[s],
                "LQ":              round(float(ms["lq_york"][s]),4),
                "RPC_FLQ":         round(float(ms["rpc_york"][s]),4),
                "RPC_SLQ":         round(float(ms["rpc_slq"][s]),4),
                "Employment":      round(float(ms["qcew"]["emp"][s]),0),
                "AvgWage":         round(float(ms["qcew"]["wage"][s]),0),
                "Jobs_per_1M_VA":  round(float(ms["emp_coeffs"]["jobs_per_va"][s]),4),
                "LI_Share":        round(float(ms["emp_coeffs"]["li_share"][s]),4),
                "A_national":      round(float(ms["A_nat15"][:,s].sum()),4),
                "A_york_FLQ":      round(float(ms["A_york"][:,s].sum()),4),
                "Direct_Output":   round(float(res["vecs"]["direct"][s]),2),
                "Indirect_Output": round(float(res["vecs"]["indirect"][s]),2),
                "Induced_Output":  round(float(res["vecs"]["induced"][s]),2),
                "Total_Output":    round(float(res["vecs"]["total"][s]),2),
                "Direct_Jobs":     round(dj,2), "Indirect_Jobs":round(ij,2),
                "Induced_Jobs":    round(nj,2), "Total_Jobs":round(dj+ij+nj,2),
            })
        st.download_button("📥 Sector Detail CSV",
                           data=pd.DataFrame(sec_rows).to_csv(index=False).encode(),
                           file_name="impact_by_sector_v5.csv", mime="text/csv",
                           use_container_width=True)

    with col_e2:
        m = res["multipliers"]
        mult_df = pd.DataFrame([{
            "Project":              rp["project_name"],
            "NAICS":                rp["naics"],
            "Region":               f"FIPS {ms['qcew']['fips']}",
            "Investment($)":        rp["investment"],
            "IO_Data":              "BEA 2017 Benchmark After-Redefinitions Domestic",
            "Regionalization":      f"FLQ delta={ms['flq_delta']} lambda={ms['flq_lambda']:.4f}",
            "MPC":                  round(ms["nat_mpc"],4),
            "Regional_Retention":   ms["regional_retention"],
            "HH_Share":             round(ms["emp_coeffs"]["hh_share"],4),
            "LI_Cap":               0.70,
            "Spectral_Radius":      round(ms["stability"]["spectral_radius"],6),
            "Type_I_Multiplier":    round(m["type1"],4),
            "Type_II_Multiplier":   round(m["type2"],4),
            "Emp_Multiplier":       round(m["emp"],4),
            "LaborIncome_Mult":     round(m["li"],4),
            "ValueAdded_Mult":      round(m["va"],4),
            "Avg_Import_Leakage_FLQ": round(ms["import_leakage_rate"],4),
            "Avg_Import_Leakage_SLQ": round(ms["import_leakage_rate_slq"],4),
        }])
        st.download_button("📥 Multipliers CSV",
                           data=mult_df.to_csv(index=False).encode(),
                           file_name="multipliers_v5.csv", mime="text/csv",
                           use_container_width=True)

        A_york_df = pd.DataFrame(ms["A_york"], index=me.SECTOR_LABELS, columns=me.SECTOR_LABELS)
        st.download_button("📥 A_york Matrix CSV",
                           data=A_york_df.to_csv().encode(),
                           file_name="A_york_FLQ_15sector.csv", mime="text/csv",
                           use_container_width=True)
        A_nat_df = pd.DataFrame(ms["A_nat15"], index=me.SECTOR_LABELS, columns=me.SECTOR_LABELS)
        st.download_button("📥 A_domestic Matrix CSV",
                           data=A_nat_df.to_csv().encode(),
                           file_name="A_domestic_national_15sector.csv", mime="text/csv",
                           use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Methodology Notes</div>', unsafe_allow_html=True)
    st.markdown(f"""
**Model version:** v5.0  
**I-O Framework:** BEA 2017 Benchmark, After-Redefinitions, 402 industries → 15 sectors  
**Import Correction:** Two-stage — BEA domestic B_domestic × FLQ regionalization  
**Regionalization:** Flegg Location Quotient (Flegg & Webber 2000), δ={ms['flq_delta']}, λ={ms['flq_lambda']:.4f}  
**Employment Coefficients:** BLS CES national employment / BEA value added (Table 1)  
**Wages:** BLS QCEW {ms['qcew']['year']} county-level average annual pay  
**Induced Effects:** Type II Leontief — MPC={ms['nat_mpc']:.4f} × regional retention={ms['regional_retention']:.2f} → HH share={ms['emp_coeffs']['hh_share']:.4f}  
**Leontief stability:** Spectral radius = {ms['stability']['spectral_radius']:.6f} ({'STABLE' if ms['stability']['stable'] else 'UNSTABLE'})  
**LI% Cap:** 0.70  
**BEA data plausibility floors:** Output ≥ $10T, Compensation ≥ $3T, Value Added ≥ $5T  

*Results 20–40% below IMPLAN due to: 15-sector aggregation, FLQ vs. gravity-model trade flows, absence of SAM institutional accounts, 2017 I-O vintage. FLQ increases import leakage vs. SLQ, further reducing multipliers toward IMPLAN-comparable levels for small counties.*
    """)
