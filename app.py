"""
Regional Economic Impact Analysis Tool  v6.0
Streamlit Application

Run:  streamlit run app.py
"""
import io, warnings
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
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
.app-header{background:#0f172a;color:#e2e8f0;padding:1.4rem 2rem;border-radius:8px;
  margin-bottom:1.5rem;border-left:4px solid #3b82f6;}
.app-header h1{margin:0;font-size:1.5rem;font-weight:700;color:#f8fafc;letter-spacing:-0.02em;}
.app-header p{margin:0.2rem 0 0;font-size:0.82rem;color:#94a3b8;font-family:'IBM Plex Mono',monospace;}
.metric-card{background:#1e293b;border:1px solid #334155;border-radius:8px;
  padding:1.1rem 1.3rem;text-align:center;}
.metric-card .label{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;
  color:#64748b;margin-bottom:0.35rem;}
.metric-card .value{font-size:1.7rem;font-weight:700;color:#f1f5f9;
  font-family:'IBM Plex Mono',monospace;}
.metric-card .sub{font-size:0.75rem;color:#94a3b8;margin-top:0.2rem;}
.metric-card .band{font-size:0.70rem;color:#64748b;margin-top:0.15rem;
  font-family:'IBM Plex Mono',monospace;}
.section-header{font-size:0.72rem;font-weight:600;text-transform:uppercase;
  letter-spacing:0.1em;color:#64748b;border-bottom:1px solid #1e293b;
  padding-bottom:0.4rem;margin-bottom:1rem;}
.source-chip{background:#0f172a;border:1px solid #334155;color:#94a3b8;
  font-size:0.68rem;padding:2px 8px;border-radius:4px;
  font-family:'IBM Plex Mono',monospace;display:inline-block;}
.warn-box{background:#7f1d1d;border:1px solid #dc2626;border-radius:6px;
  padding:0.7rem 1rem;color:#fca5a5;font-size:0.82rem;margin:0.5rem 0;}
.ok-box{background:#14532d;border:1px solid #16a34a;border-radius:6px;
  padding:0.7rem 1rem;color:#86efac;font-size:0.82rem;margin:0.5rem 0;}
.info-box{background:#1e3a5f;border:1px solid #2563eb;border-radius:6px;
  padding:0.7rem 1rem;color:#93c5fd;font-size:0.82rem;margin:0.5rem 0;}
.unc-note{background:#1c1917;border:1px solid #57534e;border-radius:6px;
  padding:0.6rem 1rem;color:#a8a29e;font-size:0.75rem;margin:0.4rem 0;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
section[data-testid="stSidebar"]{background:#0f172a !important;}
section[data-testid="stSidebar"] *{color:#cbd5e1 !important;}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k in ["model_state","results","run_params","scenarios"]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "scenarios" else []

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fmt(v):
    if abs(v)>=1e9: return f"${v/1e9:,.2f}B"
    if abs(v)>=1e6: return f"${v/1e6:,.1f}M"
    if abs(v)>=1e3: return f"${v/1e3:,.0f}K"
    return f"${v:,.0f}"

def fmt_band(lo, hi):
    return f"[{fmt(lo)} – {fmt(hi)}]"

def fmt_jobs_band(lo, hi):
    return f"[{lo:,.0f} – {hi:,.0f}]"

def metric_card(label, value, sub="", band=""):
    return (f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            + (f'<div class="sub">{sub}</div>' if sub else "")
            + (f'<div class="band">±30%: {band}</div>' if band else "")
            + '</div>')

DARK = "#0f172a"
PANEL = "#1e293b"
CLRS = {"direct":"#3b82f6","indirect":"#22c55e","induced":"#f59e0b","total":"#a855f7"}

def dark_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#94a3b8",labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(color="#334155",alpha=0.5,linewidth=0.5)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>📊 Regional Economic Impact Analysis</h1>
  <p>v6.0 · 17-Sector Model · FLQ Regionalization · Uncertainty Bands · Sensitivity Analysis · Scenario Comparison</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    st.markdown("**API Keys**")
    bea_key = st.text_input("BEA API Key", value="", type="password", key="bea_key")
    bls_key = st.text_input("BLS API Key", value="", type="password", key="bls_key")
    st.markdown("---")
    st.markdown("**Model Parameters**")
    flq_delta = st.slider("FLQ Delta (δ)", 0.10, 0.35, 0.25, 0.05,
        help="Flegg & Webber (2000) sensitivity. 0.25 recommended for small counties.")
    mpc_param = st.slider("MPC", 0.70, 0.98, 0.90, 0.01,
        help="Marginal propensity to consume. BEA NIPA 2022 ≈ 0.96; default 0.90 is conservative.")
    rr_param  = st.slider("Regional Retention", 0.30, 0.70, 0.50, 0.05,
        help="Fraction of household spending that stays in the county.")
    st.markdown(f"""
    <div class='info-box'>HH share = {mpc_param:.2f} × {rr_param:.2f} = <b>{mpc_param*rr_param:.4f}</b></div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📂 Data Files**")
    st.caption("BEA Use Table (IOUse_After_Redefinitions…)")
    use_file  = st.file_uploader("Use Table",  type=["xlsx"],label_visibility="collapsed",key="uf")
    st.caption("Market Share Matrix (IxC_MS_Detail…)")
    ms_file   = st.file_uploader("MS Matrix",  type=["xlsx"],label_visibility="collapsed",key="msf")
    st.caption("Domestic Requirements (CxI_Domestic…)")
    dom_file  = st.file_uploader("Dom Req",    type=["xlsx"],label_visibility="collapsed",key="df")
    st.caption("QCEW County CSV (51199.csv)")
    qcew_file = st.file_uploader("QCEW CSV",   type=["csv"], label_visibility="collapsed",key="qf")

    all_up = all([use_file, ms_file, dom_file, qcew_file])
    st.markdown("---")
    if st.button("🔨 Build Model", disabled=not all_up,
                 use_container_width=True, type="primary"):
        pb = st.progress(0); st_txt = st.empty()
        def _cb(msg,pct): pb.progress(pct); st_txt.caption(msg)
        with st.spinner("Building…"):
            try:
                state = me.build_model(
                    use_file, ms_file, dom_file, qcew_file,
                    bea_key or "", bls_key or "",
                    flq_delta=flq_delta, mpc=mpc_param,
                    regional_retention=rr_param,
                    progress_callback=_cb)
                st.session_state.model_state = state
                st.session_state.results = None
                st.session_state.scenarios = []
                pb.progress(100); st_txt.caption("✅ Model ready.")
                stab = state["stability"]
                if stab["stable"]:
                    st.success(f"✅ Stable — spectral radius = {stab['spectral_radius']:.4f}")
                else:
                    st.error(f"⚠️ Unstable — spectral radius = {stab['spectral_radius']:.4f}")
            except Exception as e:
                st.error(f"Build failed: {e}")
                import traceback; st.code(traceback.format_exc())
    if not all_up:
        st.caption("Upload all 4 files to enable build.")

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["▶  Run","📈  Results","🔀  Scenarios",
                "📉  Sensitivity","🏭  Sectors","🔢  Matrix",
                "🔬  Coefficients","🔍  Diagnostics","💾  Export"])
tab_run,tab_res,tab_scen,tab_sens,tab_sec,tab_mat,tab_coef,tab_diag,tab_exp = tabs

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:
    ms = st.session_state.model_state
    if ms is None:
        st.info("Upload all 4 data files and click **Build Model** in the sidebar.")
        st.markdown("""
        | File | Source |
        |---|---|
        | `IOUse_After_Redefinitions_PRO_Detail.xlsx` | [BEA I-O Accounts](https://www.bea.gov/industry/input-output-accounts-data) |
        | `IxC_MS_Detail.xlsx` | [BEA I-O Accounts](https://www.bea.gov/industry/input-output-accounts-data) |
        | `CxI_Domestic_DR_Detail.xlsx` | [BEA I-O Accounts](https://www.bea.gov/industry/input-output-accounts-data) |
        | `51199.csv` | [BLS QCEW](https://www.bls.gov/cew/downloadable-data.htm) |
        """); st.stop()

    st.markdown('<div class="section-header">Project Parameters</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        project_name = st.text_input("Project Name", value="New Investment Project")
        naics_input  = st.text_input("Primary NAICS Code (2–4 digit)", value="23")
        naics_valid = False; primary_s = None
        try:
            primary_s = me.naics_to_sector(naics_input); naics_valid = True
            st.success(f"→ Sector [{primary_s}]: **{me.SECTOR_LABELS[primary_s]}**")
        except ValueError:
            if naics_input: st.error(f"NAICS '{naics_input}' not recognized.")
    with c2:
        investment_raw = st.number_input("Total Investment ($)",
            min_value=100_000, max_value=10_000_000_000,
            value=10_000_000, step=500_000, format="%d")
        st.markdown(f"""
        <div class='info-box'>
          FLQ δ={ms['flq_delta']:.2f} · λ={ms['flq_lambda']:.4f}<br>
          MPC={ms['nat_mpc']:.3f} · Retention={ms['regional_retention']:.2f} · HH={ms['emp_coeffs']['hh_share']:.4f}
        </div>""", unsafe_allow_html=True)

    if naics_valid and primary_s is not None:
        st.markdown('<div class="section-header" style="margin-top:1rem">BEA Spending Profile</div>',
                    unsafe_allow_html=True)
        profile = ms["all_profiles"][primary_s]
        profile_sorted = sorted(profile.items(), key=lambda x: -x[1])
        ca, cb = st.columns([1, 1.5])
        with ca:
            prof_df = pd.DataFrame([
                {"Sector": f"[{s:2d}] {me.SECTOR_LABELS[s]}", "Share": v}
                for s, v in profile_sorted])
            st.dataframe(prof_df.style.format({"Share":"{:.1%}"})
                               .bar(subset=["Share"],color="#3b82f6"),
                         hide_index=True, use_container_width=True, height=320)
        with cb:
            fig_p, ax_p = plt.subplots(figsize=(6,3.5))
            fig_p.patch.set_facecolor(DARK); dark_ax(ax_p)
            lbls=[me.SECTOR_LABELS[s][:22] for s,_ in profile_sorted[:10]]
            vals=[v for _,v in profile_sorted[:10]]
            ax_p.barh(lbls[::-1],vals[::-1],color="#3b82f6",edgecolor="none")
            ax_p.set_xlabel("Share",color="#94a3b8",fontsize=9)
            ax_p.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            fig_p.tight_layout()
            st.pyplot(fig_p,use_container_width=True); plt.close(fig_p)

    st.markdown("---")
    if st.button("▶  Run Impact Analysis",
                 disabled=not(naics_valid and project_name.strip()),
                 type="primary"):
        with st.spinner("Running…"):
            profile = ms["all_profiles"][primary_s]
            Y = np.zeros(me.N)
            ts = sum(profile.values())
            for s, sh in profile.items():
                Y[s] += investment_raw * sh / ts
            raw_res = me.compute_impacts(Y, ms["inverses"], ms["emp_coeffs"])
            res_unc = me.add_uncertainty(raw_res)
            st.session_state.results = res_unc
            rp = {"project_name":project_name,"naics":naics_input,
                  "primary_s":primary_s,"investment":investment_raw,"profile":profile}
            st.session_state.run_params = rp
            # Add to scenario store
            scenario = me.make_scenario(project_name, naics_input,
                                        investment_raw, res_unc, rp)
            scens = st.session_state.scenarios or []
            scens.append(scenario)
            st.session_state.scenarios = scens
        st.success("Analysis complete — see Results tab.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_res:
    res = st.session_state.results
    rp  = st.session_state.run_params
    ms  = st.session_state.model_state
    if res is None: st.info("Run an analysis first."); st.stop()

    t  = res["total"]; d  = res["direct"]
    ii = res["indirect"]; ind = res["indirect"]
    ind= res["induced"]; m = res["multipliers"]

    st.markdown(f"""
    <div style='margin-bottom:1rem'>
      <span style='font-size:1.1rem;font-weight:700;color:#f1f5f9'>{rp['project_name']}</span>
      &nbsp;&nbsp;
      <span class='source-chip'>NAICS {rp['naics']} · {me.SECTOR_LABELS[rp['primary_s']]}</span>
      &nbsp;<span class='source-chip'>{fmt(rp['investment'])}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='unc-note'>
    ⚠️ <b>Uncertainty disclosure:</b> All values shown with ±{res['uncertainty_pct']*100:.0f}% bands.
    This reflects methodological uncertainty inherent in Leontief I-O models (Watson et al. 2015)
    combined with the 2017 data vintage applied to the current economy. Treat point estimates
    as the midpoint of a plausible range, not as precise predictions.
    </div>
    """, unsafe_allow_html=True)

    for chk in res["mult_checks"]:
        if not chk["ok"]:
            st.markdown(f"""<div class='warn-box'>⚠️ <b>{chk['metric']}</b> =
            {chk['value']:.4f}× outside range [{chk['low']:.1f}–{chk['high']:.1f}].</div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-header">Total Economic Impact</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    t_bands = t["bands"]
    for col,lbl,val,sub,band in [
        (k1,"Total Output",     fmt(t["output"]),      f"{t['output']/rp['investment']:.2f}× ROI",
         fmt_band(t_bands["output_low"],t_bands["output_high"])),
        (k2,"Total Jobs",       f"{t['jobs']:,.0f}",    f"Mult {m['emp']:.2f}×",
         fmt_jobs_band(t_bands["jobs_low"],t_bands["jobs_high"])),
        (k3,"GDP Contribution", fmt(t["value_added"]), f"{t['value_added']/rp['investment']:.1%} of invest.",
         fmt_band(t_bands["value_added_low"],t_bands["value_added_high"])),
        (k4,"Labor Income",     fmt(t["labor_income"]),f"{t['labor_income']/t['output']:.1%} of output",
         fmt_band(t_bands["labor_income_low"],t_bands["labor_income_high"])),
        (k5,"Type II Mult.",    f"{m['type2']:.3f}×",  f"Type I: {m['type1']:.3f}×",""),
    ]:
        col.markdown(metric_card(lbl,val,sub,band), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Impact by Type</div>', unsafe_allow_html=True)

    impact_rows = []
    for key,lbl in [("direct","Direct"),("indirect","Indirect"),
                    ("induced","Induced"),("total","TOTAL")]:
        imp = res[key]; b = imp["bands"]
        impact_rows.append({
            "Type": lbl,
            "Output": fmt(imp["output"]),
            "Output Range": fmt_band(b["output_low"],b["output_high"]),
            "Jobs": f"{imp['jobs']:,.1f}",
            "Jobs Range": fmt_jobs_band(b["jobs_low"],b["jobs_high"]),
            "Labor Income": fmt(imp["labor_income"]),
            "Value Added": fmt(imp["value_added"]),
        })
    st.dataframe(pd.DataFrame(impact_rows), hide_index=True, use_container_width=True)

    # Charts
    st.markdown('<div class="section-header" style="margin-top:1.5rem">Visualizations</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(2,3,figsize=(16,9))
    fig.patch.set_facecolor(DARK)
    for ax in axes.flat: dark_ax(ax)

    # 1 – Stacked output with uncertainty
    ax = axes[0,0]
    vals_out = [d["output"]/1e6, res["indirect"]["output"]/1e6, ind["output"]/1e6]
    ax.bar([0],[vals_out[0]],color=CLRS["direct"],label="Direct",width=0.5)
    ax.bar([0],[vals_out[1]],color=CLRS["indirect"],label="Indirect",
           bottom=[vals_out[0]],width=0.5)
    ax.bar([0],[vals_out[2]],color=CLRS["induced"],label="Induced",
           bottom=[vals_out[0]+vals_out[1]],width=0.5)
    total_out = sum(vals_out)
    ax.errorbar([0],[total_out],
                yerr=[[total_out*me.UNCERTAINTY_PCT],[total_out*me.UNCERTAINTY_PCT]],
                fmt="none",color="#f1f5f9",capsize=8,linewidth=2)
    ax.set_title("Output Stacked ($M)",color="#e2e8f0",fontweight="bold",fontsize=9)
    ax.set_ylabel("$M",color="#94a3b8",fontsize=8)
    ax.set_xticks([0]); ax.set_xticklabels(["Total"],color="#94a3b8")
    ax.legend(fontsize=7,facecolor=PANEL,labelcolor="#e2e8f0")

    # 2 – Jobs with uncertainty error bars
    ax = axes[0,1]
    job_cats = ["Direct","Indirect","Induced","Total"]
    job_vals = [d["jobs"],res["indirect"]["jobs"],ind["jobs"],t["jobs"]]
    clrs2 = [CLRS["direct"],CLRS["indirect"],CLRS["induced"],CLRS["total"]]
    bars2 = ax.bar(job_cats,job_vals,color=clrs2,width=0.5)
    ax.errorbar(range(4),job_vals,
                yerr=[[v*me.UNCERTAINTY_PCT for v in job_vals],
                      [v*me.UNCERTAINTY_PCT for v in job_vals]],
                fmt="none",color="#94a3b8",capsize=5,linewidth=1.2)
    ax.set_title("Jobs (with ±30% uncertainty)",color="#e2e8f0",fontweight="bold",fontsize=9)
    for b in bars2:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,
                f"{b.get_height():.0f}",ha="center",va="bottom",color="#e2e8f0",fontsize=8)

    # 3 – Multipliers
    ax = axes[0,2]
    ml = ["Type I\nOutput","Type II\nOutput","Employment","Labor\nIncome","Value\nAdded"]
    mv = [m["type1"],m["type2"],m["emp"],m["li"],m["va"]]
    mc_ok = [chk["ok"] for chk in res["mult_checks"]] + [True,True]
    clrs3 = [CLRS["indirect"] if ok else "#ef4444" for ok in mc_ok]
    b3 = ax.bar(ml,mv,color=clrs3,width=0.5)
    ax.axhline(y=1.0,color="#f87171",linestyle="--",linewidth=1)
    ax.set_title("Economic Multipliers",color="#e2e8f0",fontweight="bold",fontsize=9)
    for b in b3:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,
                f"{b.get_height():.3f}×",ha="center",va="bottom",color="#e2e8f0",fontsize=8)

    # 4 – Output pie
    ax = axes[1,0]
    ax.pie([d["output"],res["indirect"]["output"],ind["output"]],
           labels=["Direct","Indirect","Induced"],
           colors=[CLRS["direct"],CLRS["indirect"],CLRS["induced"]],
           autopct="%1.1f%%",startangle=90,
           textprops={"color":"#e2e8f0","fontsize":8},
           wedgeprops={"edgecolor":DARK,"linewidth":1.5})
    ax.set_title("Output Composition",color="#e2e8f0",fontweight="bold",fontsize=9)

    # 5 – Top sectors by total jobs (17 sectors now)
    ax = axes[1,1]
    sec_jobs = [(me.SECTOR_LABELS[s],
                 res["direct"]["jobs_vec"][s]+res["indirect"]["jobs_vec"][s]
                 +res["induced"]["jobs_vec"][s])
                for s in range(me.N)]
    sec_jobs = sorted([(l,v) for l,v in sec_jobs if v>0.05],key=lambda x:-x[1])
    if sec_jobs:
        lbls,vls = zip(*sec_jobs[:10])
        colors_j = [me.SECTOR_COLORS[me.SECTOR_LABELS.index(l)] for l in lbls]
        ax.barh([l[:22] for l in lbls][::-1],list(vls)[::-1],
                color=colors_j[::-1],edgecolor="none")
        ax.set_title("Top Sectors — Total Jobs",color="#e2e8f0",fontweight="bold",fontsize=9)
        ax.set_xlabel("Jobs",color="#94a3b8",fontsize=8)

    # 6 – HH retention breakdown
    ax = axes[1,2]
    mpc_v=ms["nat_mpc"]; rr_v=ms["regional_retention"]; hh_v=ms["emp_coeffs"]["hh_share"]
    ax.barh(["MPC","× Retention","= HH Share"],[mpc_v,rr_v,hh_v],
            color=[CLRS["direct"],CLRS["indirect"],CLRS["total"]],edgecolor="none")
    ax.set_xlim(0,1.1)
    ax.set_title("Household Spending Retention",color="#e2e8f0",fontweight="bold",fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    for i,v in enumerate([mpc_v,rr_v,hh_v]):
        ax.text(v+0.02,i,f"{v:.3f}",va="center",color="#e2e8f0",fontsize=9,fontweight="bold")

    fig.suptitle(f"{rp['project_name']}  ·  {fmt(rp['investment'])}",
                 color="#f1f5f9",fontsize=11,fontweight="bold",y=1.01)
    fig.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scen:
    scens = st.session_state.scenarios or []
    if not scens:
        st.info("Run at least one analysis to see scenarios here. "
                "Each run is automatically saved. Run multiple projects "
                "to compare them side by side."); st.stop()

    st.markdown('<div class="section-header">Scenario Comparison</div>',
                unsafe_allow_html=True)
    st.caption(f"{len(scens)} scenario(s) saved. Run more from the Run Analysis tab.")

    comp_rows = []
    for sc in scens:
        comp_rows.append({
            "Project":      sc["name"],
            "NAICS":        sc["naics"],
            "Sector":       sc["sector"],
            "Investment":   fmt(sc["investment"]),
            "Total Output": fmt(sc["output"]),
            "Total Jobs":   f"{sc['jobs']:,.0f}",
            "Jobs Range":   fmt_jobs_band(sc["jobs_low"],sc["jobs_high"]),
            "Labor Income": fmt(sc["labor_income"]),
            "GDP (VA)":     fmt(sc["value_added"]),
            "Type I":       f"{sc['type1']:.3f}×",
            "Type II":      f"{sc['type2']:.3f}×",
            "Emp Mult":     f"{sc['emp_mult']:.3f}×",
        })
    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    if len(scens) >= 2:
        st.markdown('<div class="section-header" style="margin-top:1.5rem">Visual Comparison</div>',
                    unsafe_allow_html=True)
        fig_sc, axes_sc = plt.subplots(1,3,figsize=(15,5))
        fig_sc.patch.set_facecolor(DARK)
        for ax in axes_sc: dark_ax(ax)

        names   = [sc["name"][:20] for sc in scens]
        jobs_v  = [sc["jobs"]      for sc in scens]
        jobs_lo = [sc["jobs_low"]  for sc in scens]
        jobs_hi = [sc["jobs_high"] for sc in scens]
        out_v   = [sc["output"]/1e6 for sc in scens]
        t2_v    = [sc["type2"]     for sc in scens]
        colors_sc = [me.SECTOR_COLORS[i % len(me.SECTOR_COLORS)] for i in range(len(scens))]

        # Jobs comparison with uncertainty
        ax = axes_sc[0]
        x = np.arange(len(scens))
        ax.bar(x,jobs_v,color=colors_sc,edgecolor="none",width=0.5)
        ax.errorbar(x,jobs_v,
                    yerr=[[j*me.UNCERTAINTY_PCT for j in jobs_v],
                          [j*me.UNCERTAINTY_PCT for j in jobs_v]],
                    fmt="none",color="#94a3b8",capsize=6,linewidth=1.5)
        ax.set_xticks(x); ax.set_xticklabels(names,rotation=25,ha="right",
                                               color="#94a3b8",fontsize=8)
        ax.set_title("Total Jobs (±30%)",color="#e2e8f0",fontweight="bold",fontsize=10)

        # Output comparison
        ax = axes_sc[1]
        ax.bar(x,out_v,color=colors_sc,edgecolor="none",width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(names,rotation=25,ha="right",
                                               color="#94a3b8",fontsize=8)
        ax.set_title("Total Output ($M)",color="#e2e8f0",fontweight="bold",fontsize=10)
        ax.set_ylabel("$M",color="#94a3b8",fontsize=8)

        # Type II multiplier comparison
        ax = axes_sc[2]
        ax.bar(x,t2_v,color=colors_sc,edgecolor="none",width=0.5)
        ax.axhline(y=1.0,color="#f87171",linestyle="--",linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(names,rotation=25,ha="right",
                                               color="#94a3b8",fontsize=8)
        ax.set_title("Type II Multiplier",color="#e2e8f0",fontweight="bold",fontsize=10)
        for i,v in enumerate(t2_v):
            ax.text(i,v+0.01,f"{v:.3f}×",ha="center",va="bottom",
                    color="#e2e8f0",fontsize=8)

        fig_sc.tight_layout()
        st.pyplot(fig_sc,use_container_width=True); plt.close(fig_sc)

    if st.button("🗑️ Clear All Scenarios"):
        st.session_state.scenarios = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sens:
    res = st.session_state.results
    ms  = st.session_state.model_state
    rp  = st.session_state.run_params
    if res is None or ms is None:
        st.info("Run an analysis first."); st.stop()

    st.markdown('<div class="section-header">FLQ Delta Sensitivity</div>',
                unsafe_allow_html=True)
    st.markdown("""
    This shows how your results change as the FLQ delta parameter varies from 0.10 to 0.35.
    The **current model** uses the delta you set in the sidebar (highlighted in the table).
    Use this to understand how sensitive your results are to the regionalization assumption.
    """)

    with st.spinner("Computing sensitivity across delta values…"):
        profile = rp["profile"]
        Y = np.zeros(me.N)
        ts = sum(profile.values())
        for s, sh in profile.items():
            Y[s] += rp["investment"] * sh / ts

        sens_rows = me.sensitivity_analysis(
            Y, ms["A_nat17"], ms["qcew"], ms["national_emp"].sum(),
            ms["emp_coeffs"], ms["pce_shares"],
        )

    sens_df = pd.DataFrame(sens_rows)
    current_delta = ms["flq_delta"]

    # Display table
    disp_df = sens_df.copy()
    disp_df["delta"] = disp_df["delta"].apply(lambda v: f"{v:.2f}" + (" ◀ current" if abs(v-current_delta)<0.001 else ""))
    disp_df["lambda"] = disp_df["lambda"].apply(lambda v: f"{v:.4f}")
    disp_df["total_jobs"] = disp_df["total_jobs"].apply(lambda v: f"{v:,.1f}")
    disp_df["total_output"] = disp_df["total_output"].apply(lambda v: fmt(v))
    disp_df["type2"] = disp_df["type2"].apply(lambda v: f"{v:.4f}×")
    disp_df["avg_rpc_flq"] = disp_df["avg_rpc_flq"].apply(lambda v: f"{v:.4f}")
    disp_df.columns = ["Delta","Lambda","Total Jobs","Total Output",
                       "Type I","Type II","Emp Mult","Avg RPC"]
    st.dataframe(disp_df, hide_index=True, use_container_width=True)

    # Sensitivity chart
    fig_s, axes_s = plt.subplots(1,3,figsize=(15,4))
    fig_s.patch.set_facecolor(DARK)
    for ax in axes_s: dark_ax(ax)

    deltas = [r["delta"] for r in sens_rows]
    jobs_s = [r["total_jobs"] for r in sens_rows]
    t2_s   = [r["type2"] for r in sens_rows]
    rpc_s  = [r["avg_rpc_flq"] for r in sens_rows]

    for ax, yvals, title, ylabel in [
        (axes_s[0], jobs_s, "Total Jobs vs Delta", "Jobs"),
        (axes_s[1], t2_s,   "Type II Mult vs Delta", "Multiplier"),
        (axes_s[2], rpc_s,  "Avg RPC vs Delta", "Avg RPC"),
    ]:
        ax.plot(deltas, yvals, color="#3b82f6", linewidth=2, marker="o",
                markersize=6)
        # Fill uncertainty band around the line
        if ax == axes_s[0]:
            ax.fill_between(deltas,
                            [v*(1-me.UNCERTAINTY_PCT) for v in yvals],
                            [v*(1+me.UNCERTAINTY_PCT) for v in yvals],
                            alpha=0.2, color="#3b82f6")
        # Mark current delta
        ax.axvline(x=current_delta, color="#f59e0b", linestyle="--",
                   linewidth=1.5, label=f"Current δ={current_delta}")
        ax.set_title(title, color="#e2e8f0", fontweight="bold", fontsize=9)
        ax.set_xlabel("FLQ Delta (δ)", color="#94a3b8", fontsize=8)
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=8)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor="#e2e8f0")

    fig_s.suptitle(f"Sensitivity — {rp['project_name']}  ·  {fmt(rp['investment'])}",
                   color="#f1f5f9", fontsize=10, fontweight="bold")
    fig_s.tight_layout()
    st.pyplot(fig_s, use_container_width=True); plt.close(fig_s)

    jobs_min = min(jobs_s); jobs_max = max(jobs_s)
    st.markdown(f"""
    <div class='info-box'>
    <b>Sensitivity range across δ=0.10–0.35:</b><br>
    Total jobs: {jobs_min:,.0f} – {jobs_max:,.0f}
    (spread of {jobs_max-jobs_min:,.0f} jobs = {(jobs_max-jobs_min)/jobs_min*100:.0f}% of low estimate)<br>
    Type II multiplier: {min(t2_s):.3f}× – {max(t2_s):.3f}×
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SECTOR DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sec:
    res = st.session_state.results; ms = st.session_state.model_state
    rp  = st.session_state.run_params
    if res is None or ms is None: st.info("Run an analysis first."); st.stop()

    st.markdown('<div class="section-header">17-Sector Output ($)</div>',
                unsafe_allow_html=True)
    out_rows = []
    for s in range(me.N):
        dv=res["vecs"]["direct"][s]; iv=res["vecs"]["indirect"][s]
        nv=res["vecs"]["induced"][s]; tv=dv+iv+nv
        # Highlight split sectors
        flag = " 🔵" if s==15 else " 🟢" if s==16 else (" ⚪" if s==14 else "")
        out_rows.append({"Sector":me.SECTOR_LABELS[s]+flag,
                         "Direct":fmt(dv),"Indirect":fmt(iv),"Induced":fmt(nv),"Total":fmt(tv)})
    st.dataframe(pd.DataFrame(out_rows), hide_index=True, use_container_width=True)
    st.caption("🔵 Federal Gov & Defense  🟢 State & Local Gov  ⚪ Private Other Services — the three sub-sectors split from old sector 14")

    st.markdown('<div class="section-header" style="margin-top:1rem">17-Sector Employment (Jobs)</div>',
                unsafe_allow_html=True)
    job_rows = []
    for s in range(me.N):
        dj=res["direct"]["jobs_vec"][s]; ij=res["indirect"]["jobs_vec"][s]
        nj=res["induced"]["jobs_vec"][s]
        job_rows.append({"Sector":me.SECTOR_LABELS[s],
                         "Direct":f"{dj:.2f}","Indirect":f"{ij:.2f}",
                         "Induced":f"{nj:.2f}","Total":f"{dj+ij+nj:.2f}"})
    st.dataframe(pd.DataFrame(job_rows), hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1rem">LQ & FLQ Regionalization</div>',
                unsafe_allow_html=True)
    lq_rows = []
    for s in range(me.N):
        lq=float(ms["lq_york"][s]); rpc_flq=float(ms["rpc_york"][s])
        rpc_slq=float(ms["rpc_slq"][s])
        lq_rows.append({"Sector":me.SECTOR_LABELS[s],
                         "LQ":f"{lq:.3f}","RPC (FLQ)":f"{rpc_flq:.4f}",
                         "RPC (SLQ)":f"{rpc_slq:.3f}",
                         "FLQ Δ":f"+{(rpc_slq-rpc_flq):.3f}",
                         "Empl":f"{ms['qcew']['emp'][s]:,.0f}",
                         "A_york":f"{ms['A_york'][:,s].sum():.4f}",
                         "Status":"✓ Local" if lq>=1.0 else f"↓{(1-rpc_flq):.0%} imported"})
    st.dataframe(pd.DataFrame(lq_rows), hide_index=True, use_container_width=True)
    st.caption(f"FLQ λ={ms['flq_lambda']:.4f} (δ={ms['flq_delta']})  ·  "
               f"Avg FLQ leakage: {ms['import_leakage_rate']:.1%}  ·  "
               f"Avg SLQ leakage: {ms['import_leakage_rate_slq']:.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — A MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mat:
    ms = st.session_state.model_state
    if ms is None: st.info("Build the model first."); st.stop()

    view_choice = st.radio("View",["York County (FLQ)","National (A_domestic)"],horizontal=True)
    A_show = ms["A_york"] if "York" in view_choice else ms["A_nat17"]
    A_df = pd.DataFrame(A_show, index=me.SECTOR_LABELS, columns=me.SECTOR_LABELS)
    st.dataframe(A_df.style.format("{:.4f}").background_gradient(cmap="Blues",axis=None),
                 use_container_width=True)
    st.caption(f"17×17 matrix.  Rows = supplying sector.  Columns = purchasing sector.  "
               f"Sectors 14–16 are the split government rows.")

    fig_h, ax_h = plt.subplots(figsize=(12,10))
    fig_h.patch.set_facecolor(DARK); ax_h.set_facecolor(DARK)
    im = ax_h.imshow(A_show, cmap="YlOrRd", aspect="auto", vmin=0)
    ax_h.set_xticks(range(me.N)); ax_h.set_yticks(range(me.N))
    ax_h.set_xticklabels([f"{s}" for s in range(me.N)], color="#94a3b8", fontsize=8)
    ax_h.set_yticklabels([l[:25] for l in me.SECTOR_LABELS], color="#94a3b8", fontsize=8)
    ax_h.axhline(y=13.5, color="#f59e0b", linewidth=1, linestyle="--", alpha=0.5)
    ax_h.axvline(x=13.5, color="#f59e0b", linewidth=1, linestyle="--", alpha=0.5)
    fig_h.colorbar(im, ax=ax_h, label="Coefficient")
    ax_h.set_title(f"A Matrix — {'York County FLQ' if 'York' in view_choice else 'National'}"
                   f" (dashed line = sector 14 split)",
                   color="#e2e8f0", fontsize=11, fontweight="bold")
    fig_h.tight_layout()
    st.pyplot(fig_h, use_container_width=True); plt.close(fig_h)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_coef:
    ms = st.session_state.model_state
    if ms is None: st.info("Build the model first."); st.stop()

    src_map = {"live":"✅ Live BEA","csv_cache":"📁 CSV Cache","hardcoded_fallback":"⚠️ 2022 Const"}
    st.markdown(f"""
    <div style='display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:1rem'>
      <span class='source-chip'>VA: {src_map.get(ms['bea_va_src'],'?')}</span>
      <span class='source-chip'>MPC: {ms['nat_mpc']:.4f}</span>
      <span class='source-chip'>Retention: {ms['regional_retention']:.2f}</span>
      <span class='source-chip'>HH share: {ms['emp_coeffs']['hh_share']:.4f}</span>
    </div>""", unsafe_allow_html=True)

    val_results = ms["validation"]
    if any(not v["ok"] for v in val_results):
        st.warning("⚠️ One or more coefficients are outside plausible ranges.")
    else:
        st.success("✅ All 17 employment coefficients are within plausible ranges.")

    coeff_rows = []
    for i, v in enumerate(val_results):
        coeff_rows.append({
            "Sector":         v["sector"],
            "Nat'l Empl (k)": f"{ms['national_emp'][i]:,.1f}",
            "VA $M":          f"{ms['bea_va'][i]:,.0f}",
            "Jobs/$1M VA":    f"{v['jpm']:.2f}",
            "York Avg Wage":  fmt(v["wage"]),
            "LI Share":       f"{v['li']:.1%}",
            "Status":         "✅" if v["ok"] else "⚠️ " + "; ".join(v["flags"]),
        })
    st.dataframe(pd.DataFrame(coeff_rows), hide_index=True, use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1rem">QCEW Employment (17 sectors)</div>',
                unsafe_allow_html=True)
    qr = []
    for s in range(me.N):
        own = "Private" if s<=14 else ("Federal" if s==15 else "State/Local")
        qr.append({"Sector":me.SECTOR_LABELS[s],"Ownership":own,
                   "Employment":f"{ms['qcew']['emp'][s]:,.0f}",
                   "LQ":f"{ms['qcew']['lq'][s]:.3f}",
                   "FLQ RPC":f"{ms['rpc_york'][s]:.4f}",
                   "Avg Annual Pay":fmt(ms['qcew']['wage'][s])})
    st.dataframe(pd.DataFrame(qr), hide_index=True, use_container_width=True)
    st.caption(f"QCEW {ms['qcew']['year']}  ·  FIPS {ms['qcew']['fips']}  ·  "
               f"Total: {ms['qcew']['emp'].sum():,.0f} workers")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    ms  = st.session_state.model_state
    res = st.session_state.results
    if ms is None: st.info("Build the model first."); st.stop()

    stab = ms["stability"]
    if stab["stable"]:
        st.markdown(f"""<div class='ok-box'>✅ <b>Leontief system stable</b><br>
        Spectral radius = <b>{stab['spectral_radius']:.6f}</b> &lt; 1.0  ·
        Max col sum = {stab['max_col_sum']:.4f}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='warn-box'>⚠️ <b>Unstable</b><br>
        Spectral radius = {stab['spectral_radius']:.6f} ≥ 1.0</div>""", unsafe_allow_html=True)

    eigvals = np.linalg.eigvals(ms["A_york"])
    fig_e, ax_e = plt.subplots(figsize=(6,6))
    fig_e.patch.set_facecolor(DARK); ax_e.set_facecolor(PANEL)
    theta = np.linspace(0,2*np.pi,200)
    ax_e.plot(np.cos(theta),np.sin(theta),color="#ef4444",linewidth=1.5,
              linestyle="--",label="Unit circle")
    ax_e.scatter(eigvals.real,eigvals.imag,color="#3b82f6",s=40,zorder=5,label="Eigenvalues")
    ax_e.axhline(0,color="#475569",linewidth=0.5); ax_e.axvline(0,color="#475569",linewidth=0.5)
    ax_e.set_xlim(-1.5,1.5); ax_e.set_ylim(-1.5,1.5)
    ax_e.set_title(f"A_york Eigenvalues (ρ={stab['spectral_radius']:.4f})",
                   color="#e2e8f0",fontweight="bold")
    ax_e.legend(fontsize=8,facecolor=PANEL,labelcolor="#e2e8f0")
    for sp in ax_e.spines.values(): sp.set_visible(False)
    ax_e.tick_params(colors="#94a3b8")
    fig_e.tight_layout()
    st.pyplot(fig_e,use_container_width=True); plt.close(fig_e)

    st.markdown("---")
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Sectors | 17 (15 original + Federal + State/Local split) |
    | FLQ delta (δ) | {ms['flq_delta']} |
    | FLQ lambda (λ) | {ms['flq_lambda']:.4f} |
    | County employment | {ms['qcew']['total_county_emp']:,.0f} |
    | National employment | {ms['national_emp'].sum()*1000:,.0f} |
    | Avg RPC — SLQ | {ms['rpc_slq'].mean():.4f} |
    | Avg RPC — FLQ | {ms['rpc_york'].mean():.4f} |
    | Import leakage — SLQ | {ms['import_leakage_rate_slq']:.1%} |
    | Import leakage — FLQ | {ms['import_leakage_rate']:.1%} |
    | Uncertainty band | ±{me.UNCERTAINTY_PCT*100:.0f}% (Watson et al. 2015) |
    """)
    if res:
        st.markdown("**Multiplier checks (last run):**")
        for chk in res["mult_checks"]:
            st.markdown(f"{'✅' if chk['ok'] else '⚠️'} **{chk['metric']}** = "
                        f"{chk['value']:.4f}× (range {chk['low']:.1f}–{chk['high']:.1f})")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_exp:
    res = st.session_state.results
    ms  = st.session_state.model_state
    rp  = st.session_state.run_params
    if res is None or ms is None: st.info("Run an analysis first."); st.stop()

    st.markdown('<div class="section-header">Download Results</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # Impact summary with uncertainty bands
        summ_rows = []
        for key,lbl in [("direct","Direct"),("indirect","Indirect"),
                        ("induced","Induced"),("total","TOTAL")]:
            imp=res[key]; b=imp["bands"]
            summ_rows.append({
                "Impact Type":      lbl,
                "Output ($)":       round(imp["output"],2),
                "Output Low ($)":   round(b["output_low"],2),
                "Output High ($)":  round(b["output_high"],2),
                "Jobs":             round(imp["jobs"],2),
                "Jobs Low":         round(b["jobs_low"],2),
                "Jobs High":        round(b["jobs_high"],2),
                "Labor Income ($)": round(imp["labor_income"],2),
                "Value Added ($)":  round(imp["value_added"],2),
            })
        st.download_button("📥 Impact Summary CSV (with bands)",
            data=pd.DataFrame(summ_rows).to_csv(index=False).encode(),
            file_name="impact_summary_v6.csv", mime="text/csv",
            use_container_width=True)

        # Sector detail
        sec_rows = []
        for s in range(me.N):
            dj=res["direct"]["jobs_vec"][s]; ij=res["indirect"]["jobs_vec"][s]
            nj=res["induced"]["jobs_vec"][s]
            sec_rows.append({
                "Sector":        me.SECTOR_LABELS[s],
                "LQ":            round(float(ms["lq_york"][s]),4),
                "RPC_FLQ":       round(float(ms["rpc_york"][s]),4),
                "RPC_SLQ":       round(float(ms["rpc_slq"][s]),4),
                "Employment":    round(float(ms["qcew"]["emp"][s]),0),
                "AvgWage":       round(float(ms["qcew"]["wage"][s]),0),
                "Jobs_per_1M_VA":round(float(ms["emp_coeffs"]["jobs_per_va"][s]),4),
                "LI_Share":      round(float(ms["emp_coeffs"]["li_share"][s]),4),
                "Direct_Jobs":   round(dj,2),"Indirect_Jobs":round(ij,2),
                "Induced_Jobs":  round(nj,2),"Total_Jobs":round(dj+ij+nj,2),
                "Direct_Output": round(float(res["vecs"]["direct"][s]),2),
                "Total_Output":  round(float(res["vecs"]["total"][s]),2),
            })
        st.download_button("📥 Sector Detail CSV (17 sectors)",
            data=pd.DataFrame(sec_rows).to_csv(index=False).encode(),
            file_name="impact_by_sector_v6.csv", mime="text/csv",
            use_container_width=True)

    with c2:
        m = res["multipliers"]
        mult_df = pd.DataFrame([{
            "Project":           rp["project_name"],
            "NAICS":             rp["naics"],
            "Region":            f"FIPS {ms['qcew']['fips']}",
            "Investment($)":     rp["investment"],
            "Sectors":           17,
            "IO_Data":           "BEA 2017 Benchmark After-Redefinitions Domestic",
            "Regionalization":   f"FLQ delta={ms['flq_delta']} lambda={ms['flq_lambda']:.4f}",
            "MPC":               round(ms["nat_mpc"],4),
            "Regional_Retention":ms["regional_retention"],
            "Uncertainty_Pct":   me.UNCERTAINTY_PCT,
            "Type_I":            round(m["type1"],4),
            "Type_II":           round(m["type2"],4),
            "Emp_Mult":          round(m["emp"],4),
            "Spectral_Radius":   round(ms["stability"]["spectral_radius"],6),
        }])
        st.download_button("📥 Multipliers CSV",
            data=mult_df.to_csv(index=False).encode(),
            file_name="multipliers_v6.csv", mime="text/csv",
            use_container_width=True)

        # Scenario comparison CSV
        if st.session_state.scenarios:
            scen_rows = [{
                "Name":sc["name"],"NAICS":sc["naics"],"Sector":sc["sector"],
                "Investment":sc["investment"],
                "Total_Output":round(sc["output"],2),
                "Total_Jobs":round(sc["jobs"],2),
                "Jobs_Low":round(sc["jobs_low"],2),
                "Jobs_High":round(sc["jobs_high"],2),
                "Labor_Income":round(sc["labor_income"],2),
                "Value_Added":round(sc["value_added"],2),
                "Type_I":round(sc["type1"],4),
                "Type_II":round(sc["type2"],4),
                "Emp_Mult":round(sc["emp_mult"],4),
            } for sc in st.session_state.scenarios]
            st.download_button("📥 Scenario Comparison CSV",
                data=pd.DataFrame(scen_rows).to_csv(index=False).encode(),
                file_name="scenario_comparison_v6.csv", mime="text/csv",
                use_container_width=True)

        A_york_df = pd.DataFrame(ms["A_york"],
                                  index=me.SECTOR_LABELS, columns=me.SECTOR_LABELS)
        st.download_button("📥 A_york Matrix CSV (17×17)",
            data=A_york_df.to_csv().encode(),
            file_name="A_york_FLQ_17sector.csv", mime="text/csv",
            use_container_width=True)

    st.markdown("---")
    st.markdown(f"""
**v6.0 methodology:** 17-sector model (split sector 14 into Private Other Services, Federal Gov & Defense, State & Local Gov)  ·
FLQ δ={ms['flq_delta']} λ={ms['flq_lambda']:.4f}  ·  MPC={ms['nat_mpc']:.4f}  ·
Retention={ms['regional_retention']:.2f}  ·  Uncertainty ±{me.UNCERTAINTY_PCT*100:.0f}%  ·
Spectral radius={ms['stability']['spectral_radius']:.4f}
    """)
