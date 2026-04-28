"""
Regional Economic Impact Analysis Tool  v7.3
Streamlit Application — SDP Regionalization

Run:  streamlit run app.py
"""
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
.mixed-use-card{background:#1e293b;border:1px solid #334155;border-radius:8px;
  padding:1rem 1.2rem;margin-bottom:0.5rem;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}
section[data-testid="stSidebar"]{background:#0f172a !important;}
section[data-testid="stSidebar"] *{color:#cbd5e1 !important;}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k in ["model_state","results","run_params","scenarios","mixed_use_results"]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "scenarios" else []

# ── HELPERS ───────────────────────────────────────────────────────────────────
def fmt(v):
    if abs(v)>=1e9: return f"${v/1e9:,.2f}B"
    if abs(v)>=1e6: return f"${v/1e6:,.1f}M"
    if abs(v)>=1e3: return f"${v/1e3:,.0f}K"
    return f"${v:,.0f}"

def fmt_band(lo, hi): return f"[{fmt(lo)} – {fmt(hi)}]"
def fmt_jobs_band(lo, hi): return f"[{lo:,.0f} – {hi:,.0f}]"

def metric_card(label, value, sub="", band=""):
    return (f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            + (f'<div class="sub">{sub}</div>' if sub else "")
            + (f'<div class="band">±30%: {band}</div>' if band else "")
            + '</div>')

DARK = "#0f172a"; PANEL = "#1e293b"
CLRS = {"direct":"#3b82f6","indirect":"#22c55e","induced":"#f59e0b","total":"#a855f7"}

def dark_ax(ax):
    ax.set_facecolor(PANEL); ax.tick_params(colors="#94a3b8",labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.grid(color="#334155",alpha=0.5,linewidth=0.5)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <h1>📊 Regional Economic Impact Analysis</h1>
  <p>v{me.VERSION} · Supply-Demand Pool Regionalization · {me.N} Sectors ·
  Proprietor Employment · York County Income-Weighted PCE ·
  Uncertainty Bands · Sensitivity · Scenario Comparison · Mixed-Use Builder</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    st.markdown("**API Keys**")
    bea_key = st.text_input("BEA API Key", value="", type="password")
    bls_key = st.text_input("BLS API Key", value="", type="password")
    st.markdown("---")
    st.markdown("**Model Parameters**")

    sdp_alpha = st.slider(
        "SDP Alpha (α) — Cross-Hauling",
        min_value=0.10, max_value=0.30, value=0.20, step=0.05,
        help=(
            "Supply-Demand Pool cross-hauling parameter.\n"
            "Controls how much importing occurs even in locally-specialized sectors.\n\n"
            "0.10 = low cross-hauling, denser regional economy\n"
            "0.20 = standard for US counties (IMPLAN-comparable multipliers)\n"
            "0.30 = high cross-hauling, very open small economy\n\n"
            "Source: Kronenberg (2009), Flegg & Tohmo (2016)"
        )
    )
    mpc_param = st.slider("MPC", 0.70, 0.98, 0.90, 0.01,
        help="Marginal propensity to consume. BEA NIPA 2022 ≈ 0.96.")
    rr_param  = st.slider("Regional Retention", 0.30, 0.70, 0.50, 0.05,
        help="Share of household spending staying in the county.")

    st.markdown(f"""
    <div class='info-box'>
    <b>Regionalization:</b> SDP α={sdp_alpha:.2f}<br>
    <b>HH share:</b> {mpc_param:.2f} × {rr_param:.2f} = {mpc_param*rr_param:.4f}
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📂 Data Files**")
    st.caption("BEA Use Table"); use_file  = st.file_uploader("Use Table",  type=["xlsx"],label_visibility="collapsed",key="uf")
    st.caption("Market Share Matrix"); ms_file   = st.file_uploader("MS Matrix",  type=["xlsx"],label_visibility="collapsed",key="msf")
    st.caption("Domestic Requirements"); dom_file  = st.file_uploader("Dom Req",    type=["xlsx"],label_visibility="collapsed",key="df")
    st.caption("QCEW County CSV"); qcew_file = st.file_uploader("QCEW CSV",   type=["csv"], label_visibility="collapsed",key="qf")

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
                    sdp_alpha=sdp_alpha, flq_delta=0.25,
                    mpc=mpc_param, regional_retention=rr_param,
                    progress_callback=_cb)
                st.session_state.model_state = state
                st.session_state.results = None
                st.session_state.scenarios = []
                st.session_state.mixed_use_results = None
                pb.progress(100); st_txt.caption("Model ready.")

                # ── DATA SOURCE STATUS PANEL ──────────────────────────────
                src_map = {
                    "live":               ("API live",      "✅"),
                    "csv_cache":          ("BEA 2022 CSV",  "✅"),
                    "hardcoded_fallback": ("2022 constant", "⚠️"),
                }
                va_lbl, va_icon = src_map.get(state["bea_va_src"],  ("unknown","❓"))
                go_lbl, go_icon = src_map.get(state["bea_output_src"],("unknown","❓"))
                comp_lbl = va_lbl + " (derived)"

                all_live = (state["bea_va_src"] in ("live","csv_cache") and
                            state["bea_output_src"] in ("live","csv_cache"))

                if all_live:
                    st.success(f"✅ Model built successfully — engine v{me.VERSION}")
                else:
                    st.warning("⚠️ One or more BEA tables used hardcoded constants")

                st.markdown(f"""
| Data source | Status | Details |
|---|---|---|
| BEA Value Added (Table 1) | {va_icon} {va_lbl} | jobs/\\$1M VA coefficients |
| BEA Gross Output (Table 15) | {go_icon} {go_lbl} | labor income shares |
| BEA Compensation | {va_icon} {comp_lbl} | wages derived from VA |
| BLS Employment | ✅ LIVE | {state['national_emp'].sum()*1000:,.0f} workers |
| QCEW County | ✅ FILE | {state['qcew']['emp_ws'].sum():,.0f} W&S + {state['qcew']['emp_prop'].sum():,.0f} proprietors = {state['qcew']['emp'].sum():,.0f} total, FIPS {state['qcew']['fips']} {state['qcew']['year']} |
| BEA I-O Structure | ✅ FILE | 402-industry BEA benchmark tables |
| Household PCE | ✅ FILE | York County income-weighted (ACS B19001 + BLS CES 2022) |
                """)

                if not all_live:
                    with st.expander("🔍 API diagnostic detail"):
                        st.code(
                            f"VA:  {state.get('bea_va_diag','n/a')}\n"
                            f"GO:  {state.get('bea_output_diag','n/a')}",
                            language="text"
                        )
                    st.caption(
                        "Fallback values are real BEA 2022 published data "
                        "bundled with the app. Results remain valid but "
                        "reflect 2022 rather than the most current revision. "
                        "Check the Coefficients tab for full detail."
                    )

                stab = state["stability"]
                if stab["stable"]:
                    st.success(f"✅ Leontief stable — spectral radius={stab['spectral_radius']:.4f}")
                else:
                    st.error(f"⚠️ Unstable — ρ={stab['spectral_radius']:.4f}")
            except Exception as e:
                st.error(f"Build failed: {e}")
                import traceback; st.code(traceback.format_exc())
    if not all_up:
        st.caption("Upload all 4 files to enable build.")

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["▶  Run","🏗️  Mixed-Use","📈  Results","🔀  Scenarios",
                "📉  Sensitivity","🏭  Sectors","🔢  Matrix",
                "🔬  Coefficients","🔍  Diagnostics","💾  Export"])
tab_run,tab_mixed,tab_res,tab_scen,tab_sens,tab_sec,tab_mat,tab_coef,tab_diag,tab_exp = tabs

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

        NAICS_OPTIONS = [
            # Construction
            ("23   — Construction (all types)",                    "23"),
            ("236  — Residential Building Construction",           "236"),
            ("2361 — Residential Building",                        "2361"),
            ("2362 — Nonresidential Building",                     "2362"),
            ("237  — Heavy & Civil Engineering Construction",       "237"),
            ("238  — Specialty Trade Contractors",                  "238"),
            ("2381 — Foundation & Structure Contractors",           "2381"),
            ("2382 — Building Equipment Contractors",               "2382"),
            ("2383 — Building Finishing Contractors",               "2383"),
            # Retail Trade
            ("44   — Retail Trade (all)",                          "44"),
            ("441  — Motor Vehicle & Parts Dealers",               "441"),
            ("4411 — Automobile Dealers",                          "4411"),
            ("444  — Building Material & Garden Supply",           "444"),
            ("4441 — Building Material & Supplies Dealers",        "4441"),
            ("445  — Food & Beverage Stores",                      "445"),
            ("446  — Health & Personal Care Stores",               "446"),
            ("452  — General Merchandise Stores",                  "452"),
            # Food Services & Accommodation
            ("721  — Accommodation (Hotels & Motels)",             "721"),
            ("7211 — Hotels & Motels",                             "7211"),
            ("722  — Food Services & Drinking Places",             "722"),
            ("7221 — Full-Service Restaurants",                    "7221"),
            ("7222 — Limited-Service Eating Places",               "7222"),
            # Professional & Technical Services
            ("54   — Professional & Technical Services (all)",     "54"),
            ("541  — Professional, Scientific & Tech Services",    "541"),
            ("5411 — Legal Services",                              "5411"),
            ("5412 — Accounting & Tax Preparation",                "5412"),
            ("5413 — Architectural & Engineering Services",        "5413"),
            ("5415 — Computer Systems Design",                     "5415"),
            ("5416 — Management Consulting",                       "5416"),
            ("5417 — Scientific Research & Development",           "5417"),
            ("551  — Management of Companies",                     "551"),
            # Healthcare & Education
            ("61   — Educational Services",                        "61"),
            ("611  — Educational Services",                        "611"),
            ("62   — Health Care & Social Assistance (all)",       "62"),
            ("621  — Ambulatory Health Care Services",             "621"),
            ("6211 — Offices of Physicians",                       "6211"),
            ("6212 — Offices of Dentists",                         "6212"),
            ("622  — Hospitals",                                   "622"),
            ("623  — Nursing & Residential Care",                  "623"),
            ("624  — Social Assistance",                           "624"),
            # Real Estate & Finance
            ("52   — Finance & Insurance",                         "52"),
            ("522  — Credit Intermediation",                       "522"),
            ("524  — Insurance Carriers & Related",                "524"),
            ("53   — Real Estate & Rental",                        "53"),
            ("531  — Real Estate",                                 "531"),
            ("532  — Rental & Leasing Services",                   "532"),
            # Arts, Recreation & Entertainment
            ("71   — Arts, Entertainment & Recreation",            "71"),
            ("711  — Performing Arts & Spectator Sports",          "711"),
            ("713  — Amusement, Gambling & Recreation",            "713"),
            # Manufacturing
            ("33   — Durable Goods Manufacturing (all)",           "33"),
            ("332  — Fabricated Metal Products",                   "332"),
            ("334  — Computer & Electronic Products",              "334"),
            ("335  — Electrical Equipment & Appliances",           "335"),
            ("336  — Transportation Equipment",                    "336"),
            ("32   — Non-Durable Goods Manufacturing (all)",       "32"),
            ("311  — Food Manufacturing",                          "311"),
            ("325  — Chemical Manufacturing",                      "325"),
            # Transportation & Warehousing
            ("48   — Transportation (all)",                        "48"),
            ("484  — Truck Transportation",                        "484"),
            ("485  — Transit & Ground Passenger Transport",        "485"),
            ("493  — Warehousing & Storage",                       "493"),
            # Information
            ("51   — Information",                                 "51"),
            ("517  — Telecommunications",                          "517"),
            # Wholesale Trade
            ("42   — Wholesale Trade",                             "42"),
            ("423  — Durable Goods Wholesale",                     "423"),
            ("424  — Non-Durable Goods Wholesale",                 "424"),
            # Other Services
            ("81   — Other Services (all)",                        "81"),
            ("811  — Repair & Maintenance",                        "811"),
            ("812  — Personal & Laundry Services",                 "812"),
            # Government
            ("921  — Federal Government & Defense",                "921"),
            ("922  — State Government",                            "922"),
            ("923  — Local Government",                            "923"),
            # Utilities & Mining
            ("22   — Utilities",                                   "22"),
            ("21   — Mining",                                      "21"),
            # Agriculture
            ("11   — Agriculture, Forestry & Fishing",             "11"),
        ]

        naics_labels = [opt[0] for opt in NAICS_OPTIONS]
        naics_codes  = [opt[1] for opt in NAICS_OPTIONS]

        # Default to Construction (index 0)
        naics_selection = st.selectbox(
            "NAICS Code",
            options=naics_labels,
            index=0,
            help="Select the industry that best describes your investment. "
                 "For mixed-use projects with multiple components, use the Mixed-Use tab."
        )
        naics_input = naics_codes[naics_labels.index(naics_selection)]

        naics_valid = False; primary_s = None
        try:
            primary_s = me.naics_to_sector(naics_input); naics_valid = True
            st.success(f"→ Sector [{primary_s}]: **{me.SECTOR_LABELS[primary_s]}**")
        except ValueError:
            st.error(f"NAICS '{naics_input}' not recognized.")
    with c2:
        investment_raw = st.number_input("Total Investment ($)",
            min_value=100_000, max_value=10_000_000_000,
            value=10_000_000, step=500_000, format="%d")
        st.markdown(f"""
        <div class='info-box'>
          SDP α={ms['sdp_alpha']:.2f} · Avg RPC={ms['rpc_sdp'].mean():.4f}<br>
          MPC={ms['nat_mpc']:.3f} · Retention={ms['regional_retention']:.2f}<br>
          HH share={ms['emp_coeffs']['hh_share']:.4f}
        </div>""", unsafe_allow_html=True)

    if naics_valid and primary_s is not None:
        st.markdown('<div class="section-header" style="margin-top:1rem">BEA Spending Profile</div>',
                    unsafe_allow_html=True)
        profile = ms["all_profiles"][primary_s]
        profile_sorted = sorted(profile.items(), key=lambda x: -x[1])
        ca,cb = st.columns([1,1.5])
        with ca:
            prof_df = pd.DataFrame([
                {"Sector":f"[{s:2d}] {me.SECTOR_LABELS[s]}","Share":v}
                for s,v in profile_sorted])
            st.dataframe(prof_df.style.format({"Share":"{:.1%}"})
                               .bar(subset=["Share"],color="#3b82f6"),
                         hide_index=True,use_container_width=True,height=340)
        with cb:
            fig_p,ax_p = plt.subplots(figsize=(6,3.8))
            fig_p.patch.set_facecolor(DARK); dark_ax(ax_p)
            lbls=[me.SECTOR_LABELS[s][:22] for s,_ in profile_sorted[:10]]
            vals=[v for _,v in profile_sorted[:10]]
            ax_p.barh(lbls[::-1],vals[::-1],color="#3b82f6",edgecolor="none")
            ax_p.set_xlabel("Share",color="#94a3b8",fontsize=9)
            ax_p.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            fig_p.tight_layout()
            st.pyplot(fig_p,use_container_width=True); plt.close(fig_p)

    st.markdown("---")
    st.caption("💡 For projects with multiple components (hotel + retail + restaurant), use the **Mixed-Use** tab for a more accurate split analysis.")
    if st.button("▶  Run Impact Analysis",
                 disabled=not(naics_valid and project_name.strip()),
                 type="primary"):
        with st.spinner("Running…"):
            profile = ms["all_profiles"][primary_s]
            Y = np.zeros(me.N)
            ts = sum(profile.values())
            for s,sh in profile.items():
                Y[s] += investment_raw * sh / ts
            raw = me.compute_impacts(Y, ms["inverses"], ms["emp_coeffs"])
            res_unc = me.add_uncertainty(raw)
            st.session_state.results = res_unc
            rp = {"project_name":project_name,"naics":naics_input,
                  "primary_s":primary_s,"investment":investment_raw,"profile":profile}
            st.session_state.run_params = rp
            scens = st.session_state.scenarios or []
            scens.append(me.make_scenario(project_name,naics_input,investment_raw,res_unc,rp))
            st.session_state.scenarios = scens
        st.success("Analysis complete — see Results tab.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MIXED-USE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mixed:
    ms = st.session_state.model_state
    if ms is None:
        st.info("Build the model first."); st.stop()

    st.markdown('<div class="section-header">Mixed-Use Development Impact Builder</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Break a complex project into its real components and run each separately.
    This produces results comparable to IMPLAN's multi-sector approach rather
    than forcing everything through a single NAICS code.
    Each component uses its own BEA production function and York County LQ.
    """)

    # ── Pre-defined templates ─────────────────────────────────────────────────
    TEMPLATES = {
        "Mixed-Use Development (Retail + Restaurant + Residential)": [
            {"name":"Construction Phase",     "naics":"23",  "pct":100.0, "phase":"Construction"},
            {"name":"Retail Operations",       "naics":"44",  "pct":0.0,   "phase":"Operations"},
            {"name":"Restaurant / Food Svc",   "naics":"722", "pct":0.0,   "phase":"Operations"},
            {"name":"Residential / Property",  "naics":"531", "pct":0.0,   "phase":"Operations"},
        ],
        "Hotel Development": [
            {"name":"Construction Phase",     "naics":"23",  "pct":100.0, "phase":"Construction"},
            {"name":"Hotel Operations",       "naics":"721", "pct":0.0,   "phase":"Operations"},
            {"name":"Food & Beverage",        "naics":"722", "pct":0.0,   "phase":"Operations"},
            {"name":"Retail Amenities",       "naics":"44",  "pct":0.0,   "phase":"Operations"},
        ],
        "Medical / Healthcare Campus": [
            {"name":"Construction Phase",     "naics":"236", "pct":100.0, "phase":"Construction"},
            {"name":"Ambulatory Care",        "naics":"621", "pct":0.0,   "phase":"Operations"},
            {"name":"Professional Services",  "naics":"541", "pct":0.0,   "phase":"Operations"},
            {"name":"Retail Pharmacy",        "naics":"446", "pct":0.0,   "phase":"Operations"},
        ],
        "Defense / Professional Campus": [
            {"name":"Construction Phase",     "naics":"23",  "pct":100.0, "phase":"Construction"},
            {"name":"Professional / Tech Svcs","naics":"541", "pct":0.0,  "phase":"Operations"},
            {"name":"Management Functions",   "naics":"551", "pct":0.0,   "phase":"Operations"},
            {"name":"Support Services",       "naics":"56",  "pct":0.0,   "phase":"Operations"},
        ],
        "Custom (blank)": [],
    }

    template_choice = st.selectbox("Start from a template or build custom:",
                                   list(TEMPLATES.keys()))

    st.markdown("---")
    st.markdown('<div class="section-header">Project Components</div>', unsafe_allow_html=True)
    st.caption("Define each component: its NAICS code, annual operating spend, and phase. "
               "Construction is typically one-time; operations recur annually.")

    # Build component inputs
    if template_choice == "Custom (blank)":
        default_components = [
            {"name":"Component 1","naics":"23","spend":5_000_000,"phase":"Construction"},
            {"name":"Component 2","naics":"722","spend":2_000_000,"phase":"Operations"},
        ]
    else:
        template = TEMPLATES[template_choice]
        default_components = [
            {"name":t["name"],"naics":t["naics"],
             "spend":0,"phase":t["phase"]}
            for t in template
        ]

    n_components = st.number_input("Number of components", min_value=1, max_value=8,
                                   value=len(default_components), step=1)

    components = []
    for i in range(int(n_components)):
        default = default_components[i] if i < len(default_components) else \
                  {"name":f"Component {i+1}","naics":"23","spend":0,"phase":"Construction"}
        with st.expander(f"Component {i+1} — {default['name']}", expanded=True):
            cc1,cc2,cc3,cc4 = st.columns([2,1,1.5,1])
            with cc1:
                cname = st.text_input("Component name", value=default["name"],
                                      key=f"cname_{i}")
            with cc2:
                # Find default index matching the template NAICS
                default_idx = next(
                    (i for i,opt in enumerate(NAICS_OPTIONS) if opt[1] == default["naics"]),
                    0
                )
                cnaics_label = st.selectbox(
                    "NAICS code", options=naics_labels,
                    index=default_idx, key=f"cnaics_{i}"
                )
                cnaics = naics_codes[naics_labels.index(cnaics_label)]
            with cc3:
                cspend = st.number_input("Annual spend ($)", min_value=0,
                                         max_value=2_000_000_000,
                                         value=int(default["spend"]),
                                         step=100_000, format="%d",
                                         key=f"cspend_{i}")
            with cc4:
                cphase = st.selectbox("Phase", ["Construction","Operations","Both"],
                                      index=["Construction","Operations","Both"].index(
                                          default["phase"]) if default["phase"] in
                                          ["Construction","Operations","Both"] else 0,
                                      key=f"cphase_{i}")

            # Validate NAICS
            try:
                cs = me.naics_to_sector(cnaics)
                st.caption(f"→ Sector [{cs}]: **{me.SECTOR_LABELS[cs]}** · "
                           f"LQ={ms['qcew']['lq'][cs]:.2f} · "
                           f"RPC={ms['rpc_sdp'][cs]:.3f}")
                valid = True
            except ValueError:
                st.error(f"NAICS '{cnaics}' not recognized.")
                valid = False

            components.append({
                "name":cname,"naics":cnaics,"spend":cspend,
                "phase":cphase,"valid":valid
            })

    st.markdown("---")
    mu_name = st.text_input("Project name (for scenario store)",
                            value=template_choice.split("(")[0].strip())

    all_valid = all(c["valid"] for c in components)
    any_spend = sum(c["spend"] for c in components) > 0

    if st.button("▶  Run Mixed-Use Analysis",
                 disabled=not(all_valid and any_spend),
                 type="primary"):
        with st.spinner("Running component analyses…"):
            component_results = []
            Y_combined = np.zeros(me.N)

            for comp in components:
                if not comp["valid"] or comp["spend"] == 0:
                    continue
                s_idx = me.naics_to_sector(comp["naics"])
                profile = ms["all_profiles"][s_idx]
                Y = np.zeros(me.N)
                ts = sum(profile.values())
                for s,sh in profile.items():
                    Y[s] += comp["spend"] * sh / ts
                Y_combined += Y
                raw = me.compute_impacts(Y, ms["inverses"], ms["emp_coeffs"])
                res_c = me.add_uncertainty(raw)
                component_results.append({
                    "component": comp,
                    "sector_idx": s_idx,
                    "results": res_c,
                    "Y": Y,
                })

            # Combined run
            raw_combined = me.compute_impacts(Y_combined, ms["inverses"], ms["emp_coeffs"])
            res_combined = me.add_uncertainty(raw_combined)

            st.session_state.mixed_use_results = {
                "components": component_results,
                "combined": res_combined,
                "project_name": mu_name,
                "total_spend": sum(c["spend"] for c in components if c["valid"]),
            }

            # Save to scenario store
            total_spend = sum(c["spend"] for c in components if c["valid"])
            scens = st.session_state.scenarios or []
            scens.append(me.make_scenario(
                mu_name + " (Mixed-Use)", "multi",
                total_spend, res_combined,
                {"components": [c["name"] for c in components]}
            ))
            st.session_state.scenarios = scens

        st.success("Mixed-use analysis complete.")

    # ── Display results ───────────────────────────────────────────────────────
    mu = st.session_state.mixed_use_results
    if mu:
        st.markdown("---")
        st.markdown(f'<div class="section-header">{mu["project_name"]} — Results</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class='unc-note'>
        ⚠️ All values shown with ±30% uncertainty bands (Watson et al. 2015).
        Total investment: {fmt(mu['total_spend'])}.
        Each component uses its own BEA production function and York County LQ.
        </div>""", unsafe_allow_html=True)

        # Component-level summary table
        st.markdown('<div class="section-header" style="margin-top:1rem">By Component</div>',
                    unsafe_allow_html=True)
        comp_rows = []
        for cr in mu["components"]:
            t = cr["results"]["total"]
            m = cr["results"]["multipliers"]
            b = t["bands"]
            comp_rows.append({
                "Component":     cr["component"]["name"],
                "NAICS":         cr["component"]["naics"],
                "Sector":        me.SECTOR_LABELS[cr["sector_idx"]],
                "Phase":         cr["component"]["phase"],
                "Spend":         fmt(cr["component"]["spend"]),
                "Total Output":  fmt(t["output"]),
                "Total Jobs":    f"{t['jobs']:,.0f}",
                "Jobs Range":    fmt_jobs_band(b["jobs_low"],b["jobs_high"]),
                "Labor Income":  fmt(t["labor_income"]),
                "Type II":       f"{m['type2']:.3f}×",
            })
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        # Combined totals
        st.markdown('<div class="section-header" style="margin-top:1rem">Combined Totals</div>',
                    unsafe_allow_html=True)
        t = mu["combined"]["total"]
        m = mu["combined"]["multipliers"]
        b = t["bands"]
        k1,k2,k3,k4,k5 = st.columns(5)
        for col,lbl,val,sub,band in [
            (k1,"Total Output",     fmt(t["output"]),
             f"{t['output']/mu['total_spend']:.2f}× ROI",
             fmt_band(b["output_low"],b["output_high"])),
            (k2,"Total Jobs",       f"{t['jobs']:,.0f}",
             f"Emp Mult {m['emp']:.2f}×",
             fmt_jobs_band(b["jobs_low"],b["jobs_high"])),
            (k3,"GDP Contribution", fmt(t["value_added"]),
             f"{t['value_added']/mu['total_spend']:.1%} of invest.",
             fmt_band(b["value_added_low"],b["value_added_high"])),
            (k4,"Labor Income",     fmt(t["labor_income"]),
             f"{t['labor_income']/t['output']:.1%} of output",
             fmt_band(b["labor_income_low"],b["labor_income_high"])),
            (k5,"Type II Mult.",    f"{m['type2']:.3f}×",
             f"Type I: {m['type1']:.3f}×",""),
        ]:
            col.markdown(metric_card(lbl,val,sub,band), unsafe_allow_html=True)

        # Component comparison chart
        if len(mu["components"]) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            fig_mu, axes_mu = plt.subplots(1,3,figsize=(15,4))
            fig_mu.patch.set_facecolor(DARK)
            for ax in axes_mu: dark_ax(ax)
            cnames = [cr["component"]["name"][:18] for cr in mu["components"]]
            cjobs  = [cr["results"]["total"]["jobs"] for cr in mu["components"]]
            couts  = [cr["results"]["total"]["output"]/1e6 for cr in mu["components"]]
            ct2    = [cr["results"]["multipliers"]["type2"] for cr in mu["components"]]
            clrs_mu = [me.SECTOR_COLORS[cr["sector_idx"] % len(me.SECTOR_COLORS)]
                       for cr in mu["components"]]
            x_mu = np.arange(len(mu["components"]))

            axes_mu[0].bar(x_mu, cjobs, color=clrs_mu, edgecolor="none", width=0.55)
            axes_mu[0].errorbar(x_mu, cjobs,
                yerr=[[v*me.UNCERTAINTY_PCT for v in cjobs],
                      [v*me.UNCERTAINTY_PCT for v in cjobs]],
                fmt="none",color="#94a3b8",capsize=5,linewidth=1.2)
            axes_mu[0].set_title("Jobs by Component (±30%)",
                                  color="#e2e8f0",fontweight="bold",fontsize=9)

            axes_mu[1].bar(x_mu, couts, color=clrs_mu, edgecolor="none", width=0.55)
            axes_mu[1].set_title("Output by Component ($M)",
                                  color="#e2e8f0",fontweight="bold",fontsize=9)

            axes_mu[2].bar(x_mu, ct2, color=clrs_mu, edgecolor="none", width=0.55)
            axes_mu[2].axhline(y=1.0,color="#f87171",linestyle="--",linewidth=1)
            axes_mu[2].set_title("Type II Multiplier by Component",
                                  color="#e2e8f0",fontweight="bold",fontsize=9)
            for i,v in enumerate(ct2):
                axes_mu[2].text(i,v+0.01,f"{v:.3f}×",ha="center",va="bottom",
                                color="#e2e8f0",fontsize=8)

            for ax in axes_mu:
                ax.set_xticks(x_mu)
                ax.set_xticklabels(cnames,rotation=20,ha="right",
                                   color="#94a3b8",fontsize=8)
            fig_mu.suptitle(mu["project_name"],color="#f1f5f9",
                            fontsize=10,fontweight="bold")
            fig_mu.tight_layout()
            st.pyplot(fig_mu, use_container_width=True); plt.close(fig_mu)

        st.markdown(f"""
        <div class='info-box'>
        <b>Methodology note:</b> Each component uses its own BEA 402-industry
        production function and York County location quotient, providing a more
        accurate representation of mixed-use projects than a single NAICS code.
        This approach aligns with IMPLAN's multi-sector treatment of complex developments.
        Results are saved to the Scenarios tab for comparison.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_res:
    res=st.session_state.results; rp=st.session_state.run_params; ms=st.session_state.model_state
    if res is None: st.info("Run an analysis first."); st.stop()

    t=res["total"]; d=res["direct"]; ii=res["indirect"]; ind=res["induced"]; m=res["multipliers"]

    st.markdown(f"""
    <div style='margin-bottom:1rem'>
      <span style='font-size:1.1rem;font-weight:700;color:#f1f5f9'>{rp['project_name']}</span>
      &nbsp;&nbsp;
      <span class='source-chip'>NAICS {rp['naics']} · {me.SECTOR_LABELS[rp['primary_s']]}</span>
      &nbsp;<span class='source-chip'>{fmt(rp['investment'])}</span>
      &nbsp;<span class='source-chip'>SDP α={ms['sdp_alpha']:.2f}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='unc-note'>
    ⚠️ <b>Uncertainty disclosure:</b> All values shown with ±{res['uncertainty_pct']*100:.0f}%
    bands. Based on Watson et al. (2015) I-O model validation literature.
    Treat point estimates as the midpoint of a plausible range.
    </div>""", unsafe_allow_html=True)

    for chk in res["mult_checks"]:
        if not chk["ok"]:
            st.markdown(f"""<div class='warn-box'>⚠️ <b>{chk['metric']}</b> =
            {chk['value']:.4f}× outside [{chk['low']:.1f}–{chk['high']:.1f}]</div>""",
            unsafe_allow_html=True)

    st.markdown('<div class="section-header">Total Economic Impact</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    tb = t["bands"]
    for col,lbl,val,sub,band in [
        (k1,"Total Output",     fmt(t["output"]),     f"{t['output']/rp['investment']:.2f}× ROI",
         fmt_band(tb["output_low"],tb["output_high"])),
        (k2,"Total Jobs",       f"{t['jobs']:,.0f}",   f"Mult {m['emp']:.2f}×",
         fmt_jobs_band(tb["jobs_low"],tb["jobs_high"])),
        (k3,"GDP Contribution", fmt(t["value_added"]),f"{t['value_added']/rp['investment']:.1%} of invest.",
         fmt_band(tb["value_added_low"],tb["value_added_high"])),
        (k4,"Labor Income",     fmt(t["labor_income"]),f"{t['labor_income']/t['output']:.1%} of output",
         fmt_band(tb["labor_income_low"],tb["labor_income_high"])),
        (k5,"Type II Mult.",    f"{m['type2']:.3f}×", f"Type I: {m['type1']:.3f}×",""),
    ]:
        col.markdown(metric_card(lbl,val,sub,band),unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Impact by Type</div>', unsafe_allow_html=True)
    impact_rows = []
    for key,lbl in [("direct","Direct"),("indirect","Indirect"),
                    ("induced","Induced"),("total","TOTAL")]:
        imp=res[key]; b=imp["bands"]
        impact_rows.append({
            "Type":lbl,"Output":fmt(imp["output"]),
            "Output Range":fmt_band(b["output_low"],b["output_high"]),
            "Jobs":f"{imp['jobs']:,.1f}",
            "Jobs Range":fmt_jobs_band(b["jobs_low"],b["jobs_high"]),
            "Labor Income":fmt(imp["labor_income"]),"Value Added":fmt(imp["value_added"]),
        })
    st.dataframe(pd.DataFrame(impact_rows),hide_index=True,use_container_width=True)

    # Charts
    st.markdown('<div class="section-header" style="margin-top:1.5rem">Visualizations</div>',
                unsafe_allow_html=True)
    fig,axes = plt.subplots(2,3,figsize=(16,9))
    fig.patch.set_facecolor(DARK)
    for ax in axes.flat: dark_ax(ax)

    ax=axes[0,0]
    v_d=d["output"]/1e6; v_i=ii["output"]/1e6; v_n=ind["output"]/1e6
    ax.bar([0],[v_d],color=CLRS["direct"],label="Direct",width=0.5)
    ax.bar([0],[v_i],color=CLRS["indirect"],label="Indirect",bottom=[v_d],width=0.5)
    ax.bar([0],[v_n],color=CLRS["induced"],label="Induced",bottom=[v_d+v_i],width=0.5)
    total_out=v_d+v_i+v_n
    ax.errorbar([0],[total_out],
                yerr=[[total_out*me.UNCERTAINTY_PCT],[total_out*me.UNCERTAINTY_PCT]],
                fmt="none",color="#f1f5f9",capsize=8,linewidth=2)
    ax.set_title("Output Stacked ($M)",color="#e2e8f0",fontweight="bold",fontsize=9)
    ax.set_ylabel("$M",color="#94a3b8",fontsize=8)
    ax.set_xticks([0]); ax.set_xticklabels(["Total"],color="#94a3b8")
    ax.legend(fontsize=7,facecolor=PANEL,labelcolor="#e2e8f0")

    ax=axes[0,1]
    jv=[d["jobs"],ii["jobs"],ind["jobs"],t["jobs"]]
    clrs2=[CLRS["direct"],CLRS["indirect"],CLRS["induced"],CLRS["total"]]
    b2=ax.bar(["Direct","Indirect","Induced","Total"],jv,color=clrs2,width=0.5)
    ax.errorbar(range(4),jv,
                yerr=[[v*me.UNCERTAINTY_PCT for v in jv],
                      [v*me.UNCERTAINTY_PCT for v in jv]],
                fmt="none",color="#94a3b8",capsize=5,linewidth=1.2)
    ax.set_title("Jobs (±30% uncertainty)",color="#e2e8f0",fontweight="bold",fontsize=9)
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,
                f"{b.get_height():.0f}",ha="center",va="bottom",color="#e2e8f0",fontsize=8)

    ax=axes[0,2]
    ml=["Type I\nOutput","Type II\nOutput","Employment","Labor\nIncome","Value\nAdded"]
    mv=[m["type1"],m["type2"],m["emp"],m["li"],m["va"]]
    mc_ok=[chk["ok"] for chk in res["mult_checks"]]+[True,True]
    b3=ax.bar(ml,mv,color=[CLRS["indirect"] if ok else "#ef4444" for ok in mc_ok],width=0.5)
    ax.axhline(y=1.0,color="#f87171",linestyle="--",linewidth=1)
    ax.set_title("Economic Multipliers",color="#e2e8f0",fontweight="bold",fontsize=9)
    for b in b3:
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,
                f"{b.get_height():.3f}×",ha="center",va="bottom",color="#e2e8f0",fontsize=8)

    ax=axes[1,0]
    ax.pie([d["output"],ii["output"],ind["output"]],
           labels=["Direct","Indirect","Induced"],
           colors=[CLRS["direct"],CLRS["indirect"],CLRS["induced"]],
           autopct="%1.1f%%",startangle=90,
           textprops={"color":"#e2e8f0","fontsize":8},
           wedgeprops={"edgecolor":DARK,"linewidth":1.5})
    ax.set_title("Output Composition",color="#e2e8f0",fontweight="bold",fontsize=9)

    ax=axes[1,1]
    sj=[(me.SECTOR_LABELS[s],
         res["direct"]["jobs_vec"][s]+res["indirect"]["jobs_vec"][s]
         +res["induced"]["jobs_vec"][s])
        for s in range(me.N)]
    sj=sorted([(l,v) for l,v in sj if v>0.05],key=lambda x:-x[1])
    if sj:
        lbls,vls=zip(*sj[:10])
        clrs5=[me.SECTOR_COLORS[me.SECTOR_LABELS.index(l) % len(me.SECTOR_COLORS)]
               for l in lbls]
        ax.barh([l[:22] for l in lbls][::-1],list(vls)[::-1],
                color=clrs5[::-1],edgecolor="none")
        ax.set_title("Top Sectors — Jobs",color="#e2e8f0",fontweight="bold",fontsize=9)
        ax.set_xlabel("Jobs",color="#94a3b8",fontsize=8)

    ax=axes[1,2]
    sdp_rpc=ms["rpc_sdp"]; slq_rpc=ms["rpc_slq"]; flq_rpc=ms["rpc_flq"]
    x6=np.arange(me.N); w=0.28
    ax.bar(x6-w,  slq_rpc, w, label="SLQ",color="#64748b",alpha=0.7)
    ax.bar(x6,    sdp_rpc, w, label=f"SDP α={ms['sdp_alpha']:.2f}",color="#3b82f6",alpha=0.9)
    ax.bar(x6+w,  flq_rpc, w, label="FLQ δ=0.25",color="#ef4444",alpha=0.7)
    ax.set_title("RPC by Method (SDP = primary)",color="#e2e8f0",fontweight="bold",fontsize=9)
    ax.set_xlabel("Sector index",color="#94a3b8",fontsize=8)
    ax.set_ylabel("RPC",color="#94a3b8",fontsize=8)
    ax.legend(fontsize=7,facecolor=PANEL,labelcolor="#e2e8f0")

    fig.suptitle(f"{rp['project_name']}  ·  {fmt(rp['investment'])}",
                 color="#f1f5f9",fontsize=11,fontweight="bold",y=1.01)
    fig.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scen:
    scens=st.session_state.scenarios or []
    if not scens:
        st.info("Run at least one analysis. Each run is saved automatically."); st.stop()
    st.markdown('<div class="section-header">Scenario Comparison</div>',unsafe_allow_html=True)
    st.caption(f"{len(scens)} scenario(s). Run more from the Run or Mixed-Use tabs to compare.")
    comp_rows=[{
        "Project":sc["name"],"NAICS":sc["naics"],"Sector":sc["sector"],
        "Investment":fmt(sc["investment"]),"Total Output":fmt(sc["output"]),
        "Total Jobs":f"{sc['jobs']:,.0f}","Jobs Range":fmt_jobs_band(sc["jobs_low"],sc["jobs_high"]),
        "Labor Income":fmt(sc["labor_income"]),"GDP (VA)":fmt(sc["value_added"]),
        "Type I":f"{sc['type1']:.3f}×","Type II":f"{sc['type2']:.3f}×","Emp Mult":f"{sc['emp_mult']:.3f}×",
    } for sc in scens]
    st.dataframe(pd.DataFrame(comp_rows),hide_index=True,use_container_width=True)
    if len(scens)>=2:
        fig_sc,axes_sc=plt.subplots(1,3,figsize=(15,5))
        fig_sc.patch.set_facecolor(DARK)
        for ax in axes_sc: dark_ax(ax)
        names=[sc["name"][:20] for sc in scens]
        jv=[sc["jobs"] for sc in scens]
        ov=[sc["output"]/1e6 for sc in scens]
        t2v=[sc["type2"] for sc in scens]
        clrs_sc=[me.SECTOR_COLORS[i%len(me.SECTOR_COLORS)] for i in range(len(scens))]
        x=np.arange(len(scens))
        axes_sc[0].bar(x,jv,color=clrs_sc,edgecolor="none",width=0.5)
        axes_sc[0].errorbar(x,jv,
            yerr=[[v*me.UNCERTAINTY_PCT for v in jv],[v*me.UNCERTAINTY_PCT for v in jv]],
            fmt="none",color="#94a3b8",capsize=6,linewidth=1.5)
        axes_sc[0].set_title("Total Jobs (±30%)",color="#e2e8f0",fontweight="bold",fontsize=10)
        axes_sc[1].bar(x,ov,color=clrs_sc,edgecolor="none",width=0.5)
        axes_sc[1].set_title("Total Output ($M)",color="#e2e8f0",fontweight="bold",fontsize=10)
        axes_sc[2].bar(x,t2v,color=clrs_sc,edgecolor="none",width=0.5)
        axes_sc[2].axhline(y=1.0,color="#f87171",linestyle="--",linewidth=1)
        axes_sc[2].set_title("Type II Multiplier",color="#e2e8f0",fontweight="bold",fontsize=10)
        for i,v in enumerate(t2v):
            axes_sc[2].text(i,v+0.01,f"{v:.3f}×",ha="center",va="bottom",color="#e2e8f0",fontsize=8)
        for ax in axes_sc:
            ax.set_xticks(x); ax.set_xticklabels(names,rotation=25,ha="right",color="#94a3b8",fontsize=8)
        fig_sc.tight_layout()
        st.pyplot(fig_sc,use_container_width=True); plt.close(fig_sc)
    if st.button("🗑️ Clear All Scenarios"):
        st.session_state.scenarios=[]; st.session_state.mixed_use_results=None; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sens:
    res=st.session_state.results; ms=st.session_state.model_state; rp=st.session_state.run_params
    if res is None or ms is None: st.info("Run an analysis first."); st.stop()

    st.markdown('<div class="section-header">SDP Alpha Sensitivity — Method Comparison</div>',
                unsafe_allow_html=True)
    st.markdown("""
    This table compares results across SDP alpha values and includes SLQ and FLQ
    reference runs. The **current model** uses the alpha set in the sidebar.
    """)

    with st.spinner("Computing sensitivity…"):
        profile=rp["profile"]
        Y=np.zeros(me.N)
        ts=sum(profile.values())
        for s,sh in profile.items(): Y[s]+=rp["investment"]*sh/ts
        sens_rows=me.sensitivity_analysis(
            Y,ms["A_nat17"],ms["qcew"],ms["national_emp"].sum(),
            ms["emp_coeffs"],ms["pce_shares"])

    current_alpha=ms["sdp_alpha"]
    disp_df=pd.DataFrame(sens_rows).copy()
    disp_df["method"]=disp_df["method"].apply(
        lambda v: v+(" ◀ current" if f"α={current_alpha:.2f}" in v else ""))
    disp_df["avg_rpc"]=disp_df["avg_rpc"].apply(lambda v:f"{v:.4f}")
    disp_df["total_jobs"]=disp_df["total_jobs"].apply(lambda v:f"{v:,.1f}")
    disp_df["total_output"]=disp_df["total_output"].apply(lambda v:fmt(v))
    disp_df["type1"]=disp_df["type1"].apply(lambda v:f"{v:.4f}×")
    disp_df["type2"]=disp_df["type2"].apply(lambda v:f"{v:.4f}×")
    disp_df["emp_mult"]=disp_df["emp_mult"].apply(lambda v:f"{v:.4f}×")
    disp_df=disp_df[["method","avg_rpc","total_jobs","total_output","type1","type2","emp_mult"]]
    disp_df.columns=["Method","Avg RPC","Total Jobs","Total Output","Type I","Type II","Emp Mult"]
    st.dataframe(disp_df,hide_index=True,use_container_width=True)

    sdp_rows=[r for r in sens_rows if "SDP" in r["method"] and r["alpha"] is not None]
    slq_row=[r for r in sens_rows if "SLQ" in r["method"]]
    flq_row=[r for r in sens_rows if "FLQ" in r["method"]]

    fig_s,axes_s=plt.subplots(1,3,figsize=(15,4))
    fig_s.patch.set_facecolor(DARK)
    for ax in axes_s: dark_ax(ax)
    alphas=[r["alpha"] for r in sdp_rows]
    for ax,(ykey,title,ylabel) in zip(axes_s,[
        ("total_jobs","Total Jobs vs SDP α","Jobs"),
        ("type2","Type II Mult vs SDP α","Multiplier"),
        ("avg_rpc","Avg RPC vs SDP α","Avg RPC"),
    ]):
        yvals=[r[ykey] for r in sdp_rows]
        ax.plot(alphas,yvals,color="#3b82f6",linewidth=2,marker="o",markersize=6,label="SDP")
        if ykey=="total_jobs":
            ax.fill_between(alphas,
                [v*(1-me.UNCERTAINTY_PCT) for v in yvals],
                [v*(1+me.UNCERTAINTY_PCT) for v in yvals],
                alpha=0.2,color="#3b82f6")
        ax.axvline(x=current_alpha,color="#f59e0b",linestyle="--",linewidth=1.5,
                   label=f"Current α={current_alpha:.2f}")
        if slq_row:
            ax.axhline(y=slq_row[0][ykey],color="#64748b",linestyle=":",linewidth=1,label="SLQ")
        if flq_row:
            ax.axhline(y=flq_row[0][ykey],color="#ef4444",linestyle=":",linewidth=1,label="FLQ")
        ax.set_title(title,color="#e2e8f0",fontweight="bold",fontsize=9)
        ax.set_xlabel("SDP Alpha (α)",color="#94a3b8",fontsize=8)
        ax.set_ylabel(ylabel,color="#94a3b8",fontsize=8)
        ax.legend(fontsize=7,facecolor=PANEL,labelcolor="#e2e8f0")

    fig_s.suptitle(f"Sensitivity — {rp['project_name']}  ·  {fmt(rp['investment'])}",
                   color="#f1f5f9",fontsize=10,fontweight="bold")
    fig_s.tight_layout()
    st.pyplot(fig_s,use_container_width=True); plt.close(fig_s)

    jobs_min=min(r["total_jobs"] for r in sdp_rows)
    jobs_max=max(r["total_jobs"] for r in sdp_rows)
    st.markdown(f"""
    <div class='info-box'>
    <b>SDP sensitivity range (α=0.10–0.30):</b>
    Jobs {jobs_min:,.0f}–{jobs_max:,.0f}
    (spread {jobs_max-jobs_min:,.0f} = {(jobs_max-jobs_min)/jobs_min*100:.0f}% of low estimate)<br>
    SLQ reference: {slq_row[0]['total_jobs']:,.1f} jobs · Type II {slq_row[0]['type2']:.3f}×<br>
    FLQ reference: {flq_row[0]['total_jobs']:,.1f} jobs · Type II {flq_row[0]['type2']:.3f}×
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SECTOR DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sec:
    res=st.session_state.results; ms=st.session_state.model_state; rp=st.session_state.run_params
    if res is None or ms is None: st.info("Run an analysis first."); st.stop()

    st.markdown(f'<div class="section-header">{me.N}-Sector Output ($)</div>',
                unsafe_allow_html=True)
    out_rows=[]
    for s in range(me.N):
        dv=res["vecs"]["direct"][s]; iv=res["vecs"]["indirect"][s]
        nv=res["vecs"]["induced"][s]; tv=dv+iv+nv
        flag=" 🔵" if s==15 else (" 🟢" if s==16 else (" ⚪" if s==14 else ""))
        out_rows.append({"Sector":me.SECTOR_LABELS[s]+flag,
                         "Direct":fmt(dv),"Indirect":fmt(iv),"Induced":fmt(nv),"Total":fmt(tv)})
    st.dataframe(pd.DataFrame(out_rows),hide_index=True,use_container_width=True)
    st.caption("🔵 Federal Gov & Defense  🟢 State & Local Gov  ⚪ Private Other Services")

    st.markdown(f'<div class="section-header" style="margin-top:1rem">{me.N}-Sector Employment (Jobs)</div>',
                unsafe_allow_html=True)
    job_rows=[]
    for s in range(me.N):
        dj=res["direct"]["jobs_vec"][s]; ij=res["indirect"]["jobs_vec"][s]; nj=res["induced"]["jobs_vec"][s]
        job_rows.append({"Sector":me.SECTOR_LABELS[s],
                         "Direct":f"{dj:.2f}","Indirect":f"{ij:.2f}",
                         "Induced":f"{nj:.2f}","Total":f"{dj+ij+nj:.2f}"})
    st.dataframe(pd.DataFrame(job_rows),hide_index=True,use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1rem">LQ & Regionalization (SDP Primary)</div>',
                unsafe_allow_html=True)
    lq_rows=[]
    for s in range(me.N):
        lq=float(ms["lq_york"][s])
        lq_rows.append({
            "Sector":me.SECTOR_LABELS[s],
            "LQ":f"{lq:.3f}",
            "RPC (SDP)":f"{ms['rpc_sdp'][s]:.4f}",
            "RPC (SLQ)":f"{ms['rpc_slq'][s]:.3f}",
            "RPC (FLQ)":f"{ms['rpc_flq'][s]:.4f}",
            "Employment":f"{ms['qcew']['emp'][s]:,.0f}",
            "A_york":f"{ms['A_york'][:,s].sum():.4f}",
            "Status":"✓ Local" if lq>=1.0 else f"↓{(1-ms['rpc_sdp'][s]):.0%} imported",
        })
    st.dataframe(pd.DataFrame(lq_rows),hide_index=True,use_container_width=True)
    st.caption(f"SDP α={ms['sdp_alpha']:.2f}  ·  "
               f"Avg RPC SDP={ms['rpc_sdp'].mean():.4f}  "
               f"SLQ={ms['rpc_slq'].mean():.4f}  "
               f"FLQ={ms['rpc_flq'].mean():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — A MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mat:
    ms=st.session_state.model_state
    if ms is None: st.info("Build the model first."); st.stop()
    view_choice=st.radio("View",["York County (SDP)","National (A_domestic)"],horizontal=True)
    A_show=ms["A_york"] if "York" in view_choice else ms["A_nat17"]
    A_df=pd.DataFrame(A_show,index=me.SECTOR_LABELS,columns=me.SECTOR_LABELS)
    st.dataframe(A_df.style.format("{:.4f}").background_gradient(cmap="Blues",axis=None),
                 use_container_width=True)
    fig_h,ax_h=plt.subplots(figsize=(12,10))
    fig_h.patch.set_facecolor(DARK); ax_h.set_facecolor(DARK)
    im=ax_h.imshow(A_show,cmap="YlOrRd",aspect="auto",vmin=0)
    ax_h.set_xticks(range(me.N)); ax_h.set_yticks(range(me.N))
    ax_h.set_xticklabels([f"{s}" for s in range(me.N)],color="#94a3b8",fontsize=8)
    ax_h.set_yticklabels([l[:25] for l in me.SECTOR_LABELS],color="#94a3b8",fontsize=8)
    fig_h.colorbar(im,ax=ax_h,label="Coefficient")
    ax_h.set_title(f"A Matrix — {'York County SDP' if 'York' in view_choice else 'National'}",
                   color="#e2e8f0",fontsize=11,fontweight="bold")
    fig_h.tight_layout()
    st.pyplot(fig_h,use_container_width=True); plt.close(fig_h)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_coef:
    ms=st.session_state.model_state
    if ms is None: st.info("Build the model first."); st.stop()
    src_map={"live":"✅ Live BEA","csv_cache":"📁 CSV Cache","hardcoded_fallback":"⚠️ 2022 Const"}
    st.markdown(f"""
    <div style='display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:1rem'>
      <span class='source-chip'>VA: {src_map.get(ms['bea_va_src'],'?')}</span>
      <span class='source-chip'>MPC: {ms['nat_mpc']:.4f}</span>
      <span class='source-chip'>Retention: {ms['regional_retention']:.2f}</span>
      <span class='source-chip'>HH share: {ms['emp_coeffs']['hh_share']:.4f}</span>
      <span class='source-chip'>SDP α: {ms['sdp_alpha']:.2f}</span>
      <span class='source-chip'>Sectors: {me.N}</span>
      <span class='source-chip'>Engine: v{me.VERSION}</span>
    </div>""", unsafe_allow_html=True)
    val_results=ms["validation"]
    n_bad = sum(1 for v in val_results if not v["ok"])
    if n_bad:
        st.warning(f"⚠️ {n_bad} coefficient(s) outside plausible ranges.")
    else:
        st.success(f"✅ All {me.N} employment coefficients within plausible ranges.")
    coeff_rows=[{
        "Sector":v["sector"],"Nat'l Empl (k)":f"{ms['national_emp'][i]:,.1f}",
        "VA $M":f"{ms['bea_va'][i]:,.0f}","Jobs/$1M VA":f"{v['jpm']:.2f}",
        "York Avg Wage":fmt(v["wage"]),"LI Share":f"{v['li']:.1%}",
        "Status":"✅" if v["ok"] else "⚠️ "+("; ".join(v["flags"])),
    } for i,v in enumerate(val_results)]
    st.dataframe(pd.DataFrame(coeff_rows),hide_index=True,use_container_width=True)

    st.markdown('<div class="section-header" style="margin-top:1rem">QCEW Employment by Sector</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class='info-box'>
    Total employment: {ms['qcew']['emp'].sum():,.0f} workers
    ({ms['qcew']['emp_ws'].sum():,.0f} wage-and-salary from QCEW +
    {ms['qcew']['emp_prop'].sum():,.0f} proprietors estimated from BEA NIPA Table 6.4)
    </div>""", unsafe_allow_html=True)
    qr=[{"Sector":me.SECTOR_LABELS[s],
         "Ownership":"Private" if s<=14 else ("Federal" if s==15 else "State/Local"),
         "W&S Emp":f"{ms['qcew']['emp_ws'][s]:,.0f}",
         "Proprietors":f"{ms['qcew']['emp_prop'][s]:,.0f}",
         "Total Emp":f"{ms['qcew']['emp'][s]:,.0f}",
         "LQ":f"{ms['qcew']['lq'][s]:.3f}",
         "RPC (SDP)":f"{ms['rpc_sdp'][s]:.4f}",
         "Avg Annual Pay":fmt(ms['qcew']['wage'][s])}
        for s in range(me.N)]
    st.dataframe(pd.DataFrame(qr),hide_index=True,use_container_width=True)
    st.caption(
        f"QCEW {ms['qcew']['year']}  ·  FIPS {ms['qcew']['fips']}  ·  "
        f"W&S from BLS QCEW payroll records  ·  "
        f"Proprietors estimated from BEA NIPA Table 6.4 national ratios  ·  "
        f"PCE vector: York County income-weighted (ACS B19001 + BLS CES 2022)"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 — DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    ms=st.session_state.model_state; res=st.session_state.results
    if ms is None: st.info("Build the model first."); st.stop()
    stab=ms["stability"]
    if stab["stable"]:
        st.markdown(f"""<div class='ok-box'>✅ <b>Leontief system stable</b><br>
        ρ=<b>{stab['spectral_radius']:.6f}</b> &lt; 1.0  ·
        Max col sum={stab['max_col_sum']:.4f}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='warn-box'>⚠️ ρ={stab['spectral_radius']:.6f} ≥ 1.0</div>""",
                    unsafe_allow_html=True)
    eigvals=np.linalg.eigvals(ms["A_york"])
    fig_e,ax_e=plt.subplots(figsize=(6,6))
    fig_e.patch.set_facecolor(DARK); ax_e.set_facecolor(PANEL)
    theta=np.linspace(0,2*np.pi,200)
    ax_e.plot(np.cos(theta),np.sin(theta),color="#ef4444",linewidth=1.5,linestyle="--")
    ax_e.scatter(eigvals.real,eigvals.imag,color="#3b82f6",s=40,zorder=5)
    ax_e.axhline(0,color="#475569",linewidth=0.5); ax_e.axvline(0,color="#475569",linewidth=0.5)
    ax_e.set_xlim(-1.5,1.5); ax_e.set_ylim(-1.5,1.5)
    ax_e.set_title(f"Eigenvalues (ρ={stab['spectral_radius']:.4f})",color="#e2e8f0",fontweight="bold")
    for sp in ax_e.spines.values(): sp.set_visible(False)
    ax_e.tick_params(colors="#94a3b8")
    fig_e.tight_layout(); st.pyplot(fig_e,use_container_width=True); plt.close(fig_e)
    st.markdown("---")
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Engine version | v{me.VERSION} |
    | Sectors | {me.N} |
    | Government split | Private / Federal / State+Local |
    | Regionalization | SDP (primary) · SLQ & FLQ (comparison) |
    | SDP Alpha | {ms['sdp_alpha']:.2f} |
    | FLQ Delta | {ms['flq_delta']:.2f} (comparison only) |
    | Avg RPC — SDP | {ms['rpc_sdp'].mean():.4f} |
    | Avg RPC — SLQ | {ms['rpc_slq'].mean():.4f} |
    | Avg RPC — FLQ | {ms['rpc_flq'].mean():.4f} |
    | Import leakage SDP | {ms['import_leakage_rate']:.1%} |
    | Proprietor employment | {ms['qcew']['emp_prop'].sum():,.0f} workers added |
    | PCE vector | York County income-weighted |
    | Uncertainty band | ±{me.UNCERTAINTY_PCT*100:.0f}% |
    """)
    if res:
        st.markdown("**Multiplier checks (last run):**")
        for chk in res["mult_checks"]:
            st.markdown(f"{'✅' if chk['ok'] else '⚠️'} **{chk['metric']}** = "
                        f"{chk['value']:.4f}× (range {chk['low']:.1f}–{chk['high']:.1f})")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_exp:
    res=st.session_state.results; ms=st.session_state.model_state; rp=st.session_state.run_params
    if res is None or ms is None: st.info("Run an analysis first."); st.stop()
    st.markdown('<div class="section-header">Download Results</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        summ_rows=[{
            "Impact Type":k.capitalize(),"Output ($)":round(res[k]["output"],2),
            "Output Low":round(res[k]["bands"]["output_low"],2),
            "Output High":round(res[k]["bands"]["output_high"],2),
            "Jobs":round(res[k]["jobs"],2),
            "Jobs Low":round(res[k]["bands"]["jobs_low"],2),
            "Jobs High":round(res[k]["bands"]["jobs_high"],2),
            "Labor Income ($)":round(res[k]["labor_income"],2),
            "Value Added ($)":round(res[k]["value_added"],2),
        } for k in ["direct","indirect","induced","total"]]
        st.download_button("📥 Impact Summary CSV",
            data=pd.DataFrame(summ_rows).to_csv(index=False).encode(),
            file_name=f"impact_summary_v{me.VERSION}.csv",mime="text/csv",
            use_container_width=True)
        sec_rows=[]
        for s in range(me.N):
            dj=res["direct"]["jobs_vec"][s]; ij=res["indirect"]["jobs_vec"][s]; nj=res["induced"]["jobs_vec"][s]
            sec_rows.append({
                "Sector":me.SECTOR_LABELS[s],"LQ":round(float(ms["lq_york"][s]),4),
                "RPC_SDP":round(float(ms["rpc_sdp"][s]),4),
                "RPC_SLQ":round(float(ms["rpc_slq"][s]),4),
                "RPC_FLQ":round(float(ms["rpc_flq"][s]),4),
                "WS_Employment":round(float(ms["qcew"]["emp_ws"][s]),0),
                "Proprietors":round(float(ms["qcew"]["emp_prop"][s]),0),
                "Total_Employment":round(float(ms["qcew"]["emp"][s]),0),
                "AvgWage":round(float(ms["qcew"]["wage"][s]),0),
                "Jobs_per_1M_VA":round(float(ms["emp_coeffs"]["jobs_per_va"][s]),4),
                "LI_Share":round(float(ms["emp_coeffs"]["li_share"][s]),4),
                "Direct_Jobs":round(dj,2),"Indirect_Jobs":round(ij,2),
                "Induced_Jobs":round(nj,2),"Total_Jobs":round(dj+ij+nj,2),
                "Direct_Output":round(float(res["vecs"]["direct"][s]),2),
                "Total_Output":round(float(res["vecs"]["total"][s]),2),
            })
        st.download_button("📥 Sector Detail CSV",
            data=pd.DataFrame(sec_rows).to_csv(index=False).encode(),
            file_name=f"impact_by_sector_v{me.VERSION}.csv",mime="text/csv",
            use_container_width=True)

        # Mixed-use export if available
        mu=st.session_state.mixed_use_results
        if mu:
            mu_rows=[]
            for cr in mu["components"]:
                t_=cr["results"]["total"]; m_=cr["results"]["multipliers"]
                mu_rows.append({
                    "Component":cr["component"]["name"],
                    "NAICS":cr["component"]["naics"],
                    "Sector":me.SECTOR_LABELS[cr["sector_idx"]],
                    "Phase":cr["component"]["phase"],
                    "Spend":cr["component"]["spend"],
                    "Total_Output":round(t_["output"],2),
                    "Total_Jobs":round(t_["jobs"],2),
                    "Labor_Income":round(t_["labor_income"],2),
                    "Value_Added":round(t_["value_added"],2),
                    "Type_I":round(m_["type1"],4),
                    "Type_II":round(m_["type2"],4),
                })
            st.download_button("📥 Mixed-Use Component CSV",
                data=pd.DataFrame(mu_rows).to_csv(index=False).encode(),
                file_name=f"mixed_use_components_v{me.VERSION}.csv",mime="text/csv",
                use_container_width=True)

    with c2:
        m_=res["multipliers"]
        mult_df=pd.DataFrame([{
            "Project":rp["project_name"],"NAICS":rp["naics"],
            "Region":f"FIPS {ms['qcew']['fips']}","Investment":rp["investment"],
            "Sectors":me.N,
            "Engine_Version":me.VERSION,
            "IO_Data":"BEA 2017 Benchmark After-Redefinitions Domestic",
            "Regionalization":f"SDP alpha={ms['sdp_alpha']:.2f}",
            "MPC":round(ms["nat_mpc"],4),"Regional_Retention":ms["regional_retention"],
            "Proprietors_Added":int(ms["qcew"]["emp_prop"].sum()),
            "PCE_Vector":"York County income-weighted (ACS B19001 + BLS CES 2022)",
            "Uncertainty_Pct":me.UNCERTAINTY_PCT,
            "Type_I":round(m_["type1"],4),"Type_II":round(m_["type2"],4),
            "Emp_Mult":round(m_["emp"],4),
            "Spectral_Radius":round(ms["stability"]["spectral_radius"],6),
        }])
        st.download_button("📥 Multipliers & Metadata CSV",
            data=mult_df.to_csv(index=False).encode(),
            file_name=f"multipliers_v{me.VERSION}.csv",mime="text/csv",
            use_container_width=True)
        if st.session_state.scenarios:
            scen_rows=[{
                "Name":sc["name"],"NAICS":sc["naics"],"Sector":sc["sector"],
                "Investment":sc["investment"],"Total_Output":round(sc["output"],2),
                "Total_Jobs":round(sc["jobs"],2),"Jobs_Low":round(sc["jobs_low"],2),
                "Jobs_High":round(sc["jobs_high"],2),"Labor_Income":round(sc["labor_income"],2),
                "Value_Added":round(sc["value_added"],2),"Type_I":round(sc["type1"],4),
                "Type_II":round(sc["type2"],4),"Emp_Mult":round(sc["emp_mult"],4),
            } for sc in st.session_state.scenarios]
            st.download_button("📥 Scenario Comparison CSV",
                data=pd.DataFrame(scen_rows).to_csv(index=False).encode(),
                file_name=f"scenario_comparison_v{me.VERSION}.csv",mime="text/csv",
                use_container_width=True)
        A_df2=pd.DataFrame(ms["A_york"],index=me.SECTOR_LABELS,columns=me.SECTOR_LABELS)
        st.download_button("📥 A_york Matrix CSV",
            data=A_df2.to_csv().encode(),
            file_name=f"A_york_SDP_{me.N}sector.csv",mime="text/csv",
            use_container_width=True)
    st.markdown("---")
    st.markdown(f"""
**v{me.VERSION}:** {me.N}-sector · SDP regionalization α={ms['sdp_alpha']:.2f} ·
MPC={ms['nat_mpc']:.4f} · Retention={ms['regional_retention']:.2f} ·
Proprietors: {int(ms['qcew']['emp_prop'].sum()):,} added ·
PCE: York County income-weighted ·
Uncertainty ±{me.UNCERTAINTY_PCT*100:.0f}% · ρ={ms['stability']['spectral_radius']:.4f}
    """)
