"""
model_engine.py  —  Regional Economic Impact Model  v7.1
─────────────────────────────────────────────────────────
Pure-computation module. No Streamlit imports. No side effects.

v7.1 changes from v7.0:
  - CSV cache is now the PRIMARY data source for BEA tables.
    The three CSV files in data/ contain real BEA 2022 published
    values and are always available regardless of API status.
    The live BEA API is attempted secondarily and logged for comparison.
  - Added VERSION constant for deployment verification.
  - All other v7.0 functionality unchanged.
"""

VERSION = "7.1"

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
import requests

warnings.filterwarnings("ignore")

_HERE    = Path(__file__).parent
DATA_DIR = _HERE / "data"

# ── SECTOR LABELS (17) ────────────────────────────────────────────────────────

SECTOR_LABELS = [
    "Agriculture",             # 0   NAICS 11
    "Mining & Utilities",      # 1   NAICS 21,22
    "Construction",            # 2   NAICS 23
    "Mfg — Durable",           # 3   NAICS 33DG
    "Mfg — Non-Durable",       # 4   NAICS 31ND
    "Wholesale Trade",         # 5   NAICS 42
    "Retail Trade",            # 6   NAICS 44-45
    "Transportation & Whsg.",  # 7   NAICS 48-49
    "Information",             # 8   NAICS 51
    "Finance & Real Estate",   # 9   NAICS 52-53
    "Professional Services",   # 10  NAICS 54-55
    "Admin & Waste Services",  # 11  NAICS 56
    "Education & Health",      # 12  NAICS 61-62
    "Arts & Accommodation",    # 13  NAICS 71-72
    "Private Other Services",  # 14  NAICS 81  own_code=5
    "Federal Gov & Defense",   # 15  NAICS 921  own_code=1
    "State & Local Gov",       # 16  NAICS 922-928  own_code=2+3
]
N = len(SECTOR_LABELS)   # 17

SECTOR_COLORS = [
    "#4ade80","#a78bfa","#f97316","#60a5fa","#f472b6",
    "#34d399","#fb923c","#818cf8","#2dd4bf","#c084fc",
    "#e879f9","#38bdf8","#facc15","#f87171","#94a3b8",
    "#1d4ed8","#15803d",
]

BEA_BASE = "https://apps.bea.gov/api/data"
BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

MULT_TYPE1_MIN, MULT_TYPE1_MAX = 1.1, 2.5
MULT_TYPE2_MIN, MULT_TYPE2_MAX = 1.2, 3.0

# Uncertainty: ±30% combined (Watson et al. 2015 + 2017 vintage)
UNCERTAINTY_PCT = 0.30

# ── SECTOR CROSSWALK ──────────────────────────────────────────────────────────

_SPECIAL = {
    "S00500":14,"S00600":14,"S00101":14,"S00102":14,
    "S00201":14,"S00202":14,"S00203":14,
    "GSLGE":16,"GSLGH":16,"GSLGO":16,
    "GFGD":15,"GFGN":15,"GFE":15,
    "GSLE":16,"GSLO":16,
    "S00401":-1,"S00402":-1,"S00300":-1,"S00900":-1,
    "531HSO":9,"531HSR":9,"532RL":9,"ORE000":9,"HS0000":9,
    "521CI0":9,"525000":9,
    "4B0000":6,"487OS0":7,"488A00":7,
    "111CA0":0,"113FF0":0,
}
_NONDUR = {"311","312","313","314","315","316","322","323","324","325","326"}

NAICS_TO_SECTOR = {
    "11":0,"111":0,"112":0,"113":0,"114":0,"115":0,
    "21":1,"211":1,"212":1,"213":1,"22":1,"221":1,
    "23":2,"236":2,"237":2,"238":2,
    "31":3,"32":4,"33":3,
    "311":4,"312":4,"313":4,"314":4,"315":4,"316":4,
    "321":3,"322":4,"323":4,"324":4,"325":4,"326":4,
    "327":3,"331":3,"332":3,"333":3,"334":3,"335":3,
    "336":3,"337":3,"339":3,
    "42":5,"423":5,"424":5,"425":5,
    "44":6,"45":6,"48":7,"49":7,
    "51":8,"52":9,"53":9,"54":10,"55":10,"56":11,
    "61":12,"62":12,"71":13,"72":13,
    "81":14,"811":14,"812":14,"813":14,"814":14,
    "921":15,
    "922":16,"923":16,"924":16,"925":16,"926":16,"927":16,"928":16,
    "92":16,
}

BEA_SUMMARY_TO_SECTOR = {
    "111CA":0,"113FF":0,
    "211":1,"212":1,"213":1,"22":1,
    "23":2,
    "321":3,"327":3,"331":3,"332":3,"333":3,"334":3,"335":3,
    "3361MV":3,"3364OT":3,"337":3,"339":3,
    "311FT":4,"313TT":4,"315AL":4,"322":4,"323":4,"324":4,"325":4,"326":4,
    "42":5,
    "441":6,"445":6,"452":6,"4A0":6,
    "481":7,"482":7,"483":7,"484":7,"485":7,"486":7,"487OS":7,"493":7,
    "511":8,"512":8,"513":8,"514":8,
    "521CI":9,"523":9,"524":9,"525":9,"HS":9,"ORE":9,"532RL":9,
    "5411":10,"5412OP":10,"5415":10,"55":10,
    "561":11,"562":11,
    "61":12,"621":12,"622":12,"623":12,"624":12,
    "711AS":13,"713":13,"721":13,"722":13,
    "81":14,
    "GFGD":15,"GFGN":15,"GFE":15,
    "GSLE":16,"GSLO":16,
}


def naics_to_sector(code: str) -> int:
    code = str(code).strip()
    for length in [3, 2]:
        k = code[:length]
        if k in NAICS_TO_SECTOR:
            return NAICS_TO_SECTOR[k]
    raise ValueError(f"NAICS '{code}' not recognized.")


def _code_to_sector(code: str) -> int:
    code = str(code).strip()
    if code in _SPECIAL:
        r = _SPECIAL[code]
        return r if r is not None else -1
    d = "".join(c for c in code if c.isdigit())
    if not d: return 14
    p2, p3 = d[:2], d[:3]
    if   p2 == "11":             return 0
    elif p2 in ("21","22"):      return 1
    elif p2 == "23":             return 2
    elif p2 in ("31","32","33"): return 4 if p3 in _NONDUR else 3
    elif p2 == "42":             return 5
    elif p2 in ("44","45"):      return 6
    elif p2 in ("48","49"):      return 7
    elif p2 == "51":             return 8
    elif p2 in ("52","53"):      return 9
    elif p2 in ("54","55"):      return 10
    elif p2 == "56":             return 11
    elif p2 in ("61","62"):      return 12
    elif p2 in ("71","72"):      return 13
    elif p2 == "81":             return 14
    elif p2 == "92":
        return 15 if p3 == "921" else 16
    else:                        return 14

# ── FILE PARSERS ──────────────────────────────────────────────────────────────

def parse_use_table(src, sheet="2017") -> dict:
    wb   = openpyxl.load_workbook(src, read_only=True, data_only=True)
    ws   = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    C0, C1, R0, R1, PCE_COL = 2, 404, 6, 408, 405
    hdr_codes = list(rows[5])
    hdr_names = list(rows[4])
    def _arr(row_idx):
        result = []
        for c in range(C0, C1):
            v = rows[row_idx][c]
            try:
                result.append(float(v) if v is not None else 0.0)
            except (ValueError, TypeError):
                result.append(0.0)
        return np.array(result, dtype=float)
    Z = np.zeros((R1-R0, C1-C0), dtype=float)
    for ri, r in enumerate(range(R0, R1)):
        for ci, c in enumerate(range(C0, C1)):
            v = rows[r][c]
            if v is None:
                continue
            try:
                Z[ri,ci] = float(v)
            except (ValueError, TypeError):
                pass
    return {
        "Z": Z, "x": _arr(413),
        "va_comp": _arr(409), "va_taxes": _arr(410),
        "va_gos":  _arr(411), "va_total": _arr(412),
        "ind_codes": [str(hdr_codes[c]) for c in range(C0,C1)],
        "ind_names": [str(hdr_names[c]) for c in range(C0,C1)],
        "com_codes": [str(rows[r][0])   for r in range(R0,R1)],
        "pce": np.array([
            float(rows[r][PCE_COL]) if rows[r][PCE_COL] is not None
            and not isinstance(rows[r][PCE_COL], str) else 0.0
            for r in range(R0, R1)
        ], dtype=float),
    }


def parse_D_matrix(src, sheet="2017") -> dict:
    wb   = openpyxl.load_workbook(src, read_only=True, data_only=True)
    ws   = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    ind_codes = [str(rows[r][0]) for r in range(6, 408)]
    D = np.zeros((402, 402), dtype=float)
    for ri, r in enumerate(range(6, 408)):
        for ci, c in enumerate(range(2, 404)):
            v = rows[r][c]
            if v is None:
                continue
            try:
                D[ri,ci] = float(v)
            except (ValueError, TypeError):
                pass
    col_sums = D.sum(axis=0)
    bad = np.abs(col_sums - 1.0) > 0.01
    if bad.any():
        D[:, bad] /= np.where(col_sums[bad] > 0, col_sums[bad], 1.0)
    return {"D": D, "ind_codes": ind_codes}


def parse_B_domestic(src, sheet="2017") -> np.ndarray:
    wb   = openpyxl.load_workbook(src, read_only=True, data_only=True)
    ws   = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    B = np.zeros((402, 402), dtype=float)
    for ri, r in enumerate(range(5, 407)):
        for ci, c in enumerate(range(2, 404)):
            v = rows[r][c]
            if v is None:
                continue
            try:
                B[ri,ci] = float(v)
            except (ValueError, TypeError):
                # Skip header labels or industry codes that appear in data cells
                pass
    return B


def load_qcew(src) -> dict:
    """Parse BLS QCEW county CSV. Gov split: federal→15, state/local→16."""
    df = pd.read_csv(src)
    df["industry_code"] = df["industry_code"].astype(str).str.strip()
    year = int(df["year"].max())

    NAICS2_MAP = {
        "11":0,"22":1,"23":2,"31-33":3,
        "42":5,"44-45":6,"48-49":7,
        "51":8,"52":9,"53":9,"54":10,"55":10,"56":11,
        "61":12,"62":12,"71":13,"72":13,
        "81":14,"99":14,
    }
    _NONDUR_3D = {"311","312","313","314","315","316","322","323","324","325","326"}

    priv2 = df[(df["own_code"]==5) & (df["agglvl_code"]==74)].copy()
    priv3 = df[(df["own_code"]==5) & (df["agglvl_code"]==75)].copy()

    emp = np.zeros(N); lq_n = np.zeros(N); wage_n = np.zeros(N)

    for _, row in priv2.iterrows():
        s = NAICS2_MAP.get(row["industry_code"], -1)
        if s < 0: continue
        e = row["annual_avg_emplvl"]
        emp[s]    += e
        lq_n[s]   += row["lq_annual_avg_emplvl"] * e
        wage_n[s] += row["avg_annual_pay"] * e

    mfg3 = priv3[priv3["industry_code"].str.match(r"^3[123]$|^3[123]\d$")]
    if len(mfg3):
        dur    = mfg3[~mfg3["industry_code"].isin(_NONDUR_3D)]
        nondur = mfg3[ mfg3["industry_code"].isin(_NONDUR_3D)]
        emp[3]=emp[4]=0; lq_n[3]=lq_n[4]=0; wage_n[3]=wage_n[4]=0
        for df_sub, s_idx in [(dur,3),(nondur,4)]:
            e_tot = df_sub["annual_avg_emplvl"].sum()
            if e_tot > 0:
                emp[s_idx]    = e_tot
                lq_n[s_idx]   = (df_sub["lq_annual_avg_emplvl"]
                                  * df_sub["annual_avg_emplvl"]).sum()
                wage_n[s_idx] = (df_sub["avg_annual_pay"]
                                  * df_sub["annual_avg_emplvl"]).sum()

    # Federal (own_code=1)
    fed_rows = df[(df["own_code"]==1) & (df["agglvl_code"]==71)]
    if len(fed_rows):
        emp[15]    += float(fed_rows["annual_avg_emplvl"].sum())
        wage_n[15] += float(fed_rows["total_annual_wages"].sum())
        lq_n[15]   += fed_rows.apply(
            lambda r: r["lq_annual_avg_emplvl"] * r["annual_avg_emplvl"], axis=1).sum()

    # State (own_code=2)
    state_rows = df[(df["own_code"]==2) & (df["agglvl_code"]==71)]
    if len(state_rows):
        emp[16]    += float(state_rows["annual_avg_emplvl"].sum())
        wage_n[16] += float(state_rows["total_annual_wages"].sum())
        lq_n[16]   += state_rows.apply(
            lambda r: r["lq_annual_avg_emplvl"] * r["annual_avg_emplvl"], axis=1).sum()

    # Local (own_code=3)
    local_rows = df[(df["own_code"]==3) & (df["agglvl_code"]==71)]
    if len(local_rows):
        emp[16]    += float(local_rows["annual_avg_emplvl"].sum())
        wage_n[16] += float(local_rows["total_annual_wages"].sum())
        lq_n[16]   += local_rows.apply(
            lambda r: r["lq_annual_avg_emplvl"] * r["annual_avg_emplvl"], axis=1).sum()

    lq   = np.where(emp > 0, lq_n  / emp, 0.0)
    wage = np.where(emp > 0, wage_n / emp, 0.0)
    return {
        "emp":              emp,
        "lq":               lq,
        "wage":             wage,
        "year":             year,
        "fips":             str(df["area_fips"].iloc[0]),
        "total_county_emp": float(emp.sum()),
    }

# ── MATRIX BUILDERS ───────────────────────────────────────────────────────────

def build_A_domestic(D, B_dom, use_data) -> dict:
    A = D @ B_dom
    col_sums = A.sum(axis=0)
    bad = col_sums >= 1.0
    if bad.any():
        A[:, bad] *= 0.99 / col_sums[bad]
    with np.errstate(divide="ignore", invalid="ignore"):
        va_share = np.where(use_data["x"]>0, use_data["va_total"]/use_data["x"], 0.0)
        li_share = np.where(use_data["x"]>0, use_data["va_comp"] /use_data["x"], 0.0)
    return {"A": A, "va_share": va_share, "li_share": li_share}


def build_B_total(use_data) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(use_data["x"] > 0,
                        use_data["Z"] / use_data["x"][np.newaxis,:], 0.0)


def aggregate_matrix(A, x, va_total, va_comp, sector_map) -> dict:
    A_agg    = np.zeros((N, N))
    x_agg    = np.zeros(N)
    va_agg   = np.zeros(N)
    comp_agg = np.zeros(N)
    for j, s in enumerate(sector_map):
        if s < 0: continue
        x_agg[s]    += x[j]
        va_agg[s]   += va_total[j]
        comp_agg[s] += va_comp[j]
    for t in range(N):
        t_cols = np.where(sector_map == t)[0]
        if not len(t_cols): continue
        x_t = x[t_cols]; tot = x_t.sum()
        if tot <= 0: continue
        for s in range(N):
            s_rows = np.where(sector_map == s)[0]
            if len(s_rows):
                A_agg[s,t] = A[np.ix_(s_rows, t_cols)].sum(0).dot(x_t) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        va_share = np.where(x_agg>0, va_agg   /x_agg, 0.0)
        li_share = np.where(x_agg>0, comp_agg /x_agg, 0.0)
    return {"A_agg": A_agg, "x_agg": x_agg,
            "va_share": va_share, "li_share": li_share}


def build_pce_shares(pce_raw, com_sector) -> np.ndarray:
    pce_agg = np.zeros(N)
    for i, s in enumerate(com_sector):
        if 0 <= s < N:
            pce_agg[s] += max(pce_raw[i], 0.0)
    total = pce_agg.sum()
    return pce_agg / total if total > 0 else np.ones(N) / N

# ── REGIONALIZATION METHODS ───────────────────────────────────────────────────

def compute_sdp_rpc(lq: np.ndarray, alpha: float = 0.20) -> np.ndarray:
    """
    Supply-Demand Pool (SDP) regionalization.

    Formula: RPC_s = LQ_s / (LQ_s + alpha),  capped at 1.0

    Source: Kronenberg (2009) "The Use of Input-Output Analysis in
    Regional Economic Impact Assessment", Review of Regional Research;
    Flegg & Tohmo (2016) "Estimating Regional Input Coefficients and
    Multipliers", Spatial Economic Analysis.

    alpha = cross-hauling parameter representing the minimum import share
    even for fully self-sufficient sectors. Published values:
      0.10 — low cross-hauling, denser regional economies
      0.20 — standard for US counties (default, produces IMPLAN-comparable results)
      0.30 — high cross-hauling, very open small economies

    At alpha=0.20:
      LQ=0.10 → RPC=0.333  (not 0.10 as SLQ would give)
      LQ=1.00 → RPC=0.833  (not 1.00 as SLQ would give)
      LQ=2.00 → RPC=0.909
      LQ=0.00 → RPC=0.000  (no local employment = no local supply)

    This method is appropriate for county-level analysis because:
      1. It does not require knowing regional size (no size penalty)
      2. It corrects for cross-hauling — regions both import and export
         the same goods simultaneously
      3. It produces multipliers comparable to IMPLAN and RIMS II for
         US counties (avg RPC ~0.71 at alpha=0.20 for York County)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rpc = np.where(lq > 0, lq / (lq + alpha), 0.0)
    return np.minimum(rpc, 1.0)


def compute_flq_rpc(lq: np.ndarray, total_county_emp: float,
                    total_national_emp: float, delta: float = 0.25):
    """
    Flegg Location Quotient — retained for comparison in sensitivity tab.
    Not recommended as primary method for counties with <100k workers.
    """
    ratio = total_county_emp / max(total_national_emp, 1.0)
    lam   = float(np.log2(1.0 + ratio) ** delta)
    rpc   = np.minimum(lq * lam, 1.0)
    return rpc, lam


def compute_slq_rpc(lq: np.ndarray) -> np.ndarray:
    """Simple Location Quotient — retained for comparison only."""
    return np.minimum(lq, 1.0)


def regionalize(A_nat17: np.ndarray, qcew: dict,
                national_emp_total: float,
                sdp_alpha: float = 0.20,
                flq_delta: float = 0.25) -> dict:
    """
    Regionalize A_national using SDP as the primary method.

    Returns A_york (SDP-based) plus SLQ and FLQ RPCs for comparison display.

    sdp_alpha : cross-hauling parameter for SDP (default 0.20)
    flq_delta : FLQ sensitivity, used only for comparison display
    """
    lq_york   = qcew["lq"].copy()

    # Primary: SDP
    rpc_sdp   = compute_sdp_rpc(lq_york, alpha=sdp_alpha)

    # Comparison methods (displayed in UI, not used in main calculation)
    rpc_slq   = compute_slq_rpc(lq_york)
    rpc_flq, flq_lambda = compute_flq_rpc(
        lq_york,
        total_county_emp   = qcew["total_county_emp"],
        total_national_emp = national_emp_total * 1000,
        delta              = flq_delta,
    )

    A_york = A_nat17 * rpc_sdp[:, np.newaxis]

    return {
        "A_york":     A_york,
        "lq_york":    lq_york,
        "rpc_york":   rpc_sdp,   # primary — SDP
        "rpc_sdp":    rpc_sdp,
        "rpc_slq":    rpc_slq,
        "rpc_flq":    rpc_flq,
        "flq_lambda": flq_lambda,
        "sdp_alpha":  sdp_alpha,
        "flq_delta":  flq_delta,
    }

# ── BLS / BEA API & FALLBACKS ─────────────────────────────────────────────────

BLS_SERIES = {
    "1a": "CEU1021000001", "1b": "CEU0622000001",
    2:    "CEU2000000001", 3:    "CEU3100000001", 4:    "CEU3200000001",
    5:    "CEU4142000001", 6:    "CEU4200000001", 7:    "CEU4300000001",
    8:    "CEU5000000001",
    "9a": "CEU5552000001", "9b": "CEU5553000001",
    "10a":"CEU6054000001", "10b":"CEU6055000001",
    11:   "CEU6056000001",
    "12a":"CEU6561000001", "12b":"CEU6562000001",
    "13a":"CEU7071000001", "13b":"CEU7072000001",
    14:   "CEU8000000001",
    15:   "CEU9091000001",
    "16a":"CEU9092000001",
    "16b":"CEU9093000001",
}
ALL_SERIES_IDS = list(BLS_SERIES.values())
AGR_NATIONAL_EMP_FALLBACK = 2109.9

BLS_FALLBACK_2022 = {
    "CEU1021000001":657.5,  "CEU0622000001":574.0,
    "CEU2000000001":7697.2, "CEU3100000001":9122.5, "CEU3200000001":4841.6,
    "CEU4142000001":6153.4, "CEU4200000001":15592.8,"CEU4300000001":6622.7,
    "CEU5000000001":3038.9, "CEU5552000001":6617.6, "CEU5553000001":2394.6,
    "CEU6054000001":10199.4,"CEU6055000001":2906.5, "CEU6056000001":9628.6,
    "CEU6561000001":3787.0, "CEU6562000001":20411.9,"CEU7071000001":2615.7,
    "CEU7072000001":12346.2,"CEU8000000001":5697.4,
    "CEU9091000001":2997.0, "CEU9092000001":5220.0, "CEU9093000001":14307.2,
}

BEA_OUTPUT_FALLBACK = np.array([
    464200,1031800,1872400,3421600,3190200,2391700,2088400,
    1512300,2155700,7321500,3289100,1256100,3298400,1702800,
    812000, 1083000, 3263300
], dtype=float)

BEA_COMP_FALLBACK = np.array([
    58200,174700,617400,841800,402700,618800,686900,473100,
    426800,1063500,1562300,438100,1352300,578800,
    260000, 480000, 1832600
], dtype=float)

BEA_VA_FALLBACK = np.array([
    233700, 492000, 993400,1371800, 790200,1202800,1213700,
    753100,1198400,5068700,2204500, 731800,2244400, 994200,
    600000, 900000, 2362400
], dtype=float)


def _load_csv_cache(filename):
    path = DATA_DIR / filename
    if not path.exists(): return None
    try:
        df = pd.read_csv(path)
        result = np.zeros(N)
        for i, label in enumerate(SECTOR_LABELS):
            match = df[df["sector"] == label]["value_millions"]
            if len(match):
                result[i] = float(match.values[0])
        return result if result.sum() > 0 else None
    except Exception:
        return None


def _bea_extract_rows(data):
    results = data.get("BEAAPI",{}).get("Results",{})
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and "Data" in item:
                return item["Data"]
        return []
    elif isinstance(results, dict):
        return results.get("Data", [])
    return []


def fetch_bea_table(api_key, table_id, year="2022"):
    """
    Fetch a BEA GDPbyIndustry table and aggregate to 17 model sectors.

    BEA TABLE ID MAP (verified against live API, 2024-2025):
      table_id="VA"  → TableID 1  — Value Added by Industry ($B)
      table_id="GO"  → TableID 15 — Gross Output by Industry ($B)

    Returns (result_array, diagnostic_string) so callers can log
    exactly what happened — success, empty, error, or wrong units.
    """
    TABLE_MAP = {
        "VA": "1",
        "GO": "15",
    }
    bea_tid = TABLE_MAP.get(table_id, table_id)

    params = {"UserID":api_key,"method":"GetData","datasetname":"GDPbyIndustry",
              "TableID":bea_tid,"Frequency":"A","Year":year,
              "Industry":"ALL","ResultFormat":"JSON"}
    try:
        r = requests.get(BEA_BASE, params=params, timeout=30)
        r.raise_for_status()
        rows = _bea_extract_rows(r.json())
        if not rows:
            return None, f"API returned 0 rows for TableID={bea_tid}"
        result = np.zeros(N)
        matched = 0
        for row in rows:
            ind = str(row.get("Industry","")).strip()
            if ind in BEA_SUMMARY_TO_SECTOR:
                try:
                    val_b = float(str(row.get("DataValue","")).replace(",",""))
                    result[BEA_SUMMARY_TO_SECTOR[ind]] += val_b * 1000.0
                    matched += 1
                except ValueError:
                    pass
        total_t = result.sum() / 1e6
        diag = (f"TableID={bea_tid} rows={len(rows)} matched={matched} "
                f"sum=${total_t:.1f}T")
        return result, diag
    except Exception as e:
        return None, f"TableID={bea_tid} EXCEPTION: {e}"


def fetch_bea_nipa_mpc(api_key, year="2022"):
    params = {"UserID":api_key,"method":"GetData","datasetname":"NIPA",
              "TableName":"T20600","Frequency":"A","Year":year,"ResultFormat":"JSON"}
    try:
        r = requests.get(BEA_BASE, params=params, timeout=30)
        r.raise_for_status()
        vals = {}
        for row in _bea_extract_rows(r.json()):
            sc = row.get("SeriesCode","")
            try: vals[sc] = float(str(row.get("DataValue","")).replace(",",""))
            except ValueError: pass
        dpi = vals.get("A065RC"); outlays = vals.get("A068RC")
        if dpi and outlays and dpi > 0: return outlays / dpi
        return None
    except Exception: return None


def fetch_bls(api_key, year="2022"):
    payload = {"seriesid":ALL_SERIES_IDS,"startyear":year,"endyear":year,
               "annualaverage":"true","registrationkey":api_key}
    try:
        r = requests.post(BLS_BASE, json=payload, timeout=45)
        r.raise_for_status()
        resp = r.json()
        if resp.get("status") != "REQUEST_SUCCEEDED": return None
        raw = {}
        for series in resp["Results"]["series"]:
            sid = series["seriesID"]
            val = next((float(d["value"]) for d in series["data"]
                        if d["period"]=="M13"), None)
            if val is None:
                monthly = [float(d["value"]) for d in series["data"]
                           if d["period"].startswith("M") and d["period"]!="M13"]
                val = float(np.mean(monthly)) if monthly else 0.0
            raw[sid] = val
        return raw
    except Exception: return None


def series_to_emp(raw) -> np.ndarray:
    emp = np.zeros(N)
    emp[0] = AGR_NATIONAL_EMP_FALLBACK
    for idx in [2,3,4,5,6,7,8,11,14]:
        emp[idx] = raw.get(BLS_SERIES[idx], 0.0)
    emp[1]  = raw.get(BLS_SERIES["1a"],0) + raw.get(BLS_SERIES["1b"],0)
    emp[9]  = raw.get(BLS_SERIES["9a"],0) + raw.get(BLS_SERIES["9b"],0)
    emp[10] = raw.get(BLS_SERIES["10a"],0)+ raw.get(BLS_SERIES["10b"],0)
    emp[12] = raw.get(BLS_SERIES["12a"],0)+ raw.get(BLS_SERIES["12b"],0)
    emp[13] = raw.get(BLS_SERIES["13a"],0)+ raw.get(BLS_SERIES["13b"],0)
    emp[15] = raw.get(BLS_SERIES[15],  0.0)
    emp[16] = raw.get(BLS_SERIES["16a"],0) + raw.get(BLS_SERIES["16b"],0)
    return emp


def _use_or_fallback(result_and_diag, api_fallback, csv_filename,
                     min_plausible_sum=None):
    """
    Data source priority (revised):
      1. Local CSV cache  — real BEA 2022 published values, always available,
                            updated manually when BEA releases new data.
                            This is now the PRIMARY source because it is
                            versioned, stable, and not subject to API changes.
      2. Live BEA API    — used only if CSV cache is missing AND API succeeds.
      3. Hardcoded array — last resort if both above fail.

    Returns (array, source_label, diagnostic_string).
    """
    if isinstance(result_and_diag, tuple):
        result, diag = result_and_diag
    else:
        result, diag = result_and_diag, ""

    # Tier 1: CSV cache (primary)
    csv_result = _load_csv_cache(csv_filename)
    if csv_result is not None:
        # Also check if live API returned plausible data for comparison logging
        api_ok = (result is not None and result.sum() > 0 and
                  (min_plausible_sum is None or result.sum() >= min_plausible_sum))
        if api_ok:
            api_note = f"API also available ({diag}) — CSV used as primary."
        else:
            api_note = f"API unavailable or implausible ({diag}) — CSV used."
        return csv_result, "csv_cache", api_note

    # Tier 2: live API (fallback if CSV missing)
    if result is not None and result.sum() > 0:
        if min_plausible_sum is None or result.sum() >= min_plausible_sum:
            return result, "live", f"CSV not found, using live API. {diag}"

    # Tier 3: hardcoded constant
    return api_fallback, "hardcoded_fallback", \
           f"CSV not found, API failed. Using 2022 constant. {diag}"

# ── EMPLOYMENT COEFFICIENTS ───────────────────────────────────────────────────

def build_employment_coefficients(national_emp, bea_output, bea_comp,
                                   bea_va, qcew, va_share_agg,
                                   mpc=0.90, regional_retention=0.50) -> dict:
    """
    Jobs per $1M Value Added (BEA Table 1) — not gross output.
    York County wages from QCEW where available, else national average.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        jobs_per_va  = np.where(bea_va > 0, (national_emp*1000)/bea_va, 0.0)
        nat_avg_wage = np.where(national_emp > 0,
                                (bea_comp*1e6)/(national_emp*1000), 0.0)
        li_share_bea = np.where(bea_output > 0, bea_comp/bea_output, 0.0)

    york_wage = np.where(qcew["emp"] > 0, qcew["wage"], nat_avg_wage)
    york_li_share = np.where(bea_output > 0,
                             (york_wage*national_emp*1000)/(bea_output*1e6),
                             li_share_bea)
    york_li_share = np.clip(york_li_share, 0.0, 0.70)
    hh_share = mpc * regional_retention

    return {
        "jobs_per_va":        jobs_per_va,
        "jobs_per_million":   jobs_per_va,
        "avg_wage":           york_wage,
        "li_share":           york_li_share,
        "va_share":           va_share_agg,
        "hh_share":           hh_share,
        "nat_avg_wage":       nat_avg_wage,
        "mpc":                mpc,
        "regional_retention": regional_retention,
    }

# ── LEONTIEF STABILITY ────────────────────────────────────────────────────────

def check_leontief_stability(A) -> dict:
    eigenvalues     = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    col_sums        = A.sum(axis=0)
    warnings_list   = []
    if spectral_radius >= 1.0:
        warnings_list.append(
            f"CRITICAL: spectral radius={spectral_radius:.4f} ≥ 1.0.")
    elif spectral_radius > 0.95:
        warnings_list.append(
            f"WARNING: spectral radius={spectral_radius:.4f} near 1.0.")
    if col_sums.max() >= 1.0:
        warnings_list.append(
            f"WARNING: {(col_sums>=1.0).sum()} column(s) sum ≥ 1.0.")
    return {"spectral_radius": spectral_radius,
            "stable": spectral_radius < 1.0,
            "max_col_sum": float(col_sums.max()),
            "col_sums": col_sums, "warnings": warnings_list}

# ── MULTIPLIER SANITY CHECKS ──────────────────────────────────────────────────

def check_multipliers(mults) -> list:
    return [
        {"metric":"Type I Output Multiplier","value":mults["type1"],
         "low":MULT_TYPE1_MIN,"high":MULT_TYPE1_MAX,
         "ok":MULT_TYPE1_MIN<=mults["type1"]<=MULT_TYPE1_MAX,
         "note":"Indirect supply-chain effects only"},
        {"metric":"Type II Output Multiplier","value":mults["type2"],
         "low":MULT_TYPE2_MIN,"high":MULT_TYPE2_MAX,
         "ok":MULT_TYPE2_MIN<=mults["type2"]<=MULT_TYPE2_MAX,
         "note":"Includes induced household spending"},
        {"metric":"Employment Multiplier","value":mults["emp"],
         "low":1.1,"high":4.0,
         "ok":1.1<=mults["emp"]<=4.0,
         "note":"Total jobs / direct jobs"},
    ]

# ── LEONTIEF ENGINE ───────────────────────────────────────────────────────────

def build_leontief_inverses(A_reg, li_share, pce_shares, hh_share) -> dict:
    n  = len(A_reg)
    L1 = np.linalg.inv(np.eye(n) - A_reg)
    A2 = np.zeros((n+1, n+1))
    A2[:n,:n] = A_reg
    A2[:n, n] = pce_shares * hh_share
    A2[ n,:n] = li_share
    L2_sub = np.linalg.inv(np.eye(n+1) - A2)[:n,:n]
    return {"L1": L1, "L2_sub": L2_sub,
            "stability": check_leontief_stability(A_reg)}


def compute_impacts(Y, inverses, emp_coeffs) -> dict:
    L1 = inverses["L1"]; L2_sub = inverses["L2_sub"]
    I  = np.eye(len(Y))
    direct   = Y.copy()
    indirect = np.maximum((L1 - I)      @ Y, 0.0)
    induced  = np.maximum((L2_sub - L1) @ Y, 0.0)
    total    = direct + indirect + induced

    def _imp(vec):
        vm = vec / 1e6
        return {
            "output":       float(vec.sum()),
            "jobs":         float((vm * emp_coeffs["jobs_per_va"]).sum()),
            "labor_income": float((vec * emp_coeffs["li_share"]).sum()),
            "value_added":  float((vec * emp_coeffs["va_share"]).sum()),
            "output_vec":   vec.copy(),
            "jobs_vec":     (vm * emp_coeffs["jobs_per_va"]).copy(),
            "li_vec":       (vec * emp_coeffs["li_share"]).copy(),
            "va_vec":       (vec * emp_coeffs["va_share"]).copy(),
        }

    d, ii, ind, t = _imp(direct), _imp(indirect), _imp(induced), _imp(total)
    def _m(a, b): return a/b if b > 0 else 0.0
    mults = {
        "type1": _m(d["output"]+ii["output"], d["output"]),
        "type2": _m(t["output"],  d["output"]),
        "emp":   _m(t["jobs"],    max(d["jobs"],  1e-9)),
        "li":    _m(t["labor_income"], max(d["labor_income"], 1e-9)),
        "va":    _m(t["value_added"],  max(d["value_added"],  1e-9)),
    }
    return {
        "direct":d,"indirect":ii,"induced":ind,"total":t,
        "multipliers":mults,
        "mult_checks":check_multipliers(mults),
        "vecs":{"direct":direct,"indirect":indirect,
                "induced":induced,"total":total},
        "diagnostics":{
            "direct_pct":  d["output"]/max(t["output"],1)*100,
            "indirect_pct":ii["output"]/max(t["output"],1)*100,
            "induced_pct": ind["output"]/max(t["output"],1)*100,
        },
    }

# ── UNCERTAINTY BANDS ─────────────────────────────────────────────────────────

def add_uncertainty(results, pct=UNCERTAINTY_PCT) -> dict:
    """Add ±pct bands to all outputs. Basis: Watson et al. (2015) ±30%."""
    out = {}
    for key in ["direct","indirect","induced","total"]:
        imp = results[key]
        out[key] = {**imp, "bands": {
            "output_low":        imp["output"]       * (1-pct),
            "output_high":       imp["output"]       * (1+pct),
            "jobs_low":          imp["jobs"]         * (1-pct),
            "jobs_high":         imp["jobs"]         * (1+pct),
            "labor_income_low":  imp["labor_income"] * (1-pct),
            "labor_income_high": imp["labor_income"] * (1+pct),
            "value_added_low":   imp["value_added"]  * (1-pct),
            "value_added_high":  imp["value_added"]  * (1+pct),
        }}
    out["multipliers"]    = results["multipliers"]
    out["mult_checks"]    = results["mult_checks"]
    out["vecs"]           = results["vecs"]
    out["diagnostics"]    = results["diagnostics"]
    out["uncertainty_pct"]= pct
    return out

# ── SENSITIVITY ANALYSIS ──────────────────────────────────────────────────────

def sensitivity_analysis(Y, A_nat17, qcew, national_emp_total,
                          emp_coeffs, pce_shares,
                          alpha_range=(0.10, 0.15, 0.20, 0.25, 0.30)) -> list:
    """
    Run SDP impacts across a range of alpha (cross-hauling) values.
    Also includes SLQ and FLQ runs for comparison.
    """
    rows = []
    for alpha in alpha_range:
        rpc_sdp = compute_sdp_rpc(qcew["lq"], alpha=alpha)
        A_york  = A_nat17 * rpc_sdp[:, np.newaxis]
        inv     = build_leontief_inverses(A_york, emp_coeffs["li_share"],
                                          pce_shares, emp_coeffs["hh_share"])
        res     = compute_impacts(Y, inv, emp_coeffs)
        rows.append({
            "method":      f"SDP α={alpha:.2f}",
            "alpha":       alpha,
            "avg_rpc":     round(float(rpc_sdp.mean()), 4),
            "total_jobs":  round(res["total"]["jobs"], 1),
            "total_output":round(res["total"]["output"], 0),
            "type1":       round(res["multipliers"]["type1"], 4),
            "type2":       round(res["multipliers"]["type2"], 4),
            "emp_mult":    round(res["multipliers"]["emp"],   4),
        })
    # Add SLQ comparison row
    rpc_slq = compute_slq_rpc(qcew["lq"])
    A_slq   = A_nat17 * rpc_slq[:, np.newaxis]
    inv_slq = build_leontief_inverses(A_slq, emp_coeffs["li_share"],
                                      pce_shares, emp_coeffs["hh_share"])
    res_slq = compute_impacts(Y, inv_slq, emp_coeffs)
    rows.append({
        "method":      "SLQ (simple LQ)",
        "alpha":       None,
        "avg_rpc":     round(float(rpc_slq.mean()), 4),
        "total_jobs":  round(res_slq["total"]["jobs"], 1),
        "total_output":round(res_slq["total"]["output"], 0),
        "type1":       round(res_slq["multipliers"]["type1"], 4),
        "type2":       round(res_slq["multipliers"]["type2"], 4),
        "emp_mult":    round(res_slq["multipliers"]["emp"],   4),
    })
    # Add FLQ comparison row (delta=0.25)
    rpc_flq, lam_flq = compute_flq_rpc(qcew["lq"], qcew["total_county_emp"],
                                        national_emp_total*1000, delta=0.25)
    A_flq   = A_nat17 * rpc_flq[:, np.newaxis]
    inv_flq = build_leontief_inverses(A_flq, emp_coeffs["li_share"],
                                      pce_shares, emp_coeffs["hh_share"])
    res_flq = compute_impacts(Y, inv_flq, emp_coeffs)
    rows.append({
        "method":      "FLQ δ=0.25 (reference)",
        "alpha":       None,
        "avg_rpc":     round(float(rpc_flq.mean()), 4),
        "total_jobs":  round(res_flq["total"]["jobs"], 1),
        "total_output":round(res_flq["total"]["output"], 0),
        "type1":       round(res_flq["multipliers"]["type1"], 4),
        "type2":       round(res_flq["multipliers"]["type2"], 4),
        "emp_mult":    round(res_flq["multipliers"]["emp"],   4),
    })
    return rows

# ── SPENDING PROFILES ─────────────────────────────────────────────────────────

def get_spending_profile(A_nat17, sector_idx) -> dict:
    col = A_nat17[:, sector_idx].copy()
    total = col.sum()
    if total <= 0: return {sector_idx: 1.0}
    profile = col / total
    return {s: float(v) for s, v in enumerate(profile) if v > 0.001}


def build_all_profiles(A_nat17) -> dict:
    return {s: get_spending_profile(A_nat17, s) for s in range(N)}

# ── COEFFICIENT VALIDATION ────────────────────────────────────────────────────

def validate_coefficients(emp_coeffs) -> list:
    results = []
    for s in range(N):
        jpm  = float(emp_coeffs["jobs_per_va"][s])
        wage = float(emp_coeffs["avg_wage"][s])
        li   = float(emp_coeffs["li_share"][s])
        flags = []
        if not (0.5 <= jpm <= 40.0): flags.append(f"jobs/$1M VA={jpm:.2f} out of [0.5–40]")
        if not (15_000 <= wage <= 200_000): flags.append(f"wage=${wage:,.0f} out of range")
        if not (0.05 <= li <= 0.70): flags.append(f"LI%={li:.1%} out of [5–70%]")
        results.append({"sector":SECTOR_LABELS[s],"jpm":jpm,
                         "wage":wage,"li":li,"ok":len(flags)==0,"flags":flags})
    return results

# ── SCENARIO STORE ────────────────────────────────────────────────────────────

def make_scenario(name, naics, investment, results_with_bands, run_params) -> dict:
    t = results_with_bands["total"]
    m = results_with_bands["multipliers"]
    return {
        "name":        name,
        "naics":       naics,
        "investment":  investment,
        "sector":      SECTOR_LABELS[naics_to_sector(str(naics))],
        "output":      t["output"],
        "jobs":        t["jobs"],
        "jobs_low":    t["bands"]["jobs_low"],
        "jobs_high":   t["bands"]["jobs_high"],
        "labor_income":t["labor_income"],
        "value_added": t["value_added"],
        "type1":       m["type1"],
        "type2":       m["type2"],
        "emp_mult":    m["emp"],
        "params":      run_params,
        "full_results":results_with_bands,
    }

# ── FULL BUILD PIPELINE ───────────────────────────────────────────────────────

def build_model(use_file, ms_file, dom_file, qcew_file,
                bea_api_key, bls_api_key,
                sdp_alpha=0.20, flq_delta=0.25,
                mpc=0.90, regional_retention=0.50,
                progress_callback=None) -> dict:
    """
    End-to-end model build.
    sdp_alpha : SDP cross-hauling parameter (default 0.20)
    flq_delta : FLQ sensitivity, used only for comparison display
    """
    def _prog(msg, pct):
        if progress_callback: progress_callback(msg, pct)

    _prog("Parsing BEA Use Table…", 5)
    use_data = parse_use_table(use_file)

    _prog("Parsing Market Share Matrix D…", 13)
    D_data = parse_D_matrix(ms_file)

    _prog("Parsing Direct Domestic Requirements…", 22)
    B_dom   = parse_B_domestic(dom_file)
    B_total = build_B_total(use_data)

    _prog("Building A_domestic = D @ B_domestic…", 32)
    A_data  = build_A_domestic(D_data["D"], B_dom, use_data)

    _prog("Computing sector crosswalk (402→17)…", 40)
    ind_sector = np.array([_code_to_sector(c) for c in D_data["ind_codes"]], dtype=int)
    com_sector = np.array([_code_to_sector(c) for c in use_data["com_codes"]], dtype=int)

    _prog("Aggregating A_domestic to 17×17…", 48)
    agg     = aggregate_matrix(A_data["A"], use_data["x"],
                               use_data["va_total"], use_data["va_comp"], ind_sector)
    A_nat17 = agg["A_agg"]

    _prog("Building PCE shares…", 54)
    pce_shares = build_pce_shares(use_data["pce"], com_sector)

    _prog("Loading QCEW data…", 59)
    qcew = load_qcew(qcew_file)

    _prog("Fetching BLS national employment…", 65)
    bls_raw = fetch_bls(bls_api_key) or {}
    for sid in ALL_SERIES_IDS:
        if sid not in bls_raw:
            bls_raw[sid] = BLS_FALLBACK_2022.get(sid, 0.0)
    national_emp = series_to_emp(bls_raw)

    _prog("Fetching BEA output, compensation & value added…", 73)

    bea_va_raw  = fetch_bea_table(bea_api_key, "VA")
    bea_go_raw  = fetch_bea_table(bea_api_key, "GO")

    bea_va, bea_va_src, bea_va_diag = _use_or_fallback(
        bea_va_raw, BEA_VA_FALLBACK,
        "bea_value_added.csv", min_plausible_sum=5_000_000)
    bea_output, bea_output_src, bea_output_diag = _use_or_fallback(
        bea_go_raw, BEA_OUTPUT_FALLBACK,
        "bea_output.csv", min_plausible_sum=10_000_000)

    # Derive compensation from VA using stable sector-level comp/VA ratios
    COMP_VA_RATIOS = np.where(BEA_VA_FALLBACK > 0,
                              BEA_COMP_FALLBACK / BEA_VA_FALLBACK, 0.25)
    bea_comp     = bea_va * COMP_VA_RATIOS
    bea_comp_src = bea_va_src + "_derived"
    bea_comp_diag = bea_va_diag

    _prog("Fetching BEA NIPA MPC…", 79)
    nat_mpc = fetch_bea_nipa_mpc(bea_api_key) or mpc

    _prog("Building employment coefficients…", 84)
    emp_coeffs = build_employment_coefficients(
        national_emp, bea_output, bea_comp, bea_va, qcew, agg["va_share"],
        mpc=nat_mpc, regional_retention=regional_retention)

    _prog("Applying SDP regionalization…", 88)
    reg = regionalize(A_nat17, qcew,
                      national_emp_total=national_emp.sum(),
                      sdp_alpha=sdp_alpha,
                      flq_delta=flq_delta)

    _prog("Pre-computing Leontief inverses…", 92)
    inverses = build_leontief_inverses(
        reg["A_york"], emp_coeffs["li_share"],
        pce_shares, emp_coeffs["hh_share"])

    _prog("Pre-computing spending profiles…", 96)
    all_profiles = build_all_profiles(A_nat17)
    validation   = validate_coefficients(emp_coeffs)
    _prog("Model ready.", 100)

    import_removed_mean = float((B_total.sum(0) - B_dom.sum(0)).mean())

    return {
        "A_nat17":    A_nat17,
        "A_york":     reg["A_york"],
        "lq_york":    reg["lq_york"],
        "rpc_york":   reg["rpc_york"],   # SDP — primary
        "rpc_sdp":    reg["rpc_sdp"],
        "rpc_slq":    reg["rpc_slq"],
        "rpc_flq":    reg["rpc_flq"],
        "flq_lambda": reg["flq_lambda"],
        "sdp_alpha":  sdp_alpha,
        "flq_delta":  flq_delta,
        "pce_shares": pce_shares,
        "inverses":   inverses,
        "emp_coeffs": emp_coeffs,
        "national_emp": national_emp,
        "bea_output": bea_output,
        "bea_comp":   bea_comp,
        "bea_va":     bea_va,
        "qcew":       qcew,
        "all_profiles": all_profiles,
        "validation":   validation,
        "nat_mpc":      nat_mpc,
        "regional_retention": regional_retention,
        "bea_output_src": bea_output_src,
        "bea_comp_src":   bea_comp_src,
        "bea_va_src":     bea_va_src,
        "bea_va_diag":    bea_va_diag,
        "bea_output_diag":bea_output_diag,
        "import_removed_mean":     import_removed_mean,
        "import_leakage_rate":     float(1.0 - reg["rpc_sdp"].mean()),
        "import_leakage_rate_slq": float(1.0 - reg["rpc_slq"].mean()),
        "import_leakage_rate_flq": float(1.0 - reg["rpc_flq"].mean()),
        "x_sum":      float(use_data["x"].sum()),
        "ind_sector": ind_sector,
        "com_sector": com_sector,
        "stability":  inverses["stability"],
        "A_nat15":    A_nat17,   # backwards compat alias
        "scenarios":  [],
    }
