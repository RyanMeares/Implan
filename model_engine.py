"""
model_engine.py  —  Regional Economic Impact Model  v5.0
─────────────────────────────────────────────────────────
Pure-computation module. No Streamlit imports. No side effects.

Improvements over v4:
  1. FLQ regionalization  (Flegg & Webber 2000) replaces simple SLQ
  2. Separated MPC × regional_retention induced-effect parameters
  3. Jobs per $1M VALUE ADDED (BEA Table 1) replaces jobs per gross output
  4. Leontief stability check  (spectral radius < 1)
  5. Multiplier sanity checks  (Type I 1.1–2.5, Type II 1.2–3.0)
  6. Local CSV cache fallback  (/data/ folder) when BEA API fails
  7. Leontief inverse pre-computed once, reused across scenarios
  8. All existing functionality preserved (402-industry BEA, QCEW, crosswalk)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
import requests

warnings.filterwarnings("ignore")

# ── PATH TO BUNDLED FALLBACK CACHE ────────────────────────────────────────────
# Resolved relative to this file so it works regardless of working directory.
_HERE      = Path(__file__).parent
DATA_DIR   = _HERE / "data"

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

SECTOR_LABELS = [
    "Agriculture",            # 0  NAICS 11
    "Mining & Utilities",     # 1  NAICS 21,22
    "Construction",           # 2  NAICS 23
    "Mfg — Durable",          # 3  NAICS 33DG
    "Mfg — Non-Durable",      # 4  NAICS 31ND
    "Wholesale Trade",        # 5  NAICS 42
    "Retail Trade",           # 6  NAICS 44-45
    "Transportation & Whsg.", # 7  NAICS 48-49
    "Information",            # 8  NAICS 51
    "Finance & Real Estate",  # 9  NAICS 52-53
    "Professional Services",  # 10 NAICS 54-55
    "Admin & Waste Services", # 11 NAICS 56
    "Education & Health",     # 12 NAICS 61-62
    "Arts & Accommodation",   # 13 NAICS 71-72
    "Other Services & Gov",   # 14 NAICS 81,92
]
N = len(SECTOR_LABELS)

BEA_BASE = "https://apps.bea.gov/api/data"
BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# Multiplier plausibility bounds (Type I output, Type II output)
MULT_TYPE1_MIN, MULT_TYPE1_MAX = 1.1, 2.5
MULT_TYPE2_MIN, MULT_TYPE2_MAX = 1.2, 3.0

# ── SECTOR CROSSWALK ──────────────────────────────────────────────────────────

_SPECIAL = {
    "S00500":14,"S00600":14,"S00101":14,"S00102":14,
    "S00201":14,"S00202":14,"S00203":14,
    "GSLGE":14,"GSLGH":14,"GSLGO":14,
    "S00401":-1,"S00402":-1,"S00300":-1,"S00900":-1,
    "531HSO":9,"531HSR":9,"532RL":9,"ORE000":9,"HS0000":9,
    "521CI0":9,"525000":9,"4B0000":6,"487OS0":7,"488A00":7,
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
    "61":12,"62":12,"71":13,"72":13,"81":14,"92":14,
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
    if not d:
        return 14
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
    else:                        return 14

# ── FILE PARSERS ──────────────────────────────────────────────────────────────

def parse_use_table(src, sheet="2017") -> dict:
    """Parse BEA Use Table after redefinitions. Returns Z, x, va arrays."""
    wb   = openpyxl.load_workbook(src, read_only=True, data_only=True)
    ws   = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    C0, C1, R0, R1, PCE_COL = 2, 404, 6, 408, 405
    hdr_codes = list(rows[5])
    hdr_names = list(rows[4])
    def _arr(row_idx):
        return np.array([rows[row_idx][c] or 0.0 for c in range(C0,C1)], dtype=float)
    Z = np.zeros((R1-R0, C1-C0), dtype=float)
    for ri, r in enumerate(range(R0, R1)):
        for ci, c in enumerate(range(C0, C1)):
            Z[ri,ci] = rows[r][c] or 0.0
    return {
        "Z":         Z,
        "x":         _arr(413),
        "va_comp":   _arr(409),
        "va_taxes":  _arr(410),
        "va_gos":    _arr(411),
        "va_total":  _arr(412),
        "ind_codes": [str(hdr_codes[c]) for c in range(C0,C1)],
        "ind_names": [str(hdr_names[c]) for c in range(C0,C1)],
        "com_codes": [str(rows[r][0])   for r in range(R0,R1)],
        "pce":       np.array([rows[r][PCE_COL] or 0.0
                               for r in range(R0,R1)], dtype=float),
    }


def parse_D_matrix(src, sheet="2017") -> dict:
    """Parse Market Share Matrix D (after redefinitions). Returns D (402×402)."""
    wb   = openpyxl.load_workbook(src, read_only=True, data_only=True)
    ws   = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    ind_codes = [str(rows[r][0]) for r in range(6, 408)]
    D = np.zeros((402, 402), dtype=float)
    for ri, r in enumerate(range(6, 408)):
        for ci, c in enumerate(range(2, 404)):
            v = rows[r][c]
            D[ri,ci] = v if v is not None else 0.0
    col_sums = D.sum(axis=0)
    bad = np.abs(col_sums - 1.0) > 0.01
    if bad.any():
        D[:, bad] /= np.where(col_sums[bad] > 0, col_sums[bad], 1.0)
    return {"D": D, "ind_codes": ind_codes}


def parse_B_domestic(src, sheet="2017") -> np.ndarray:
    """Parse Direct Domestic Requirements B_domestic (402×402)."""
    wb   = openpyxl.load_workbook(src, read_only=True, data_only=True)
    ws   = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    B = np.zeros((402, 402), dtype=float)
    for ri, r in enumerate(range(5, 407)):
        for ci, c in enumerate(range(2, 404)):
            v = rows[r][c]
            B[ri,ci] = v if v is not None else 0.0
    return B


def load_qcew(src) -> dict:
    """
    Parse BLS QCEW county annual CSV.
    Returns emp, lq, wage arrays (length N=15), total county employment, and year.
    """
    df = pd.read_csv(src)
    df["industry_code"] = df["industry_code"].astype(str).str.strip()
    year = int(df["year"].max())

    NAICS2_MAP = {
        "11":0,"22":1,"23":2,"31-33":3,
        "42":5,"44-45":6,"48-49":7,
        "51":8,"52":9,"53":9,"54":10,"55":10,"56":11,
        "61":12,"62":12,"71":13,"72":13,"81":14,"99":14
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

    # Government: net employment with matching wage scaling
    gov_rows   = df[df["own_code"].isin([1,2,3]) & (df["agglvl_code"]==71)]
    total_all  = df[(df["own_code"]==0) & (df["agglvl_code"]==70)]["annual_avg_emplvl"]
    total_priv = df[(df["own_code"]==5) & (df["agglvl_code"]==71)]["annual_avg_emplvl"]
    if len(total_all) and len(total_priv):
        gov_emp_net = max(float(total_all.values[0]) - float(total_priv.values[0]), 0)
        if gov_emp_net > 0 and len(gov_rows):
            gov_wages_total = float(gov_rows["total_annual_wages"].sum())
            gov_emp_total   = float(gov_rows["annual_avg_emplvl"].sum())
            scale = gov_emp_net / max(gov_emp_total, 1)
            emp[14]    += gov_emp_net
            wage_n[14] += gov_wages_total * scale

    lq    = np.where(emp > 0, lq_n   / emp, 0.0)
    wage  = np.where(emp > 0, wage_n  / emp, 0.0)
    fips  = str(df["area_fips"].iloc[0])
    # Total county employment for FLQ calculation
    total_county_emp = float(emp.sum())
    return {
        "emp":              emp,
        "lq":               lq,
        "wage":             wage,
        "year":             year,
        "fips":             fips,
        "total_county_emp": total_county_emp,
    }

# ── MATRIX BUILDERS ───────────────────────────────────────────────────────────

def build_A_domestic(D: np.ndarray, B_dom: np.ndarray, use_data: dict) -> dict:
    """
    Compute A_domestic = D_redef @ B_domestic.
    Cap any columns ≥ 1.0 to 0.99 (Leontief stability requirement).
    """
    A = D @ B_dom
    col_sums = A.sum(axis=0)
    bad = col_sums >= 1.0
    if bad.any():
        A[:, bad] *= 0.99 / col_sums[bad]
    with np.errstate(divide="ignore", invalid="ignore"):
        va_share = np.where(use_data["x"] > 0,
                            use_data["va_total"] / use_data["x"], 0.0)
        li_share = np.where(use_data["x"] > 0,
                            use_data["va_comp"]  / use_data["x"], 0.0)
    return {"A": A, "va_share": va_share, "li_share": li_share}


def build_B_total(use_data: dict) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(use_data["x"] > 0,
                        use_data["Z"] / use_data["x"][np.newaxis,:], 0.0)


def aggregate_matrix(A: np.ndarray, x: np.ndarray,
                     va_total: np.ndarray, va_comp: np.ndarray,
                     sector_map: np.ndarray) -> dict:
    """Aggregate 402×402 A_domestic to N×N via output-weighted averaging."""
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
        va_share = np.where(x_agg > 0, va_agg   / x_agg, 0.0)
        li_share = np.where(x_agg > 0, comp_agg / x_agg, 0.0)
    return {"A_agg": A_agg, "x_agg": x_agg,
            "va_share": va_share, "li_share": li_share}


def build_pce_shares(pce_raw: np.ndarray, com_sector: np.ndarray) -> np.ndarray:
    pce_agg = np.zeros(N)
    for i, s in enumerate(com_sector):
        if 0 <= s < N:
            pce_agg[s] += max(pce_raw[i], 0.0)
    total = pce_agg.sum()
    return pce_agg / total if total > 0 else np.ones(N) / N

# ── FLQ REGIONALIZATION ───────────────────────────────────────────────────────

def compute_flq_rpc(
    lq: np.ndarray,
    total_county_emp: float,
    total_national_emp: float,
    delta: float = 0.25,
) -> np.ndarray:
    """
    Flegg Location Quotient (FLQ) regionalization.

    Flegg & Webber (2000) — corrects the well-documented small-region upward
    bias of Simple Location Quotient (SLQ) methods. SLQ sets RPC = min(LQ,1)
    regardless of regional size, which systematically over-estimates local
    supply capacity for small counties like York (22k workers vs. 154M national).

    Formula:
        lambda = [log2(1 + R_total / N_total)] ^ delta
        FLQ_s  = lambda * LQ_s
        RPC_s  = min(FLQ_s, 1.0)

    Parameters
    ----------
    lq                : sector location quotients from QCEW (N,)
    total_county_emp  : total county employment (workers, from QCEW)
    total_national_emp: total national employment (workers, from BLS CES)
    delta             : sensitivity parameter, 0.10–0.35
                        Flegg & Webber (2000) recommend 0.25–0.30 for most regions.
                        Lower delta → less downward adjustment (approaches SLQ).
                        Higher delta → more downward adjustment (more import leakage).
                        Default 0.25 is appropriate for a small suburban county
                        embedded in a large metro area.

    Returns
    -------
    rpc : Regional Purchase Coefficients (N,), capped at 1.0
    """
    # lambda is a scalar that scales ALL sector LQs downward
    # It approaches 1.0 as the region approaches national scale
    # It approaches 0.0 as the region becomes infinitesimally small
    ratio = total_county_emp / max(total_national_emp, 1.0)
    lam   = float(np.log2(1.0 + ratio) ** delta)

    flq = lq * lam
    rpc = np.minimum(flq, 1.0)
    return rpc, lam


def regionalize(
    A_nat15: np.ndarray,
    qcew: dict,
    national_emp_total: float,
    delta: float = 0.25,
) -> dict:
    """
    Regionalize A_national using FLQ method.

    Returns A_york, lq_york, rpc_york (FLQ-based), and the lambda scalar.
    Also returns rpc_slq for comparison display.
    """
    lq_york       = qcew["lq"].copy()
    # SLQ (old method) — kept for comparison only
    rpc_slq       = np.minimum(lq_york, 1.0)
    # FLQ (new method)
    rpc_york, lam = compute_flq_rpc(
        lq_york,
        total_county_emp   = qcew["total_county_emp"],
        total_national_emp = national_emp_total * 1000,   # convert k→workers
        delta              = delta,
    )
    A_york = A_nat15 * rpc_york[:, np.newaxis]
    return {
        "A_york":   A_york,
        "lq_york":  lq_york,
        "rpc_york": rpc_york,   # FLQ-based
        "rpc_slq":  rpc_slq,    # SLQ for comparison
        "lambda":   lam,
        "delta":    delta,
    }

# ── BLS / BEA API & CACHE ─────────────────────────────────────────────────────

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
    14:   "CEU8000000001", "14g":"CEU9000000001",
}
ALL_SERIES_IDS = list(BLS_SERIES.values())
AGR_NATIONAL_EMP_FALLBACK = 2109.9   # thousands — no CES series for agriculture

BLS_FALLBACK_2022 = {
    "CEU1021000001":657.5,  "CEU0622000001":574.0,
    "CEU2000000001":7697.2, "CEU3100000001":9122.5, "CEU3200000001":4841.6,
    "CEU4142000001":6153.4, "CEU4200000001":15592.8,"CEU4300000001":6622.7,
    "CEU5000000001":3038.9, "CEU5552000001":6617.6, "CEU5553000001":2394.6,
    "CEU6054000001":10199.4,"CEU6055000001":2906.5, "CEU6056000001":9628.6,
    "CEU6561000001":3787.0, "CEU6562000001":20411.9,"CEU7071000001":2615.7,
    "CEU7072000001":12346.2,"CEU8000000001":5697.4, "CEU9000000001":22524.2,
}

# BEA 2022 fallback arrays — all values in $M
# Source: BEA GDP-by-Industry, 2022 annual estimates
BEA_OUTPUT_FALLBACK = np.array([
    464200,1031800,1872400,3421600,3190200,2391700,2088400,
    1512300,2155700,7321500,3289100,1256100,3298400,1702800,5158300
], dtype=float)

BEA_COMP_FALLBACK = np.array([
    58200,174700,617400,841800,402700,618800,686900,473100,
    426800,1063500,1562300,438100,1352300,578800,2572600
], dtype=float)

# BEA Table 1 — Value Added by Industry 2022 ($M)
# Used for jobs_per_value_added coefficient (improvement #3)
BEA_VA_FALLBACK = np.array([
    233700, 492000, 993400,1371800, 790200,1202800,1213700,
    753100,1198400,5068700,2204500, 731800,2244400, 994200,3862400
], dtype=float)

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
    "81":14,"GFGD":14,"GFGN":14,"GFE":14,"GSLE":14,"GSLO":14,
}


def _load_csv_cache(filename: str) -> np.ndarray | None:
    """
    Load a BEA fallback CSV from the /data directory.
    CSV must have columns: sector, value_millions
    Returns array of length N aligned to SECTOR_LABELS, or None if file missing.
    """
    path = DATA_DIR / filename
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # Align by sector name order matching SECTOR_LABELS
        result = np.zeros(N)
        for i, label in enumerate(SECTOR_LABELS):
            match = df[df["sector"] == label]["value_millions"]
            if len(match):
                result[i] = float(match.values[0])
        return result if result.sum() > 0 else None
    except Exception:
        return None


def _bea_extract_rows(data: dict) -> list:
    results = data.get("BEAAPI",{}).get("Results",{})
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and "Data" in item:
                return item["Data"]
        return []
    elif isinstance(results, dict):
        return results.get("Data", [])
    return []


def fetch_bea_table(api_key: str, table_id: str, year: str = "2022") -> np.ndarray | None:
    params = {
        "UserID": api_key, "method": "GetData",
        "datasetname": "GDPbyIndustry", "TableID": table_id,
        "Frequency": "A", "Year": year, "Industry": "ALL", "ResultFormat": "JSON"
    }
    try:
        r = requests.get(BEA_BASE, params=params, timeout=30)
        r.raise_for_status()
        rows = _bea_extract_rows(r.json())
        if not rows:
            return None
        result = np.zeros(N)
        for row in rows:
            ind = str(row.get("Industry","")).strip()
            if ind in BEA_SUMMARY_TO_SECTOR:
                try:
                    result[BEA_SUMMARY_TO_SECTOR[ind]] += float(
                        str(row.get("DataValue","")).replace(",",""))
                except ValueError:
                    pass
        return result
    except Exception:
        return None


def fetch_bea_nipa_mpc(api_key: str, year: str = "2022") -> float | None:
    params = {
        "UserID": api_key, "method": "GetData", "datasetname": "NIPA",
        "TableName": "T20600", "Frequency": "A", "Year": year, "ResultFormat": "JSON"
    }
    try:
        r = requests.get(BEA_BASE, params=params, timeout=30)
        r.raise_for_status()
        vals = {}
        for row in _bea_extract_rows(r.json()):
            sc = row.get("SeriesCode","")
            try:
                vals[sc] = float(str(row.get("DataValue","")).replace(",",""))
            except ValueError:
                pass
        dpi = vals.get("A065RC"); outlays = vals.get("A068RC")
        if dpi and outlays and dpi > 0:
            return outlays / dpi
        return None
    except Exception:
        return None


def fetch_bls(api_key: str, year: str = "2022") -> dict | None:
    payload = {
        "seriesid": ALL_SERIES_IDS, "startyear": year, "endyear": year,
        "annualaverage": "true", "registrationkey": api_key
    }
    try:
        r = requests.post(BLS_BASE, json=payload, timeout=45)
        r.raise_for_status()
        resp = r.json()
        if resp.get("status") != "REQUEST_SUCCEEDED":
            return None
        raw = {}
        for series in resp["Results"]["series"]:
            sid = series["seriesID"]
            val = next((float(d["value"]) for d in series["data"]
                        if d["period"] == "M13"), None)
            if val is None:
                monthly = [float(d["value"]) for d in series["data"]
                           if d["period"].startswith("M") and d["period"] != "M13"]
                val = float(np.mean(monthly)) if monthly else 0.0
            raw[sid] = val
        return raw
    except Exception:
        return None


def series_to_emp(raw: dict) -> np.ndarray:
    emp = np.zeros(N)
    emp[0] = AGR_NATIONAL_EMP_FALLBACK
    for idx in [2,3,4,5,6,7,8,11,14]:
        emp[idx] = raw.get(BLS_SERIES[idx], 0.0)
    emp[1]  = raw.get(BLS_SERIES["1a"],0) + raw.get(BLS_SERIES["1b"],0)
    emp[9]  = raw.get(BLS_SERIES["9a"],0) + raw.get(BLS_SERIES["9b"],0)
    emp[10] = raw.get(BLS_SERIES["10a"],0)+ raw.get(BLS_SERIES["10b"],0)
    emp[12] = raw.get(BLS_SERIES["12a"],0)+ raw.get(BLS_SERIES["12b"],0)
    emp[13] = raw.get(BLS_SERIES["13a"],0)+ raw.get(BLS_SERIES["13b"],0)
    emp[14]+= raw.get(BLS_SERIES["14g"],0)
    return emp


def _use_or_fallback(
    result: np.ndarray | None,
    api_fallback: np.ndarray,
    csv_filename: str,
    min_plausible_sum: float | None = None,
) -> tuple[np.ndarray, str]:
    """
    Three-tier fallback hierarchy:
      1. Live BEA API result (if plausible)
      2. Local CSV cache from /data/ directory
      3. Hardcoded 2022 numpy array constants

    Returns (array, source_label).
    """
    # Tier 1: live API
    if result is not None and result.sum() > 0:
        if min_plausible_sum is None or result.sum() >= min_plausible_sum:
            return result, "live"

    # Tier 2: local CSV cache
    csv_result = _load_csv_cache(csv_filename)
    if csv_result is not None:
        return csv_result, "csv_cache"

    # Tier 3: hardcoded fallback
    return api_fallback, "hardcoded_fallback"

# ── EMPLOYMENT COEFFICIENTS ───────────────────────────────────────────────────

def build_employment_coefficients(
    national_emp: np.ndarray,
    bea_output:   np.ndarray,
    bea_comp:     np.ndarray,
    bea_va:       np.ndarray,
    qcew:         dict,
    va_share_agg: np.ndarray,
    mpc:                float = 0.90,
    regional_retention: float = 0.50,
) -> dict:
    """
    Build employment and income coefficient vectors.

    Improvement #3: jobs_per_value_added uses BEA Value Added (Table 1)
    as the denominator instead of gross output. Value added is the GDP
    contribution of each sector — it excludes intermediate inputs and
    directly reflects labor and capital income. This gives a more
    meaningful employment intensity measure comparable to IMPLAN.

    Unit reference:
      national_emp  : thousands of workers (BLS CES)
      bea_output    : $M gross output (BEA Table 5)
      bea_comp      : $M compensation (BEA Table 2)
      bea_va        : $M value added (BEA Table 1) ← NEW
      qcew['wage']  : $/worker/year

    Derived coefficients:
      jobs_per_va   = (emp_thousands * 1000) / va_$M
                    = workers / $M value added
      nat_avg_wage  = (comp_$M * 1e6) / (emp_thousands * 1000)
                    = $/worker/year
      york_li_share = (york_wage * emp * 1000) / (output_$M * 1e6)
                    = dimensionless wage share of gross output

    Improvement #2: hh_share is now MPC × regional_retention (two explicit
    parameters) rather than a single "dampening" scalar.
      MPC               : marginal propensity to consume (BEA NIPA, default 0.90)
      regional_retention: share of consumption spending staying in region (default 0.50)
      hh_share = MPC × regional_retention
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # jobs per $1M of VALUE ADDED (not gross output)
        jobs_per_va = np.where(bea_va > 0,
                               (national_emp * 1000) / bea_va, 0.0)
        # national average wage $/worker/year
        nat_avg_wage = np.where(national_emp > 0,
                                (bea_comp * 1e6) / (national_emp * 1000), 0.0)
        # BEA labor income share (national): comp / gross output
        li_share_bea = np.where(bea_output > 0, bea_comp / bea_output, 0.0)

    # York County wage: QCEW where available, else national average
    york_wage = np.where(qcew["emp"] > 0, qcew["wage"], nat_avg_wage)

    # York LI share: york_wage * national workers / national gross output
    york_li_share = np.where(
        bea_output > 0,
        (york_wage * national_emp * 1000) / (bea_output * 1e6),
        li_share_bea
    )
    york_li_share = np.clip(york_li_share, 0.0, 0.70)

    # Household spending share entering the regional economy per $ of output
    hh_share = mpc * regional_retention

    return {
        "jobs_per_va":        jobs_per_va,          # primary employment coeff
        "jobs_per_million":   jobs_per_va,          # alias — kept for compatibility
        "avg_wage":           york_wage,
        "li_share":           york_li_share,
        "va_share":           va_share_agg,
        "hh_share":           hh_share,
        "nat_avg_wage":       nat_avg_wage,
        "mpc":                mpc,
        "regional_retention": regional_retention,
    }

# ── LEONTIEF STABILITY CHECK ──────────────────────────────────────────────────

def check_leontief_stability(A: np.ndarray) -> dict:
    """
    Improvement #4: Verify Leontief stability before solving.

    The Leontief inverse (I-A)^{-1} only converges if the spectral radius
    of A is strictly less than 1. The spectral radius is the largest absolute
    eigenvalue of A.

    If spectral_radius >= 1, the model is theoretically unstable — the cascade
    does not converge and the inverse cannot be meaningfully interpreted.
    In practice this is caused by column-sum normalization errors or data issues.

    Returns a dict with:
      spectral_radius : float
      stable          : bool (True if < 1.0)
      max_col_sum     : float (should be < 1.0 for all columns)
      warnings        : list of warning strings
    """
    eigenvalues   = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    col_sums      = A.sum(axis=0)
    max_col_sum   = float(col_sums.max())
    warnings_list = []

    if spectral_radius >= 1.0:
        warnings_list.append(
            f"CRITICAL: spectral radius = {spectral_radius:.4f} ≥ 1.0. "
            f"Leontief inverse will not converge. Check A matrix normalization."
        )
    elif spectral_radius > 0.95:
        warnings_list.append(
            f"WARNING: spectral radius = {spectral_radius:.4f} is close to 1.0. "
            f"Model is technically stable but multipliers may be very large."
        )

    if max_col_sum >= 1.0:
        n_bad = (col_sums >= 1.0).sum()
        warnings_list.append(
            f"WARNING: {n_bad} column(s) have sum ≥ 1.0 (max={max_col_sum:.4f}). "
            f"These indicate sectors spending more on inputs than they earn."
        )

    return {
        "spectral_radius": spectral_radius,
        "stable":          spectral_radius < 1.0,
        "max_col_sum":     max_col_sum,
        "col_sums":        col_sums,
        "warnings":        warnings_list,
    }

# ── MULTIPLIER SANITY CHECKS ──────────────────────────────────────────────────

def check_multipliers(mults: dict) -> list[dict]:
    """
    Improvement #5: Warn if multipliers fall outside realistic economic ranges.

    Ranges based on:
    - BEA RIMS II published multipliers for US counties (1990–2022)
    - IMPLAN validation studies (Watson et al. 2015, Loomis & White 1996)
    - Peer-reviewed I-O literature for small US counties

    Type I  output multiplier: 1.1 – 2.5
    Type II output multiplier: 1.2 – 3.0
    Employment multiplier:     1.1 – 4.0  (wider range, sector-specific)
    """
    checks = []
    checks.append({
        "metric":  "Type I Output Multiplier",
        "value":   mults["type1"],
        "low":     MULT_TYPE1_MIN,
        "high":    MULT_TYPE1_MAX,
        "ok":      MULT_TYPE1_MIN <= mults["type1"] <= MULT_TYPE1_MAX,
        "note":    "Indirect supply-chain effects only"
    })
    checks.append({
        "metric":  "Type II Output Multiplier",
        "value":   mults["type2"],
        "low":     MULT_TYPE2_MIN,
        "high":    MULT_TYPE2_MAX,
        "ok":      MULT_TYPE2_MIN <= mults["type2"] <= MULT_TYPE2_MAX,
        "note":    "Includes induced household spending"
    })
    checks.append({
        "metric":  "Employment Multiplier",
        "value":   mults["emp"],
        "low":     1.1,
        "high":    4.0,
        "ok":      1.1 <= mults["emp"] <= 4.0,
        "note":    "Total jobs / direct jobs"
    })
    return checks

# ── LEONTIEF ENGINE WITH PRE-COMPUTED INVERSE ─────────────────────────────────

def build_leontief_inverses(
    A_reg:      np.ndarray,
    li_share:   np.ndarray,
    pce_shares: np.ndarray,
    hh_share:   float,
) -> dict:
    """
    Improvement #7: Pre-compute both Leontief inverses once.
    Store them in the model state so they are reused across scenario runs
    without recomputing the matrix inversion each time.

    Returns L1 (Type I), L2_sub (Type II, N×N submatrix), and stability info.
    """
    n  = len(A_reg)
    I  = np.eye(n)

    # Type I: standard Leontief inverse
    L1 = np.linalg.inv(I - A_reg)

    # Type II: augmented with household row/column
    A2         = np.zeros((n+1, n+1))
    A2[:n,:n]  = A_reg
    A2[:n, n]  = pce_shares * hh_share   # household expenditure column
    A2[ n,:n]  = li_share                # labor income row
    L2_full    = np.linalg.inv(np.eye(n+1) - A2)
    L2_sub     = L2_full[:n,:n]

    stability = check_leontief_stability(A_reg)

    return {
        "L1":        L1,
        "L2_sub":    L2_sub,
        "stability": stability,
    }


def compute_impacts(
    Y:          np.ndarray,
    inverses:   dict,
    emp_coeffs: dict,
) -> dict:
    """
    Compute direct, indirect, induced, and total economic impacts.

    Improvement #7: Receives pre-computed inverses dict (L1, L2_sub)
    rather than recomputing matrix inversions on each call.

    Improvement #2: Induced effect uses explicit MPC × regional_retention
    via hh_share already baked into L2_sub during build_leontief_inverses().

    Y          : final demand vector ($)
    inverses   : from build_leontief_inverses()
    emp_coeffs : from build_employment_coefficients()
    """
    L1     = inverses["L1"]
    L2_sub = inverses["L2_sub"]
    n      = len(Y)
    I      = np.eye(n)

    direct   = Y.copy()
    indirect = np.maximum((L1 - I)       @ Y, 0.0)
    induced  = np.maximum((L2_sub - L1)  @ Y, 0.0)
    total    = direct + indirect + induced

    # Leakage and retention diagnostics
    # import_leakage_rate: fraction of national A that was zeroed by regionalization
    # We compute it from what was NOT spent locally (1 - weighted RPC)
    # These are computed at the model level; we report totals here
    direct_output   = float(direct.sum())
    indirect_output = float(indirect.sum())
    induced_output  = float(induced.sum())

    def _imp(vec: np.ndarray) -> dict:
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

    def _m(a, b): return a / b if b > 0 else 0.0
    mults = {
        "type1": _m(d["output"] + ii["output"], d["output"]),
        "type2": _m(t["output"],  d["output"]),
        "emp":   _m(t["jobs"],    max(d["jobs"],  1e-9)),
        "li":    _m(t["labor_income"], max(d["labor_income"], 1e-9)),
        "va":    _m(t["value_added"],  max(d["value_added"],  1e-9)),
    }

    mult_checks = check_multipliers(mults)

    return {
        "direct":       d,
        "indirect":     ii,
        "induced":      ind,
        "total":        t,
        "multipliers":  mults,
        "mult_checks":  mult_checks,
        "vecs": {
            "direct":   direct,
            "indirect": indirect,
            "induced":  induced,
            "total":    total,
        },
        "diagnostics": {
            "direct_pct":   direct_output  / max(t["output"], 1) * 100,
            "indirect_pct": indirect_output/ max(t["output"], 1) * 100,
            "induced_pct":  induced_output / max(t["output"], 1) * 100,
        },
    }

# ── SPENDING PROFILES ─────────────────────────────────────────────────────────

def get_spending_profile(A_nat15: np.ndarray, sector_idx: int) -> dict:
    """
    BEA-derived intermediate input distribution for a sector.
    Normalized column of A_nat15 — used to split investment $ across Y vector.
    """
    col15 = A_nat15[:, sector_idx].copy()
    total = col15.sum()
    if total <= 0:
        return {sector_idx: 1.0}
    profile = col15 / total
    return {s: float(v) for s, v in enumerate(profile) if v > 0.001}


def build_all_profiles(A_nat15: np.ndarray) -> dict:
    return {s: get_spending_profile(A_nat15, s) for s in range(N)}

# ── COEFFICIENT VALIDATION ────────────────────────────────────────────────────

def validate_coefficients(emp_coeffs: dict) -> list[dict]:
    """
    Per-sector plausibility check on employment and income coefficients.
    Benchmarks from IMPLAN/RIMS II comparisons.
    Uses jobs_per_va thresholds (wider range than jobs_per_gross_output).
    """
    results = []
    for s in range(N):
        jpm  = float(emp_coeffs["jobs_per_va"][s])
        wage = float(emp_coeffs["avg_wage"][s])
        li   = float(emp_coeffs["li_share"][s])
        flags = []
        # jobs per $1M VA — wider range than per gross output
        if not (0.5 <= jpm <= 40.0):
            flags.append(f"jobs/$1M VA={jpm:.2f} out of range [0.5–40]")
        if not (15_000 <= wage <= 200_000):
            flags.append(f"wage=${wage:,.0f} out of range")
        if not (0.05 <= li <= 0.70):
            flags.append(f"LI%={li:.1%} out of range [5–70%]")
        results.append({
            "sector": SECTOR_LABELS[s],
            "jpm":    jpm,
            "wage":   wage,
            "li":     li,
            "ok":     len(flags) == 0,
            "flags":  flags,
        })
    return results

# ── FULL BUILD PIPELINE ───────────────────────────────────────────────────────

def build_model(
    use_file,
    ms_file,
    dom_file,
    qcew_file,
    bea_api_key:        str,
    bls_api_key:        str,
    flq_delta:          float = 0.25,
    mpc:                float = 0.90,
    regional_retention: float = 0.50,
    progress_callback   = None,
) -> dict:
    """
    End-to-end model build from uploaded files.

    New parameters vs. v4:
      flq_delta          : FLQ sensitivity (0.10–0.35, default 0.25)
      mpc                : marginal propensity to consume (default 0.90)
      regional_retention : share of HH spending staying local (default 0.50)

    Returns model_state dict with pre-computed Leontief inverses
    and all intermediate results needed by the UI.
    """
    def _prog(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    _prog("Parsing BEA Use Table…", 5)
    use_data = parse_use_table(use_file)

    _prog("Parsing Market Share Matrix D…", 13)
    D_data = parse_D_matrix(ms_file)

    _prog("Parsing Direct Domestic Requirements…", 22)
    B_dom   = parse_B_domestic(dom_file)
    B_total = build_B_total(use_data)

    _prog("Building A_domestic = D @ B_domestic…", 32)
    A_data  = build_A_domestic(D_data["D"], B_dom, use_data)

    _prog("Computing sector crosswalk (402→15)…", 40)
    ind_sector = np.array([_code_to_sector(c) for c in D_data["ind_codes"]], dtype=int)
    com_sector = np.array([_code_to_sector(c) for c in use_data["com_codes"]], dtype=int)

    _prog("Aggregating A_domestic to 15×15…", 48)
    agg     = aggregate_matrix(A_data["A"], use_data["x"],
                               use_data["va_total"], use_data["va_comp"], ind_sector)
    A_nat15 = agg["A_agg"]

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
    bea_output, bea_output_src = _use_or_fallback(
        fetch_bea_table(bea_api_key, "5"),
        BEA_OUTPUT_FALLBACK, "bea_output.csv", min_plausible_sum=10_000_000)
    bea_comp, bea_comp_src = _use_or_fallback(
        fetch_bea_table(bea_api_key, "2"),
        BEA_COMP_FALLBACK, "bea_compensation.csv", min_plausible_sum=3_000_000)
    bea_va, bea_va_src = _use_or_fallback(
        fetch_bea_table(bea_api_key, "1"),
        BEA_VA_FALLBACK, "bea_value_added.csv", min_plausible_sum=5_000_000)

    _prog("Fetching BEA NIPA MPC…", 79)
    nat_mpc = fetch_bea_nipa_mpc(bea_api_key) or mpc

    _prog("Building employment coefficients…", 84)
    emp_coeffs = build_employment_coefficients(
        national_emp, bea_output, bea_comp, bea_va, qcew, agg["va_share"],
        mpc=nat_mpc, regional_retention=regional_retention,
    )

    _prog("Applying FLQ regionalization…", 88)
    reg = regionalize(A_nat15, qcew,
                      national_emp_total=national_emp.sum(),
                      delta=flq_delta)
    A_york   = reg["A_york"]
    lq_york  = reg["lq_york"]
    rpc_york = reg["rpc_york"]

    _prog("Pre-computing Leontief inverses…", 92)
    inverses = build_leontief_inverses(
        A_york, emp_coeffs["li_share"], pce_shares, emp_coeffs["hh_share"]
    )

    _prog("Pre-computing spending profiles…", 96)
    all_profiles = build_all_profiles(A_nat15)
    validation   = validate_coefficients(emp_coeffs)

    _prog("Model ready.", 100)

    import_removed_mean = float((B_total.sum(0) - B_dom.sum(0)).mean())

    # Leakage diagnostics
    rpc_slq = reg["rpc_slq"]
    import_leakage_rate     = float(1.0 - rpc_york.mean())
    import_leakage_rate_slq = float(1.0 - rpc_slq.mean())

    return {
        # matrices
        "A_nat15":     A_nat15,
        "A_york":      A_york,
        "lq_york":     lq_york,
        "rpc_york":    rpc_york,   # FLQ
        "rpc_slq":     rpc_slq,   # SLQ for comparison
        "flq_lambda":  reg["lambda"],
        "flq_delta":   flq_delta,
        "pce_shares":  pce_shares,
        # pre-computed inverses (reused per scenario)
        "inverses":    inverses,
        # coefficients
        "emp_coeffs":  emp_coeffs,
        "national_emp":national_emp,
        "bea_output":  bea_output,
        "bea_comp":    bea_comp,
        "bea_va":      bea_va,
        # metadata
        "qcew":            qcew,
        "all_profiles":    all_profiles,
        "validation":      validation,
        "nat_mpc":         nat_mpc,
        "regional_retention": regional_retention,
        "bea_output_src":  bea_output_src,
        "bea_comp_src":    bea_comp_src,
        "bea_va_src":      bea_va_src,
        "import_removed_mean":      import_removed_mean,
        "import_leakage_rate":      import_leakage_rate,
        "import_leakage_rate_slq":  import_leakage_rate_slq,
        "x_sum":           float(use_data["x"].sum()),
        "ind_sector":      ind_sector,
        "com_sector":      com_sector,
        "stability":       inverses["stability"],
    }
