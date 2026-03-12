"""
model_engine.py
───────────────
All pure-computation functions for the Regional Economic Impact Model v4.
No Streamlit imports, no print statements, no side effects.
Designed to be called from the Streamlit UI layer (app.py).

Data flow:
  1. parse_*()      — read uploaded BEA/QCEW Excel/CSV files → numpy arrays
  2. build_*()      — assemble A_domestic, aggregate to 15×15
  3. fetch_*()      — pull live BLS/BEA API data (with fallbacks)
  4. build_emp_*()  — employment + wage coefficients
  5. regionalize()  — apply QCEW LQ regionalization → A_york
  6. compute_impacts() — Leontief Type I + II solve
  7. get_spending_profile() — BEA-derived sector spending shares
"""

import io
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import openpyxl
import requests

warnings.filterwarnings("ignore")

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
    if   p2 == "11":               return 0
    elif p2 in ("21","22"):        return 1
    elif p2 == "23":               return 2
    elif p2 in ("31","32","33"):   return 4 if p3 in _NONDUR else 3
    elif p2 == "42":               return 5
    elif p2 in ("44","45"):        return 6
    elif p2 in ("48","49"):        return 7
    elif p2 == "51":               return 8
    elif p2 in ("52","53"):        return 9
    elif p2 in ("54","55"):        return 10
    elif p2 == "56":               return 11
    elif p2 in ("61","62"):        return 12
    elif p2 in ("71","72"):        return 13
    else:                          return 14

# ── FILE PARSERS ──────────────────────────────────────────────────────────────

def parse_use_table(src, sheet="2017") -> dict:
    """Parse BEA Use Table. Returns Z, x, va arrays, ind/com codes, pce."""
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
        "Z": Z,
        "x": _arr(413),
        "va_comp":   _arr(409),
        "va_taxes":  _arr(410),
        "va_gos":    _arr(411),
        "va_total":  _arr(412),
        "ind_codes": [str(hdr_codes[c]) for c in range(C0,C1)],
        "ind_names": [str(hdr_names[c]) for c in range(C0,C1)],
        "com_codes": [str(rows[r][0])   for r in range(R0,R1)],
        "pce": np.array([rows[r][PCE_COL] or 0.0 for r in range(R0,R1)], dtype=float),
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
    Returns emp, lq, wage arrays (length N=15) and year.
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
                lq_n[s_idx]   = (df_sub["lq_annual_avg_emplvl"]*df_sub["annual_avg_emplvl"]).sum()
                wage_n[s_idx] = (df_sub["avg_annual_pay"]       *df_sub["annual_avg_emplvl"]).sum()

    # Government employment + wages
    gov_rows   = df[df["own_code"].isin([1,2,3]) & (df["agglvl_code"]==71)]
    total_all  = df[(df["own_code"]==0) & (df["agglvl_code"]==70)]["annual_avg_emplvl"]
    total_priv = df[(df["own_code"]==5) & (df["agglvl_code"]==71)]["annual_avg_emplvl"]
    if len(total_all) and len(total_priv):
        gov_emp_net = max(float(total_all.values[0])-float(total_priv.values[0]), 0)
        if gov_emp_net > 0 and len(gov_rows):
            gov_wages_total = float(gov_rows["total_annual_wages"].sum())
            gov_emp_total   = float(gov_rows["annual_avg_emplvl"].sum())
            scale = gov_emp_net / max(gov_emp_total, 1)
            emp[14]    += gov_emp_net
            wage_n[14] += gov_wages_total * scale

    lq   = np.where(emp>0, lq_n  /emp, 0.0)
    wage = np.where(emp>0, wage_n /emp, 0.0)
    fips = str(df["area_fips"].iloc[0])
    return {"emp": emp, "lq": lq, "wage": wage, "year": year, "fips": fips}

# ── MATRIX BUILDERS ───────────────────────────────────────────────────────────

def build_A_domestic(D: np.ndarray, B_dom: np.ndarray, use_data: dict) -> dict:
    """Compute A_domestic = D_redef @ B_domestic. Cap columns ≥ 1.0."""
    A = D @ B_dom
    col_sums = A.sum(axis=0)
    bad = col_sums >= 1.0
    if bad.any():
        A[:, bad] *= 0.99 / col_sums[bad]
    with np.errstate(divide="ignore", invalid="ignore"):
        va_share = np.where(use_data["x"]>0, use_data["va_total"]/use_data["x"], 0.0)
        li_share = np.where(use_data["x"]>0, use_data["va_comp"] /use_data["x"], 0.0)
    return {"A": A, "va_share": va_share, "li_share": li_share}


def build_B_total(use_data: dict) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(use_data["x"] > 0,
                        use_data["Z"] / use_data["x"][np.newaxis,:], 0.0)


def aggregate_matrix(A: np.ndarray, x: np.ndarray,
                     va_total: np.ndarray, va_comp: np.ndarray,
                     sector_map: np.ndarray) -> dict:
    """Aggregate 402×402 A to N×N via output-weighted averaging."""
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
        t_cols = np.where(sector_map==t)[0]
        if not len(t_cols): continue
        x_t = x[t_cols]; tot = x_t.sum()
        if tot <= 0: continue
        for s in range(N):
            s_rows = np.where(sector_map==s)[0]
            if len(s_rows):
                A_agg[s,t] = A[np.ix_(s_rows,t_cols)].sum(0).dot(x_t)/tot
    with np.errstate(divide="ignore", invalid="ignore"):
        va_share = np.where(x_agg>0, va_agg/x_agg,   0.0)
        li_share = np.where(x_agg>0, comp_agg/x_agg, 0.0)
    return {"A_agg": A_agg, "x_agg": x_agg,
            "va_share": va_share, "li_share": li_share}


def build_pce_shares(pce_raw: np.ndarray, com_sector: np.ndarray) -> np.ndarray:
    pce_agg = np.zeros(N)
    for i, s in enumerate(com_sector):
        if 0 <= s < N:
            pce_agg[s] += max(pce_raw[i], 0.0)
    total = pce_agg.sum()
    return pce_agg/total if total > 0 else np.ones(N)/N

# ── BLS / BEA API ─────────────────────────────────────────────────────────────

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
AGR_NATIONAL_EMP_FALLBACK = 2109.9  # thousands

BLS_FALLBACK_2022 = {
    "CEU1021000001":657.5,  "CEU0622000001":574.0,
    "CEU2000000001":7697.2, "CEU3100000001":9122.5, "CEU3200000001":4841.6,
    "CEU4142000001":6153.4, "CEU4200000001":15592.8,"CEU4300000001":6622.7,
    "CEU5000000001":3038.9, "CEU5552000001":6617.6, "CEU5553000001":2394.6,
    "CEU6054000001":10199.4,"CEU6055000001":2906.5, "CEU6056000001":9628.6,
    "CEU6561000001":3787.0, "CEU6562000001":20411.9,"CEU7071000001":2615.7,
    "CEU7072000001":12346.2,"CEU8000000001":5697.4, "CEU9000000001":22524.2,
}

BEA_OUTPUT_FALLBACK = np.array([
    464200,1031800,1872400,3421600,3190200,2391700,2088400,
    1512300,2155700,7321500,3289100,1256100,3298400,1702800,5158300
], dtype=float)
BEA_COMP_FALLBACK = np.array([
    58200,174700,617400,841800,402700,618800,686900,473100,
    426800,1063500,1562300,438100,1352300,578800,2572600
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


def fetch_bls(api_key: str, year: str = "2022") -> dict | None:
    payload = {"seriesid": ALL_SERIES_IDS, "startyear": year, "endyear": year,
               "annualaverage": "true", "registrationkey": api_key}
    try:
        r = requests.post(BLS_BASE, json=payload, timeout=45)
        r.raise_for_status()
        resp = r.json()
        if resp.get("status") != "REQUEST_SUCCEEDED":
            return None
        raw = {}
        for series in resp["Results"]["series"]:
            sid = series["seriesID"]
            val = next((float(d["value"]) for d in series["data"] if d["period"]=="M13"), None)
            if val is None:
                monthly = [float(d["value"]) for d in series["data"]
                           if d["period"].startswith("M") and d["period"]!="M13"]
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
    params = {"UserID": api_key, "method": "GetData",
              "datasetname": "GDPbyIndustry", "TableID": table_id,
              "Frequency": "A", "Year": year, "Industry": "ALL", "ResultFormat": "JSON"}
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
    params = {"UserID": api_key, "method": "GetData", "datasetname": "NIPA",
              "TableName": "T20600", "Frequency": "A", "Year": year, "ResultFormat": "JSON"}
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


def _use_or_fallback(result, fallback: np.ndarray, min_plausible_sum: float | None = None):
    if result is None or result.sum() == 0:
        return fallback, "fallback"
    if min_plausible_sum is not None and result.sum() < min_plausible_sum:
        return fallback, "fallback_partial"
    return result, "live"

# ── EMPLOYMENT COEFFICIENTS ───────────────────────────────────────────────────

def build_employment_coefficients(
    national_emp: np.ndarray,
    bea_output: np.ndarray,
    bea_comp: np.ndarray,
    qcew: dict,
    va_share_agg: np.ndarray,
    regional_retention: float = 0.70,
    nat_mpc: float = 0.960,
) -> dict:
    """
    Build employment and income coefficient vectors.
    All unit conversions documented inline.

    national_emp : thousands of workers (BLS CES)
    bea_output   : $M (BEA Table 5)
    bea_comp     : $M (BEA Table 2)
    qcew['wage'] : $/worker/year (BLS QCEW avg_annual_pay)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # workers / $M output = jobs per $1M
        jobs_per_million = np.where(bea_output > 0,
                                    (national_emp * 1000) / bea_output, 0.0)
        # ($M comp * 1e6 $/$M) / (k_workers * 1000 workers/k) = $/worker
        nat_avg_wage = np.where(national_emp > 0,
                                (bea_comp * 1e6) / (national_emp * 1000), 0.0)
        # BEA labor income share: $M/$M = dimensionless
        li_share_bea = np.where(bea_output > 0, bea_comp / bea_output, 0.0)

    # Use York County wages where QCEW has data, else national average
    york_wage = np.where(qcew["emp"] > 0, qcew["wage"], nat_avg_wage)

    # York LI share: ($/worker * k_workers * 1000) / ($M * 1e6) = dimensionless
    york_li_share = np.where(
        bea_output > 0,
        (york_wage * national_emp * 1000) / (bea_output * 1e6),
        li_share_bea
    )
    york_li_share = np.clip(york_li_share, 0.0, 0.70)

    hh_share = nat_mpc * regional_retention

    return {
        "jobs_per_million": jobs_per_million,
        "avg_wage":         york_wage,
        "li_share":         york_li_share,
        "va_share":         va_share_agg,
        "hh_share":         hh_share,
        "nat_avg_wage":     nat_avg_wage,
        "nat_mpc":          nat_mpc,
        "regional_retention": regional_retention,
    }

# ── REGIONALIZATION ───────────────────────────────────────────────────────────

def regionalize(A_nat15: np.ndarray, qcew: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply QCEW LQ-based RPCs to produce York County A matrix."""
    lq_york  = qcew["lq"].copy()
    rpc_york = np.minimum(lq_york, 1.0)
    A_york   = A_nat15 * rpc_york[:, np.newaxis]
    return A_york, lq_york, rpc_york

# ── LEONTIEF ENGINE ───────────────────────────────────────────────────────────

def leontief_type1(A: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.eye(len(A)) - A)


def leontief_type2(A: np.ndarray, li_share: np.ndarray,
                   pce_shares: np.ndarray, hh_share: float) -> np.ndarray:
    n = len(A)
    A2 = np.zeros((n+1, n+1))
    A2[:n,:n] = A
    A2[:n, n] = pce_shares * hh_share
    A2[ n,:n] = li_share
    return np.linalg.inv(np.eye(n+1) - A2)


def compute_impacts(Y: np.ndarray, A_reg: np.ndarray,
                    emp_coeffs: dict, pce_shares: np.ndarray,
                    ind_damp: float = 0.50) -> dict:
    """
    Compute direct, indirect, induced, and total economic impacts.

    Y           : final demand vector ($)
    A_reg       : regionalized A matrix (N×N)
    emp_coeffs  : from build_employment_coefficients()
    pce_shares  : PCE expenditure shares (N,)
    ind_damp    : induced dampening scalar (0.35–0.65)
    """
    n  = len(Y)
    I  = np.eye(n)
    L1 = leontief_type1(A_reg)
    L2 = leontief_type2(A_reg, emp_coeffs["li_share"],
                        pce_shares, emp_coeffs["hh_share"])[:n,:n]

    direct   = Y.copy()
    indirect = np.maximum((L1 - I) @ Y, 0.0)
    induced  = np.maximum((L2 - L1) @ Y * ind_damp, 0.0)
    total    = direct + indirect + induced

    def _imp(vec):
        vm = vec / 1e6
        return {
            "output":       float(vec.sum()),
            "jobs":         float((vm * emp_coeffs["jobs_per_million"]).sum()),
            "labor_income": float((vec * emp_coeffs["li_share"]).sum()),
            "value_added":  float((vec * emp_coeffs["va_share"]).sum()),
            "output_vec":   vec.copy(),
            "jobs_vec":     (vm * emp_coeffs["jobs_per_million"]).copy(),
            "li_vec":       (vec * emp_coeffs["li_share"]).copy(),
            "va_vec":       (vec * emp_coeffs["va_share"]).copy(),
        }

    d, ii, ind, t = _imp(direct), _imp(indirect), _imp(induced), _imp(total)

    def _m(a, b): return a/b if b > 0 else 0.0
    mults = {
        "type1": _m(d["output"] + ii["output"], d["output"]),
        "type2": _m(t["output"],  d["output"]),
        "emp":   _m(t["jobs"],    max(d["jobs"],  1e-9)),
        "li":    _m(t["labor_income"], max(d["labor_income"], 1e-9)),
        "va":    _m(t["value_added"],  max(d["value_added"],  1e-9)),
    }
    return {
        "direct": d, "indirect": ii, "induced": ind, "total": t,
        "multipliers": mults,
        "vecs": {"direct": direct, "indirect": indirect,
                 "induced": induced, "total": total},
    }

# ── SPENDING PROFILES ─────────────────────────────────────────────────────────

def get_spending_profile(A_nat15: np.ndarray, sector_idx: int) -> dict:
    """
    Return BEA-derived normalized spending distribution for a sector.
    Used to split investment $ across the 15-sector Y vector.
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
    Return per-sector validation results.
    Each entry: {sector, jpm, wage, li, ok, flags}
    """
    results = []
    for s in range(N):
        jpm  = float(emp_coeffs["jobs_per_million"][s])
        wage = float(emp_coeffs["avg_wage"][s])
        li   = float(emp_coeffs["li_share"][s])
        flags = []
        if not (0.2 <= jpm <= 30.0):  flags.append(f"jobs/$1M={jpm:.2f} out of range [0.2–30]")
        if not (15_000 <= wage <= 200_000): flags.append(f"wage=${wage:,.0f} out of range")
        if not (0.05 <= li <= 0.70):  flags.append(f"LI%={li:.1%} out of range [5–70%]")
        results.append({"sector": SECTOR_LABELS[s], "jpm": jpm,
                         "wage": wage, "li": li,
                         "ok": len(flags) == 0, "flags": flags})
    return results

# ── FULL BUILD PIPELINE ───────────────────────────────────────────────────────

def build_model(use_file, ms_file, dom_file, qcew_file,
                bea_api_key: str, bls_api_key: str,
                progress_callback=None) -> dict:
    """
    End-to-end model build from uploaded files.
    progress_callback(step: str, pct: int) called at each stage if provided.

    Returns a model_state dict containing everything needed to run scenarios.
    """
    def _prog(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    _prog("Parsing BEA Use Table…", 5)
    use_data = parse_use_table(use_file)

    _prog("Parsing Market Share Matrix D…", 15)
    D_data = parse_D_matrix(ms_file)

    _prog("Parsing Direct Domestic Requirements…", 25)
    B_dom   = parse_B_domestic(dom_file)
    B_total = build_B_total(use_data)

    _prog("Building A_domestic = D @ B_domestic…", 35)
    A_data  = build_A_domestic(D_data["D"], B_dom, use_data)

    _prog("Computing sector crosswalk (402→15)…", 42)
    ind_sector = np.array([_code_to_sector(c) for c in D_data["ind_codes"]], dtype=int)
    com_sector = np.array([_code_to_sector(c) for c in use_data["com_codes"]], dtype=int)

    _prog("Aggregating A_domestic to 15×15…", 50)
    agg = aggregate_matrix(A_data["A"], use_data["x"],
                           use_data["va_total"], use_data["va_comp"], ind_sector)
    A_nat15 = agg["A_agg"]

    _prog("Building PCE shares…", 55)
    pce_shares = build_pce_shares(use_data["pce"], com_sector)

    _prog("Loading QCEW data…", 60)
    qcew = load_qcew(qcew_file)

    _prog("Fetching BLS national employment…", 68)
    bls_raw = fetch_bls(bls_api_key) or {}
    for sid in ALL_SERIES_IDS:
        if sid not in bls_raw:
            bls_raw[sid] = BLS_FALLBACK_2022.get(sid, 0.0)
    national_emp = series_to_emp(bls_raw)

    _prog("Fetching BEA gross output & compensation…", 76)
    bea_output_raw = fetch_bea_table(bea_api_key, "5")
    bea_comp_raw   = fetch_bea_table(bea_api_key, "2")
    bea_output, bea_output_src = _use_or_fallback(bea_output_raw, BEA_OUTPUT_FALLBACK, 10_000_000)
    bea_comp,   bea_comp_src   = _use_or_fallback(bea_comp_raw,   BEA_COMP_FALLBACK,   3_000_000)

    _prog("Fetching BEA NIPA MPC…", 82)
    nat_mpc = fetch_bea_nipa_mpc(bea_api_key) or 0.960

    _prog("Building employment coefficients…", 87)
    emp_coeffs = build_employment_coefficients(
        national_emp, bea_output, bea_comp, qcew, agg["va_share"],
        regional_retention=0.70, nat_mpc=nat_mpc
    )

    _prog("Regionalizing A matrix with QCEW LQs…", 92)
    A_york, lq_york, rpc_york = regionalize(A_nat15, qcew)

    _prog("Pre-computing spending profiles…", 96)
    all_profiles = build_all_profiles(A_nat15)
    validation   = validate_coefficients(emp_coeffs)

    _prog("Model ready.", 100)

    import_removed_mean = float((B_total[:402,:].sum(0) - B_dom.sum(0)).mean())

    return {
        # matrices
        "A_nat15":     A_nat15,
        "A_york":      A_york,
        "lq_york":     lq_york,
        "rpc_york":    rpc_york,
        "pce_shares":  pce_shares,
        # coefficients
        "emp_coeffs":  emp_coeffs,
        "national_emp":national_emp,
        "bea_output":  bea_output,
        "bea_comp":    bea_comp,
        # metadata
        "qcew":        qcew,
        "all_profiles":all_profiles,
        "validation":  validation,
        "nat_mpc":     nat_mpc,
        "bea_output_src": bea_output_src,
        "bea_comp_src":   bea_comp_src,
        "import_removed_mean": import_removed_mean,
        "x_sum":       float(use_data["x"].sum()),
        "ind_sector":  ind_sector,
        "com_sector":  com_sector,
    }
