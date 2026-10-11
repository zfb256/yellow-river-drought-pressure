"""Compare CMFD-based PET variants with FAO-56 PET from HydroMLYR station data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HYD = ROOT / "data" / "hydromlyr" / "1_data"
SIGMA = 4.903e-9


def svp(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def ra_n(lat_deg, doy):
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return ra, 24 / np.pi * ws


def pm(t, es, ea, rn, u2, p_kpa, g=0.0):
    delta = 4098 * svp(t) / (t + 237.3) ** 2
    gamma = 0.000665 * p_kpa
    return (0.408 * delta * (rn - g) + gamma * 900 / (t + 273) * u2 * np.clip(es - ea, 0, None)) / (
        delta + gamma * (1 + 0.34 * u2))


def station_et0(bid):
    m = pd.read_csv(HYD / bid / "meteorological.txt", parse_dates=["date"]).set_index("date").loc["2000":"2019"]
    attr = json.load(open(HYD / bid / "attributes.json"))
    doy = m.index.dayofyear.values
    ra, n_max = ra_n(attr["lat"], doy)
    rs = (0.25 + 0.5 * np.clip(m.ssd / n_max, 0, 1)) * ra
    rso = (0.75 + 2e-5 * attr["elev"]) * ra
    es = (svp(m.tem_max) + svp(m.tem_min)) / 2
    ea = es * m.rhu / 10  # rhu is stored in units of 10 % (range ~1-9.3)
    rnl = SIGMA * ((m.tem_max + 273.16) ** 4 + (m.tem_min + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(ea)) * (
        1.35 * np.clip(rs / rso, 0, 1) - 0.35)
    rn = 0.77 * rs - rnl
    u2 = m.win_mean * 4.87 / np.log(67.8 * 10 - 5.42)
    et0 = pm(m.tem_mean, es, ea, rn, u2, m.prs_mean / 10)
    return et0.resample("MS").sum(min_count=25)


def cmfd_variants(c, lat, elev):
    days = c.index.days_in_month.values
    t = c.temp - 273.15
    p = c.pres / 1000
    ea = c.shum * p / (0.622 + 0.378 * c.shum)
    es = svp(t)
    rs = c.srad * 0.0864
    doy = (c.index + pd.Timedelta(days=14)).dayofyear.values
    ra, _ = ra_n(lat, doy)
    rso = (0.75 + 2e-5 * elev) * ra
    u2 = c.wind * 4.87 / np.log(67.8 * 10 - 5.42)
    g = 0.14 * t.diff().fillna(0)
    rnl_fao = SIGMA * (t + 273.16) ** 4 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.clip(rs / rso, 0, 1) - 0.35)
    rn_fao = 0.77 * rs - rnl_fao
    rn_ld = 0.77 * rs + 0.98 * (c.lrad * 0.0864 - SIGMA * (t + 273.15) ** 4)
    return pd.DataFrame({
        "cmfd_fao_rnl": pm(t, es, ea, rn_fao, u2, p, g) * days,
        "cmfd_ld": pm(t, es, ea, rn_ld, u2, p, g) * days,
    }, index=c.index)


def main():
    cm = pd.read_csv(ROOT / "data/processed/cmfd_basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    cm["basin"] = cm.basin.str.zfill(4)
    rows = []
    for bid, c in cm.groupby("basin"):
        attr = json.load(open(HYD / bid / "attributes.json"))
        v = cmfd_variants(c.set_index("date"), attr["lat"], attr["elev"]).loc["2000":"2019"]
        s = station_et0(bid).rename("station")
        j = v.join(s, how="inner").dropna()
        a = j.resample("YS").sum()
        a = a[j.resample("YS").size() == 12]
        rows.append({"basin": bid, **{k: a[k].mean() for k in a},
                     **{f"r_{k}": j[k].corr(j.station) for k in v}})
    r = pd.DataFrame(rows)
    print(r.drop(columns="basin").median().round(2))


if __name__ == "__main__":
    main()
