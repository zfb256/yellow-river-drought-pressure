"""Bias-corrected precipitation, FAO-56 PET and standardized drought indices per basin."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parent.parent
HYD = ROOT / "data" / "hydromlyr" / "1_data"
ERA = ROOT / "data" / "processed" / "basin_monthly.csv"
OUT = ROOT / "data" / "processed" / "drought_indices.csv"
SCALES = (1, 3, 6, 12)


def station_monthly(bid):
    # values are already mm/day, degC, m/s despite the "0.1" units in the description xlsx
    m = pd.read_csv(HYD / bid / "meteorological.txt", parse_dates=["date"]).set_index("date")
    return pd.DataFrame({
        "pre": m.pre.resample("MS").sum(min_count=25),
        "win": m.win_mean.resample("MS").mean(),
    })


def quantile_map(era, obs):
    out = era.copy()
    j = pd.concat([era.rename("e"), obs.rename("o")], axis=1).dropna()
    for mon in range(1, 13):
        months = [(mon - 2) % 12 + 1, mon, mon % 12 + 1]
        fit = j[j.index.month.isin(months)]
        e_q, o_q = np.sort(fit.e.values), np.sort(fit.o.values)
        target = era.index.month == mon
        x = era[target].values
        y = np.interp(x, e_q, o_q)
        lo_r = o_q[0] / e_q[0] if e_q[0] > 0 else 1.0
        hi_r = o_q[-1] / e_q[-1] if e_q[-1] > 0 else 1.0
        y = np.where(x < e_q[0], x * lo_r, y)
        y = np.where(x > e_q[-1], x * hi_r, y)
        out[target] = y
    return out.clip(lower=0)


def svp(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def fao56_pet(df, elev, u10_clim):
    t = df.t2m
    days = df.index.days_in_month
    delta = 4098 * svp(t) / (t + 237.3) ** 2
    p = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    gamma = 0.000665 * p
    es, ea = svp(t), svp(df.d2m)
    rn = (df.ssr + df.str) * 0.0864  # W/m2 -> MJ/m2/day
    g = 0.14 * t.diff().fillna(0)
    u2 = df.index.month.map(u10_clim).values * 4.87 / np.log(67.8 * 10 - 5.42)
    et0 = (0.408 * delta * (rn - g) + gamma * 900 / (t + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
    return (et0.clip(lower=0) * days).rename("pet")


def standardize(x, ref=None):
    """Non-parametric standardization per calendar month (Gringorten plotting position)."""
    z = pd.Series(np.nan, index=x.index)
    for mon in range(1, 13):
        s = x[(x.index.month == mon) & x.notna()]
        r = s.loc[ref[0]:ref[1]] if ref else s
        if len(r) < 10:
            continue
        if ref is None:
            rank = rankdata(s.values)
        else:
            sr = np.sort(r.values)
            rank = (np.searchsorted(sr, s.values, "left") + np.searchsorted(sr, s.values, "right")) / 2 + 0.5
        p = (rank - 0.44) / (len(r) + 0.12)
        z[s.index] = norm.ppf(p)
    return z


def main():
    era = pd.read_csv(ERA, dtype={"basin": str}, parse_dates=["date"])
    era["basin"] = era.basin.str.zfill(4)
    if "str" not in era:
        raise SystemExit("basin_monthly.csv has no 'str' column; download monthly_extra and re-run aggregate_basins.py")

    out = []
    for bid, e in era.groupby("basin"):
        e = e.set_index("date").sort_index()
        attr = json.load(open(HYD / bid / "attributes.json"))
        st = station_monthly(bid)
        wind_clim = st.win.groupby(st.index.month).mean()

        p = quantile_map(e.tp, st.pre)
        pet = fao56_pet(e, attr["elev"], wind_clim)
        sm = (7 * e.swvl1 + 21 * e.swvl2 + 72 * e.swvl3) / 100
        rec = pd.DataFrame({"basin": bid, "p_raw": e.tp, "p": p, "pet": pet, "ro": e.ro, "sm": sm})
        for k in SCALES:
            rec[f"spei{k}"] = standardize((p - pet).rolling(k).sum())
            rec[f"sri{k}"] = standardize(e.ro.rolling(k).sum())
            rec[f"ssmi{k}"] = standardize(sm.rolling(k).mean())
        out.append(rec)

    res = pd.concat(out).rename_axis("date").reset_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False, float_format="%.5g")

    ann = res.groupby(["basin", res.date.dt.year])[["p_raw", "p", "pet"]].sum()
    ann = ann[res.groupby(["basin", res.date.dt.year]).size() == 12]
    print(f"wrote {OUT}: {len(res)} rows")
    print("basin-median annual totals (mm):")
    print(ann.groupby("basin").mean().median().round(0))


if __name__ == "__main__":
    main()
