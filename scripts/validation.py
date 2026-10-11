"""Evaluation of forcing, soil moisture and runoff reported in Sections 2.2 and 2.3."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drought_indices import standardize

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
HYD = ROOT / "data" / "hydromlyr" / "1_data"


def month_anom(s):
    return s.groupby(s.index.month).transform(lambda x: (x - x.mean()) / x.std())


def read(name, cols=None):
    d = pd.read_csv(PROC / name, dtype={"basin": str}, parse_dates=["date"], usecols=cols)
    d["basin"] = d.basin.str.zfill(4)
    return d


def station(bid):
    m = pd.read_csv(HYD / bid / "meteorological.txt", parse_dates=["date"]).set_index("date").pre
    return m.resample("MS").sum(min_count=25), m


def observed_flow(bid):
    q = pd.read_csv(HYD / bid / "streamflow_raw.txt", parse_dates=["date"]).set_index("date").q
    qm = q.resample("MS").agg(["mean", "count"])
    return qm[qm["count"] >= 3]["mean"]


def main():
    v2 = read("drought_indices_v2.csv")
    era = read("drought_indices.csv", ["basin", "date", "p_raw", "sm"])
    cmfd = read("cmfd_basin_monthly.csv", ["basin", "date", "prec"])
    cmfd["p"] = cmfd.prec * 86400 * cmfd.date.dt.days_in_month

    ann, mon = {"station": {}, "cmfd": {}, "era5land": {}}, []
    sm, rq = [], []
    for bid, g in v2.groupby("basin"):
        g = g.set_index("date").sort_index()
        p_st, daily = station(bid)
        y = daily.resample("YS").agg(["sum", "count"])
        ann["station"][bid] = pd.Series(y[y["count"] >= 360]["sum"].values, index=y[y["count"] >= 360].index.year)
        pc = cmfd[cmfd.basin == bid].set_index("date").p
        pe = era[era.basin == bid].set_index("date").p_raw
        ann["cmfd"][bid] = pc.groupby(pc.index.year).sum()
        ann["era5land"][bid] = pe.groupby(pe.index.year).sum()
        j = pd.concat([p_st.rename("o"), pc.rename("c"), pe.rename("e")], axis=1).loc["2000":"2019"].dropna()
        dm = j - j.groupby(j.index.month).transform("mean")
        mon.append((dm.o.corr(dm.c), dm.o.corr(dm.e)))

        p6 = month_anom(p_st.rolling(6).sum())
        k = pd.concat([p6.rename("p"), month_anom(g.soil).rename("abcd"), month_anom(g.sm_gleam).rename("gleam"),
                       month_anom(era[era.basin == bid].set_index("date").sm).rename("era5land")], axis=1)
        k = k.loc["2000":"2019"].dropna()
        sm.append({c: k.p.corr(k[c]) for c in ("abcd", "gleam", "era5land")})

        qo = observed_flow(bid)
        jq = pd.concat([qo.rename("o"), month_anom(g.q).rename("abcd"), month_anom(g.q_grun).rename("grun")],
                       axis=1).loc["2010":"2019"].dropna()
        rq.append({c: jq.o.corr(jq[c], method="spearman") for c in ("abcd", "grun")})

    print("1. Precipitation vs station-based series, 2000-2019 (basin medians)")
    a = {k: pd.DataFrame(v).median(axis=1).loc[2000:2019] for k, v in ann.items()}
    yrs = a["station"].index
    for k in ("station", "cmfd", "era5land"):
        s = a[k].loc[yrs]
        print(f"  {k:9s} mean {s.mean():5.0f} mm  trend {np.polyfit(yrs, s.values, 1)[0] * 10:+6.1f} mm/decade"
              f"  r(annual) {s.corr(a['station']):.2f}")
    m = np.nanmedian(np.array(mon), axis=0)
    print(f"  monthly anomaly correlation: CMFD {m[0]:.2f}, ERA5-Land {m[1]:.2f}")

    kge = pd.read_csv(PROC / "abcd_params.csv").kge_cal
    print(f"\n2. ABCD calibration KGE (1961-2010): median {kge.median():.2f}, min {kge.min():.2f}")
    print("\n3. Runoff anomalies vs observed, 2010-2019 (Spearman, basin median):",
          pd.DataFrame(rq).median().round(2).to_dict())
    print("\n4. Soil-moisture anomalies vs 6-month station precipitation, 2000-2019 (basin median):",
          pd.DataFrame(sm).median().round(2).to_dict())

    print("\n5. Propagation time SPEI-n -> SRI-1, 1961-2019")
    for col in ("q", "q_grun"):
        pts = []
        for bid, g in v2.groupby("basin"):
            g = g.set_index("date").sort_index()
            sri1 = standardize(g[col]).loc["1961":"2019"]
            r = [standardize((g.p - g.pet).rolling(k).sum()).loc["1961":"2019"].corr(sri1) for k in range(1, 25)]
            pts.append(int(np.nanargmax(r)) + 1)
        pts = pd.Series(pts)
        print(f"  {'ABCD' if col == 'q' else 'G-RUN'}: median {pts.median():.0f} [{pts.quantile(.25):.0f}-"
              f"{pts.quantile(.75):.0f}] months, {pts.nunique()} distinct values over {len(pts)} basins")


if __name__ == "__main__":
    main()
