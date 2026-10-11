"""Drought indices from homogeneous station-based forcing (CMFD v2.0, 1951-2024)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abcd import calibrate, simulate
from drought_indices import standardize
from pet_check import cmfd_variants

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
HYD = ROOT / "data" / "hydromlyr" / "1_data"
SCALES = (1, 3, 6, 12)


def cmfd_inputs(c, lat, elev):
    days = c.index.days_in_month.values
    return pd.DataFrame({
        "p": c.prec * 86400 * days,
        "t": c.temp - 273.15,
        "pet": cmfd_variants(c, lat, elev)["cmfd_fao_rnl"].clip(lower=0),
    }, index=c.index)


def observed_runoff(bid):
    """Monthly observed runoff anomalies for 2000-2009 (at least 3 weekly values per month)."""
    q = pd.read_csv(HYD / bid / "streamflow_raw.txt", parse_dates=["date"]).set_index("date").q
    qm = q.resample("MS").agg(["mean", "count"])
    qm = qm[qm["count"] >= 3]["mean"].loc["2000":"2009"]
    return qm.groupby(qm.index.month).transform(lambda s: (s - s.mean()) / s.std())


def main():
    cmfd = pd.read_csv(PROC / "cmfd_basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    grun = pd.read_csv(PROC / "grun_basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    gleam = pd.read_csv(PROC / "gleam_basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    for df in (cmfd, grun, gleam):
        df["basin"] = df.basin.str.zfill(4)

    out, params = [], []
    for bid, c in cmfd.groupby("basin"):
        c = c.set_index("date").sort_index()
        attr = json.load(open(HYD / bid / "attributes.json"))
        x = cmfd_inputs(c, attr["lat"], attr["elev"])
        q_grun = grun[grun.basin == bid].set_index("date").ro_grun.reindex(x.index)
        cal = (x.index.year >= 1961) & (x.index.year <= 2010)
        q_obs = observed_runoff(bid).reindex(x.index)
        par, score = calibrate(x.p.values[cal], x.pet.values[cal], x.t.values[cal], q_grun.values[cal],
                               month=x.index.month.values[cal], q_anom_obs=q_obs.values[cal])
        q, soil, gw, et = simulate(par, x.p.values, x.pet.values, x.t.values)
        x["q"], x["soil"], x["et"] = q, soil, et
        params.append({"basin": bid, "a": par[0], "b": par[1], "c": par[2], "d": par[3], "kge_cal": score,
                       "p_mm_yr": x.p.mean() * 12, "pet_mm_yr": x.pet.mean() * 12, "q_mm_yr": q.mean() * 12})

        rec = x.copy()
        rec["basin"] = bid
        rec["q_grun"] = q_grun
        rec["sm_gleam"] = gleam[gleam.basin == bid].set_index("date").smrz.reindex(x.index)
        for k in SCALES:
            rec[f"spei{k}"] = standardize((x.p - x.pet).rolling(k).sum())
            rec[f"sri{k}"] = standardize(rec.q.rolling(k).sum())
            rec[f"ssmi{k}"] = standardize(rec.soil.rolling(k).mean())
            rec[f"sri{k}_grun"] = standardize(rec.q_grun.rolling(k).sum())
            rec[f"ssmi{k}_gleam"] = standardize(rec.sm_gleam.rolling(k).mean())
        out.append(rec)
        print(bid, f"KGE={score:.2f}", flush=True)

    res = pd.concat(out).rename_axis("date").reset_index()
    res.to_csv(PROC / "drought_indices_v2.csv", index=False, float_format="%.5g")
    pr = pd.DataFrame(params)
    pr.to_csv(PROC / "abcd_params.csv", index=False, float_format="%.5g")
    print(pr.describe().round(2).T[["mean", "50%", "min", "max"]])


if __name__ == "__main__":
    main()
