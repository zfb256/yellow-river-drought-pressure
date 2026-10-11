"""Basin-mean G-RUN ENSEMBLE median runoff, 1950-2019."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_basins import basin_weights

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "grun" / "G-RUN_ENSEMBLE_MMM.nc"
HYD = ROOT / "data" / "hydromlyr"
ERA = ROOT / "data" / "processed" / "basin_monthly.csv"
OUT = ROOT / "data" / "processed" / "grun_basin_monthly.csv"


def month_anom(s):
    return s.groupby(s.index.month).transform(lambda x: (x - x.mean()) / x.std())


def main():
    da = xr.open_dataset(SRC).Runoff.sel(X=slice(95, 120), Y=slice(32, 42), time=slice("1950", "2019")).load()
    w = basin_weights(da.Y.values, da.X.values)
    times = pd.to_datetime(da.time.values)
    a = da.values * times.days_in_month.values[:, None, None]
    rows = []
    for bid, (iy, ix, ww) in w.items():
        sub = a[:, iy[:, None], ix[None, :]]
        m = np.isfinite(sub) * ww[None]
        den = m.sum((1, 2))
        val = np.nansum(sub * ww[None], (1, 2)) / np.where(den > 0, den, np.nan)
        rows.append(pd.DataFrame({"basin": bid, "date": times, "ro_grun": val}))
    g = pd.concat(rows)
    g.to_csv(OUT, index=False, float_format="%.5g")

    era = pd.read_csv(ERA, dtype={"basin": str}, parse_dates=["date"], usecols=["basin", "date", "ro"])
    era["basin"] = era.basin.str.zfill(4)
    natural = set((HYD / "natural_basins.txt").read_text().split())
    res = []
    for bid, gg in g.groupby("basin"):
        q = pd.read_csv(HYD / "1_data" / bid / "streamflow_raw.txt", parse_dates=["date"]).set_index("date").q
        qm = q.resample("MS").agg(["mean", "count"])
        qm = qm[qm["count"] >= 3]["mean"]
        sim = pd.DataFrame({
            "grun": month_anom(gg.set_index("date").ro_grun),
            "era": month_anom(era[era.basin == bid].set_index("date").ro),
        })
        j = sim.join(qm.rename("obs"), how="inner").dropna()
        if len(j) < 24:
            continue
        res.append({"basin": bid, "natural": bid in natural, "n": len(j),
                    "r_grun": j.obs.corr(j.grun, method="spearman"),
                    "r_era": j.obs.corr(j.era, method="spearman")})
    r = pd.DataFrame(res)
    print(f"wrote {OUT}")
    print(f"{len(r)} basins with >=24 overlapping months (obs period 2000-2019)")
    print(r.groupby("natural")[["r_grun", "r_era"]].median().round(3))
    print("all basins median:", r[["r_grun", "r_era"]].median().round(3).to_dict())
    print("share of basins where G-RUN beats ERA5-Land:", round((r.r_grun > r.r_era).mean(), 2))


if __name__ == "__main__":
    main()
