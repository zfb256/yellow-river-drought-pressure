"""Homogeneity check of ERA5-Land precipitation against independent CPC gauge analysis (1979-2025)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_basins import basin_weights

ROOT = Path(__file__).resolve().parent.parent
CPC = ROOT / "data" / "cpc" / "monthly"
IDX = ROOT / "data" / "processed" / "drought_indices.csv"
OUT = ROOT / "data" / "processed" / "homogeneity_cpc.csv"


def main():
    cpc = xr.open_mfdataset(sorted(CPC.glob("cpc_precip_*.nc"))).precip.load()
    w = basin_weights(cpc.lat.values, cpc.lon.values)
    a = cpc.values
    times = pd.to_datetime(cpc.time.values)
    rows = []
    for bid, (iy, ix, ww) in w.items():
        sub = a[:, iy[:, None], ix[None, :]]
        m = np.isfinite(sub) * ww[None]
        val = np.nansum(sub * ww[None], (1, 2)) / np.where(m.sum((1, 2)) > 0, m.sum((1, 2)), np.nan)
        rows.append(pd.DataFrame({"basin": bid, "date": times, "cpc": val}))
    c = pd.concat(rows)

    d = pd.read_csv(IDX, dtype={"basin": str}, parse_dates=["date"], usecols=["basin", "date", "p_raw", "p"])
    d["basin"] = d.basin.str.zfill(4)
    j = d.merge(c, on=["basin", "date"]).dropna()

    ann = j.groupby(["basin", j.date.dt.year])[["p_raw", "p", "cpc"]].sum()
    ann = ann[j.groupby(["basin", j.date.dt.year]).size() == 12]
    ann["ratio"] = ann.p_raw / ann.cpc
    med = ann.groupby(level=1).median()
    med.index.name = "year"
    base = med.ratio.loc[1979:2020].median()
    med["ratio_dev"] = med.ratio / base - 1
    med.to_csv(OUT, float_format="%.4g")

    anom = j.set_index("date").groupby("basin")[["p_raw", "cpc"]].transform(
        lambda s: s - s.groupby(s.index.month).transform("mean"))
    r = anom.groupby(j.basin.values).apply(lambda g: g.p_raw.corr(g.cpc))
    print(f"monthly anomaly correlation ERA5-Land vs CPC, basin median: {r.median():.2f}")
    print(f"baseline ratio ERA5-Land/CPC (1979-2020): {base:.2f}")
    print("years with |ratio deviation| > 25%:")
    print(med[med.ratio_dev.abs() > 0.25][["p_raw", "cpc", "ratio", "ratio_dev"]].round(2))
    for col in ("p_raw", "p", "cpc"):
        k = np.polyfit(med.index, med[col], 1)[0] * 10
        print(f"trend 1979-2025 {col}: {k:+.1f} mm/decade")


if __name__ == "__main__":
    main()
