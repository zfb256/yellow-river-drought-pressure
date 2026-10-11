"""Area-weighted basin means of gridded products for the 102 HydroMLYR basins."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_basins import basin_weights

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CMFD_VARS = ("prec", "temp", "shum", "pres", "srad", "lrad", "wind")


def basin_means(da, lat_name, lon_name, weights=None):
    if weights is None:
        weights = basin_weights(da[lat_name].values, da[lon_name].values)
    a = da.values.astype("float64")
    out = {}
    for bid, (iy, ix, w) in weights.items():
        sub = a[:, iy[:, None], ix[None, :]]
        m = np.isfinite(sub) * w[None]
        den = m.sum((1, 2))
        out[bid] = np.nansum(sub * w[None], (1, 2)) / np.where(den > 0, den, np.nan)
    return out, weights


def gleam():
    da = xr.open_mfdataset(sorted((DATA / "gleam" / "SMrz").glob("SMrz_*_yrb.nc"))).SMrz.load()
    vals, _ = basin_means(da, "lat", "lon")
    times = pd.to_datetime(da.time.values).to_period("M").to_timestamp()
    df = pd.concat([pd.DataFrame({"basin": b, "date": times, "smrz": v}) for b, v in vals.items()])
    return df, DATA / "processed" / "gleam_basin_monthly.csv"


def cmfd():
    frames, weights = {}, None
    for v in CMFD_VARS:
        f = next((DATA / "cmfd" / "raw").glob(f"{v}_CMFD_*_01mo_*.nc"))
        ds = xr.open_dataset(f)
        lat = "lat" if "lat" in ds.coords else "latitude"
        lon = "lon" if "lon" in ds.coords else "longitude"
        lat_sl = slice(42.5, 31.5) if ds[lat][0] > ds[lat][-1] else slice(31.5, 42.5)
        da = ds[list(ds.data_vars)[0]].sel({lat: lat_sl, lon: slice(94.5, 120.5)}).load()
        vals, weights = basin_means(da, lat, lon, weights)
        times = pd.to_datetime(da.time.values).to_period("M").to_timestamp()
        frames[v] = pd.concat([pd.DataFrame({"basin": b, "date": times, v: x}) for b, x in vals.items()])
        print(v, da.attrs.get("units"), flush=True)
    df = frames[CMFD_VARS[0]]
    for v in CMFD_VARS[1:]:
        df = df.merge(frames[v], on=["basin", "date"])
    return df, DATA / "processed" / "cmfd_basin_monthly.csv"


if __name__ == "__main__":
    df, out = {"gleam": gleam, "cmfd": cmfd}[sys.argv[1]]()
    df.sort_values(["basin", "date"]).to_csv(out, index=False, float_format="%.6g")
    print(f"wrote {out}: {len(df)} rows, {df.basin.nunique()} basins, {df.date.min():%Y-%m} to {df.date.max():%Y-%m}")
