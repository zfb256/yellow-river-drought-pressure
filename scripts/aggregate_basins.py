"""Area-weighted basin means of ERA5-Land monthly fields for the 102 HydroMLYR basins."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import contains_xy

ROOT = Path(__file__).resolve().parent.parent
ERA_DIR = ROOT / "data" / "era5land" / "monthly"
EXTRA_DIR = ROOT / "data" / "era5land" / "monthly_extra"
BASIN_DIR = ROOT / "data" / "hydromlyr" / "0_basin_boundary"
OUT = ROOT / "data" / "processed" / "basin_monthly.csv"

ACCUM = ["tp", "e", "pev", "ro", "sro", "ssro"]
SUB = 5  # sub-samples per grid cell edge for fractional coverage


def basin_weights(lat, lon):
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    offs = (np.arange(SUB) + 0.5) / SUB - 0.5
    weights = {}
    for shp in sorted(BASIN_DIR.glob("*.shp")):
        geom = gpd.read_file(shp).union_all()
        x0, y0, x1, y1 = geom.bounds
        iy = np.where((lat >= y0 - dlat) & (lat <= y1 + dlat))[0]
        ix = np.where((lon >= x0 - dlon) & (lon <= x1 + dlon))[0]
        yy, xx = np.meshgrid(lat[iy], lon[ix], indexing="ij")
        frac = np.zeros(yy.shape)
        for oy in offs:
            for ox in offs:
                frac += contains_xy(geom, xx + ox * dlon, yy + oy * dlat)
        frac /= SUB * SUB
        w = frac * np.cos(np.deg2rad(yy))
        weights[shp.stem] = (iy, ix, w)
    return weights


def main():
    files = sorted(ERA_DIR.glob("era5land_monthly_*.nc"))
    if not files:
        raise SystemExit(f"no files in {ERA_DIR}")
    first = xr.open_dataset(files[0])
    weights = basin_weights(first.latitude.values, first.longitude.values)

    rows = []
    for f in files:
        ds = xr.open_dataset(f)
        extra = EXTRA_DIR / f.name
        if extra.exists():
            ds = xr.merge([ds, xr.open_dataset(extra)], compat="override")
        times = pd.to_datetime(ds.valid_time.values)
        days = times.days_in_month.values
        fields = {}
        for v in ds.data_vars:
            if ds[v].dims[-2:] != ("latitude", "longitude"):
                continue
            a = ds[v].values.astype("float64")
            if v in ACCUM:
                a = a * 1000.0 * days[:, None, None]
                if v in ("e", "pev"):
                    a = -a
            elif v in ("t2m", "d2m"):
                a = a - 273.15
            elif v in ("ssr", "str"):
                a = a / 86400.0
            fields[v] = a
        for bid, (iy, ix, w) in weights.items():
            rec = {"basin": bid, "date": times}
            for v, a in fields.items():
                sub = a[:, iy[:, None], ix[None, :]]
                m = np.isfinite(sub)
                ww = w[None] * m
                rec[v] = np.where(ww.sum((1, 2)) > 0, np.nansum(sub * w[None], (1, 2)) / ww.sum((1, 2)), np.nan)
            rows.append(pd.DataFrame(rec))
        print(f.name, flush=True)

    out = pd.concat(rows).sort_values(["basin", "date"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, float_format="%.5g")
    print(f"wrote {OUT}: {len(out)} rows, {out.basin.nunique()} basins")


if __name__ == "__main__":
    main()
