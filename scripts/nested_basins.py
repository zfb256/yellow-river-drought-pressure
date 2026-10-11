"""Select catchments whose areas do not overlap, keeping the smaller catchment of any overlapping pair."""
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HYD = ROOT / "data" / "hydromlyr"


def main():
    shp = sorted((HYD / "0_basin_boundary").glob("*.shp"))
    g = gpd.GeoDataFrame(pd.concat([gpd.read_file(f).assign(basin=f.stem) for f in shp]), crs="EPSG:4326")
    g = g.dissolve("basin").reset_index().to_crs("EPSG:6933")
    g["area"] = g.area
    kept = []
    for _, row in g.sort_values("area").iterrows():
        if all(row.geometry.intersection(k.geometry).area < 0.05 * min(row.area, k.area) for k in kept):
            kept.append(row)
    out = pd.DataFrame({"basin": [k.basin for k in kept], "area_km2": [k.area / 1e6 for k in kept]})
    out.to_csv(ROOT / "data/processed/nonnested_basins.csv", index=False)
    print(f"{len(out)} non-overlapping catchments covering {out.area_km2.sum():,.0f} km2")


if __name__ == "__main__":
    main()
