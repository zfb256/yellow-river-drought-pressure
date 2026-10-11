"""Contributions of temperature, humidity, wind and radiation to the 1961-2024 PET trend, and a Pettitt test of the demand share."""
import json

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from counterfactual import remove_trend
from mechanism_analysis import PROC, ROOT
from pet_check import cmfd_variants, station_et0

DRIVERS = {"temperature": "temp", "specific_humidity": "shum", "wind": "wind", "shortwave_radiation": "srad"}


def annual_trend(series):
    a = series.loc["1961":"2024"].resample("YS").sum()
    return theilslopes(a.values, a.index.year)[0] * 10


def pettitt(x):
    """Pettitt change-point test: index of the change and approximate p value."""
    x = np.asarray(x)
    n = len(x)
    u = np.array([np.sign(x[k + 1:, None] - x[None, :k + 1]).sum() for k in range(n - 1)])
    k = int(np.argmax(np.abs(u)))
    return k, min(1.0, 2 * np.exp(-6 * u[k] ** 2 / (n ** 3 + n ** 2)))


def main():
    forcing = pd.read_csv(PROC / "cmfd_basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    forcing["basin"] = forcing.basin.str.zfill(4)
    rows = []
    for bid, c in forcing.groupby("basin"):
        c = c.set_index("date").sort_index()
        attr = json.loads((ROOT / "data/hydromlyr/1_data" / bid / "attributes.json").read_text())
        pet = cmfd_variants(c, attr["lat"], attr["elev"])["cmfd_fao_rnl"].clip(lower=0)
        row = {"basin": bid, "pet_trend": annual_trend(pet)}
        for name, col in DRIVERS.items():
            modified = c.copy()
            modified[col] = remove_trend(c[col])[0]
            row[f"{name}_contribution"] = row["pet_trend"] - annual_trend(
                cmfd_variants(modified, attr["lat"], attr["elev"])["cmfd_fao_rnl"].clip(lower=0))
            row[f"{name}_driver_trend"] = theilslopes(c[col].loc["1961":"2024"].resample("YS").mean().values,
                                                      np.arange(1961, 2025))[0] * 10
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(PROC / "pet_driver_contributions.csv", index=False)
    print("Basin-median trends per decade:\n" + r.drop(columns="basin").median().round(3).to_string())
    station = []
    for bid, c in forcing.groupby("basin"):
        attr = json.loads((ROOT / "data/hydromlyr/1_data" / bid / "attributes.json").read_text())
        v = cmfd_variants(c.set_index("date").sort_index(), attr["lat"], attr["elev"])["cmfd_fao_rnl"].loc["2000":"2019"]
        j = pd.concat([v.rename("cmfd"), station_et0(bid).rename("station")], axis=1).dropna()
        a = j.resample("YS").sum()[j.resample("YS").size() == 12]
        if len(a) >= 15:
            station.append({"basin": bid, **{k: theilslopes(a[k].values, a.index.year)[0] * 10 for k in ("cmfd", "station")}})
    station = pd.DataFrame(station)
    station.to_csv(PROC / "pet_trend_station_check.csv", index=False)
    print("2000-2019 PET trend per decade, basin median:", station[["cmfd", "station"]].median().round(2).to_dict())
    checks = pd.read_csv(PROC / "demand_share_checks_annual.csv")
    trend = pd.read_csv(PROC / "mechanism_annual.csv")
    series = {"spei3_events": trend[trend.reference == "1961-1990"].set_index("year").demand_share,
              **{k: g.set_index("year").demand_share for k, g in checks.groupby("check")}}
    out = []
    for name, s in series.items():
        s = s.sort_index()
        k, p = pettitt(s.values)
        out.append({"series": name, "change_year": int(s.index[k + 1]), "p_value": p,
                    "mean_before": s.iloc[:k + 1].mean(), "mean_after": s.iloc[k + 1:].mean()})
    pd.DataFrame(out).to_csv(PROC / "demand_share_changepoints.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
