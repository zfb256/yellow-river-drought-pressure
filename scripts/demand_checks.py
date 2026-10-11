"""Checks of the demand-share trend: SPI-3 event selection and all-month pressures without event selection."""
import pandas as pd

from drought_indices import standardize
from mechanism_analysis import PROC, annual_summary, build_events, climatology, read_indices


def all_month_pressures(d, reference=("1961", "1990")):
    rows = []
    for bid, g in d.groupby("basin", sort=True):
        g = g.set_index("date").sort_index()
        dp = (climatology(g.p, reference) - g.p).loc["1961":"2024"]
        de = (g.pet - climatology(g.pet, reference)).loc["1961":"2024"]
        supply, demand = dp.clip(lower=0), de.clip(lower=0)
        a = pd.DataFrame({"supply": supply, "demand": demand}).groupby(supply.index.year).sum()
        rows.append(pd.DataFrame({"basin": bid, "year": a.index, "demand_pressure_mm": a.demand.values,
                                  "gross_pressure_mm": (a.supply + a.demand).values}))
    return pd.concat(rows, ignore_index=True)


def main():
    d = read_indices()
    d["spi3"] = d.groupby("basin", group_keys=False).apply(
        lambda g: standardize(g.set_index("date").p.rolling(3).sum()).set_axis(g.index))
    records, annuals = [], []
    months = all_month_pressures(d)
    events = pd.read_csv(PROC / "mechanism_events.csv", dtype={"basin": str})
    events["basin"] = events.basin.str.zfill(4)
    separate = set(pd.read_csv(PROC / "nonnested_basins.csv", dtype={"basin": str}).basin.str.zfill(4))
    for name, frame in (("spi3_events", build_events(d, event_index="spi3")), ("all_months", months),
                        ("nonnested_spei3_events", events[events.basin.isin(separate)]),
                        ("nonnested_all_months", months[months.basin.isin(separate)])):
        annual, stats = annual_summary(frame)
        records.append({"check": name, "records": len(frame), **stats})
        annuals.append(annual.assign(check=name).reset_index())
        print(name, {k: round(v, 3) for k, v in stats.items()}, flush=True)
    pd.DataFrame(records).to_csv(PROC / "demand_share_checks.csv", index=False)
    pd.concat(annuals).to_csv(PROC / "demand_share_checks_annual.csv", index=False)


if __name__ == "__main__":
    main()
