"""Drought propagation time and run-theory drought events, 1961-2024."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drought_indices import standardize

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
HYD = ROOT / "data" / "hydromlyr"
START, END = "1961", "2024"
THRESH, MIN_DUR, MAX_GAP = -0.5, 2, 1


def propagation_time(d, target):
    best_k, best_r = None, -np.inf
    for k in range(1, 25):
        spei = standardize((d.p - d.pet).rolling(k).sum()).loc[START:END]
        r = spei.corr(d[target].loc[START:END])
        if r > best_r:
            best_k, best_r = k, r
    return best_k, best_r


def events(s):
    below = (s < THRESH).values
    runs, i, n = [], 0, len(s)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= MAX_GAP:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out = []
    for a, b in merged:
        seg = s.iloc[a:b]
        if b - a >= MIN_DUR:
            out.append({"start": seg.index[0], "end": seg.index[-1], "duration": b - a,
                        "severity": float((THRESH - seg.clip(upper=THRESH)).sum()), "peak": float(seg.min())})
    return out


def main():
    d = pd.read_csv(PROC / "drought_indices_v2.csv", dtype={"basin": str}, parse_dates=["date"])
    d["basin"] = d.basin.str.zfill(4)
    natural = set((HYD / "natural_basins.txt").read_text().split())

    prop, ev = [], []
    for bid, g in d.groupby("basin"):
        g = g.set_index("date").sort_index()
        kh, rh = propagation_time(g, "sri1")
        ka, ra = propagation_time(g, "ssmi1")
        prop.append({"basin": bid, "natural": bid in natural, "pt_hydro": kh, "r_hydro": rh, "pt_agri": ka, "r_agri": ra})
        for kind, col in (("meteorological", "spei3"), ("hydrological", "sri3"), ("agricultural", "ssmi3")):
            for e in events(g[col].loc[START:END].dropna()):
                ev.append({"basin": bid, "type": kind, **e})

    p = pd.DataFrame(prop)
    e = pd.DataFrame(ev)
    p.to_csv(PROC / "propagation.csv", index=False, float_format="%.4g")
    e.to_csv(PROC / "drought_events.csv", index=False, float_format="%.4g")

    print("propagation time (months), median [IQR]:")
    for c in ("pt_hydro", "pt_agri"):
        print(f"  {c}: {p[c].median():.0f} [{p[c].quantile(.25):.0f}-{p[c].quantile(.75):.0f}],"
              f" r = {p[c.replace('pt', 'r')].median():.2f}")
    print("  natural vs disturbed:", p.groupby("natural")[["pt_hydro", "pt_agri"]].median().to_dict())
    yrs = int(END) - int(START) + 1
    s = e.groupby("type").agg(n_per_basin=("basin", lambda x: len(x) / d.basin.nunique()),
                              duration=("duration", "median"), severity=("severity", "median"), peak=("peak", "median"))
    s["events_per_decade"] = s.n_per_basin / yrs * 10
    print(s.round(2))
    e["decade"] = e.start.dt.year // 10 * 10
    print("events per basin by decade:")
    print((e.groupby(["type", "decade"]).size().unstack(0) / d.basin.nunique()).round(2))


if __name__ == "__main__":
    main()
