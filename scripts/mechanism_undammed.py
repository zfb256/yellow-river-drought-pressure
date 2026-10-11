"""Response contrasts and pressure-share trend in basins without dams in the local dam list."""
import argparse

import numpy as np
import pandas as pd

from mechanism_analysis import (PROC, ROOT, TARGETS, NUMERIC, Design, fit_model, contrasts,
                                block_weights, annual_summary)

VARIABLES = ("p_deficit_mean", "pet_excess_mean", "soil_pre")


def analyze(f, cs, bootstrap):
    rows = []
    for target in TARGETS:
        col = f"y_{target}"
        source = f.assign(soil_pre=f.soil_pre_gleam) if target == "gleam_agri" else f
        g = source.dropna(subset=NUMERIC + [col]).copy().assign(y=lambda x: x[col].astype(int)).reset_index(drop=True)
        des = Design(g)
        x = des.transform(g)
        pairs, info = contrasts(g, des)
        keep = [i for i, r in enumerate(info) if r["variable"] in VARIABLES]
        pairs, info = [pairs[i] for i in keep], [info[i] for i in keep]
        m = fit_model(x, g.y, cs[target])
        point = [(m.predict_proba(b)[:, 1] - m.predict_proba(a)[:, 1]).mean() for a, b in pairs]
        rng, samples = np.random.default_rng(42), []
        for _ in range(bootstrap):
            boot = fit_model(x, g.y, cs[target], block_weights(g, rng))
            samples.append([(boot.predict_proba(b)[:, 1] - boot.predict_proba(a)[:, 1]).mean() for a, b in pairs])
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
        for i, r in enumerate(info):
            rows.append({"sensitivity": "undammed_proxy", "target": target, **r, "difference": point[i],
                         "lo": lo[i], "hi": hi[i], "total_n": len(g), "basins": g.basin.nunique()})
        print(f"undammed_proxy {target}: n={len(g)}, basins={g.basin.nunique()}, response={g.y.mean():.1%}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=100)
    args = ap.parse_args()
    base = pd.read_csv(PROC / "mechanism_events.csv", dtype={"basin": str}, parse_dates=["start", "end"])
    base["basin"] = base.basin.str.zfill(4)
    undammed = set((ROOT / "data/hydromlyr/natural_basins.txt").read_text().split())
    sub = base[base.basin.isin(undammed)]
    val = pd.read_csv(PROC / "response_validation.csv")
    cs = val[val.model == "spline_logit"].set_index("target").C.to_dict()
    r = pd.DataFrame(analyze(sub, cs, args.bootstrap))
    r.to_csv(PROC / "mechanism_undammed.csv", index=False)
    _, stats = annual_summary(sub)
    pd.DataFrame([{"reference": "1961-1990_undammed_proxy", "basins": sub.basin.nunique(), **stats}]).to_csv(
        PROC / "mechanism_undammed_trend.csv", index=False)
    print("\nUndammed-proxy pressure trend:", stats)
    print(r[["target", "season", "variable", "difference", "lo", "hi", "supported_n"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
