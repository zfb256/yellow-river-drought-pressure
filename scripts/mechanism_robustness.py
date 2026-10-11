"""Predeclared sensitivity checks; all outcomes are reported, including failures to agree."""
import argparse
import numpy as np
import pandas as pd

from mechanism_analysis import (PROC, TARGETS, NUMERIC, Design, build_events, read_indices,
                                fit_model, contrasts, block_weights, annual_summary)


def analyze(f, name, cs, bootstrap=100, spline=True, targets=None, own_soil=True,
            variables=("pet_excess_mean", "soil_pre"), seasons=("DJF", "JJA")):
    rows = []
    for target in targets or TARGETS:
        col = f"y_{target}"
        source = f.assign(soil_pre=f.soil_pre_gleam) if target == "gleam_agri" and own_soil else f
        g = source.dropna(subset=NUMERIC + [col]).copy().assign(y=lambda x: x[col].astype(int)).reset_index(drop=True)
        des = Design(g, spline=spline)
        x = des.transform(g)
        pairs, info = contrasts(g, des)
        wanted = [i for i, r in enumerate(info) if r["variable"] in variables and r["season"] in seasons]
        pairs = [pairs[i] for i in wanted]
        info = [info[i] for i in wanted]
        m = fit_model(x, g.y, cs[target])
        point = [(m.predict_proba(b)[:, 1] - m.predict_proba(a)[:, 1]).mean() for a, b in pairs]
        rng, samples = np.random.default_rng(42), []
        for _ in range(bootstrap):
            boot = fit_model(x, g.y, cs[target], block_weights(g, rng))
            samples.append([(boot.predict_proba(b)[:, 1] - boot.predict_proba(a)[:, 1]).mean() for a, b in pairs])
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
        for i, r in enumerate(info):
            rows.append({"sensitivity": name, "target": target, **r, "difference": point[i], "lo": lo[i], "hi": hi[i], "total_n": len(g)})
        print(f"{name} {target}: n={len(g)}, response={g.y.mean():.1%}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=100)
    args = ap.parse_args()
    base = pd.read_csv(PROC / "mechanism_events.csv", dtype={"basin": str}, parse_dates=["start", "end"])
    val = pd.read_csv(PROC / "response_validation.csv")
    cs = val[val.model == "spline_logit"].set_index("target").C.to_dict()
    rows = analyze(base, "linear_response", cs, args.bootstrap, spline=False)
    rows += analyze(base[base.area_km2 <= 40000], "small_basins", cs, args.bootstrap)
    rows += analyze(base, "shared_abcd_antecedent_soil", cs, args.bootstrap,
                    targets=["gleam_agri"], own_soil=False)
    # Drop target droughts that start in the meteorological onset month (lagged onsets only).
    for target in TARGETS:
        lagged = base.copy()
        lagged.loc[lagged[f"lag_{target}"] == 0, f"y_{target}"] = np.nan
        rows += analyze(lagged, "lagged_onset_only", cs, args.bootstrap, targets=[target],
                        variables=("pet_excess_mean", "p_deficit_mean", "soil_pre"),
                        seasons=("DJF", "MAM", "JJA", "SON"))
    for pair in (("abcd_agri", "gleam_agri"), ("abcd_hydro", "grun_hydro")):
        matched = base.dropna(subset=[f"y_{p}" for p in pair])
        rows += analyze(matched, "matched_product_riskset", cs, args.bootstrap, targets=pair)
    d = read_indices()
    for name, settings in (("reference_1981_2010", {"reference": ("1981", "2010")}),
                           ("window_1", {"window": 1}), ("window_6", {"window": 6}),
                           ("threshold_minus1", {"threshold": -1.0})):
        alt = build_events(d, **settings)
        rows += analyze(alt, name, cs, args.bootstrap)
    r = pd.DataFrame(rows)
    r.to_csv(PROC / "mechanism_robustness.csv", index=False)
    small, stats = annual_summary(base[base.area_km2 <= 40000])
    pd.DataFrame([{"reference": "1961-1990_small_basins", **stats}]).to_csv(PROC / "mechanism_small_basin_trend.csv", index=False)
    print("\nSmall-basin pressure trend:", stats)
    print("\nSensitivity contrasts:\n" + r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
