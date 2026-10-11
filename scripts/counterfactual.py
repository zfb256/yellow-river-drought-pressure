"""ABCD model-conditional trend-removal experiments with a shared factual reference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from abcd import T_RAIN, T_SNOW, simulate
from drought_indices_v2 import cmfd_inputs
from pet_check import svp
from mechanism_analysis import SEASON, block_weights

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
SCENARIOS = ("factual", "pet_rh", "snow_temperature", "combined_rh", "pet_detrended")


def snowpack(p, t):
    """End-of-month snow storage and melt of the ABCD snow module, which depends only on P and T."""
    snow, store, melt = 0.0, np.empty(len(p)), np.empty(len(p))
    for i, (pi, ti) in enumerate(zip(p, t)):
        frac = min(max((ti - T_SNOW) / (T_RAIN - T_SNOW), 0.0), 1.0)
        snow += pi * (1.0 - frac)
        melt[i] = snow * frac
        snow -= melt[i]
        store[i] = snow
    return store, melt


def remove_trend(s, anchor=1961):
    result, slopes = s.copy(), []
    for month in range(1, 13):
        mask = s.index.month == month
        fit = s[mask].loc["1961":"2024"].dropna()
        slope = np.polyfit(fit.index.year, fit.values, 1)[0]
        result.loc[mask] -= slope * np.maximum(s.index[mask].year - anchor, 0)
        slopes.append(slope)
    return result, np.array(slopes)


def fixed_reference_index(s, reference):
    """Finite empirical mid-ranks against the factual reference (same for all scenarios)."""
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for month in range(1, 13):
        r = np.sort(reference[(reference.index.month == month) & reference.notna()].values)
        mask = (s.index.month == month) & s.notna()
        if len(r) < 10:
            continue
        x = s[mask].values
        rank = (np.searchsorted(r, x, "left") + np.searchsorted(r, x, "right")) / 2 + .5
        probability = np.clip((rank - .44) / (len(r) + .12), .56 / (len(r) + .12),
                              (len(r) - .44) / (len(r) + .12))
        out.loc[mask] = norm.ppf(probability)
    return out


def forcing_scenarios(c, attr):
    actual = cmfd_inputs(c, attr["lat"], attr["elev"])
    temperature, slopes = remove_trend(actual.t)
    p_kpa = c.pres / 1000
    ea = c.shum * p_kpa / (.622 + .378 * c.shum)
    rh = ea / svp(actual.t)
    modified = c.copy()
    modified["temp"] = temperature + 273.15
    ea_cf = rh * svp(temperature)
    modified["shum"] = .622 * ea_cf / (p_kpa - .378 * ea_cf)
    assert np.allclose(modified.shum * p_kpa / (.622 + .378 * modified.shum) / svp(temperature), rh)
    modified_inputs = cmfd_inputs(modified, attr["lat"], attr["elev"])
    assert np.allclose(actual.p, modified_inputs.p)
    pet_dt, _ = remove_trend(actual.pet)
    return actual, {
        "factual": (actual.pet, actual.t),
        "pet_rh": (modified_inputs.pet, actual.t),
        "snow_temperature": (actual.pet, temperature),
        "combined_rh": (modified_inputs.pet, temperature),
        "pet_detrended": (pet_dt.clip(lower=0), actual.t),
    }, slopes


def summary(monthly, n_boot=2000):
    metrics = ["pet", "q", "soil", "et", "snow", "melt", "spei_drought", "ssmi_drought", "sri_drought"]
    a = monthly.groupby(["basin", "year", "season", "scenario"])[metrics].mean().reset_index()
    actual = a[a.scenario == "factual"].drop(columns="scenario")
    records = []
    for scenario in SCENARIOS[1:]:
        paired = a[a.scenario == scenario].merge(actual, on=["basin", "year", "season"], suffixes=("_cf", "_actual"))
        for period, begin, end in (("1961-1990", 1961, 1990), ("1995-2024", 1995, 2024)):
            for season in ("DJF", "MAM", "JJA", "SON"):
                g = paired[paired.year.between(begin, end) & (paired.season == season)].reset_index(drop=True)
                differences = np.column_stack([g[f"{m}_cf"] - g[f"{m}_actual"] for m in metrics])
                point = differences.mean(axis=0)
                rng, boots = np.random.default_rng(42), []
                for _ in range(n_boot):
                    weights = block_weights(g, rng)
                    boots.append(np.average(differences, weights=weights, axis=0))
                lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
                for i, metric in enumerate(metrics):
                    records.append({"scenario": scenario, "period": period, "season": season, "metric": metric,
                                    "factual_mean": g[f"{metric}_actual"].mean(), "counterfactual_mean": g[f"{metric}_cf"].mean(),
                                    "difference": point[i], "lo": lo[i], "hi": hi[i]})
    return a, pd.DataFrame(records)


def plot(s):
    from mechanism_analysis import PALETTE, style
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    labels = {"pet_rh": "PET pathway", "snow_temperature": "Snow pathway", "combined_rh": "Combined"}
    colours = {"pet_rh": PALETTE["blue"], "snow_temperature": PALETTE["aqua"], "combined_rh": PALETTE["orange"]}
    panels = (("q", "Runoff change (mm/month)", "(a) Runoff"),
              ("ssmi_drought", "Change (percentage points)", "(b) Soil drought months"),
              ("sri_drought", "Change (percentage points)", "(c) Runoff drought months"))
    for ax, (metric, ylabel, title) in zip(axes, panels):
        for i, scenario in enumerate(labels):
            g = s[(s.scenario == scenario) & (s.period == "1995-2024") & (s.metric == metric)].set_index("season").loc[["DJF", "MAM", "JJA", "SON"]]
            factor = 1 if metric == "q" else 100
            ax.errorbar(np.arange(4) + (i - 1) * .18, g.difference * factor,
                        yerr=np.array([g.difference - g.lo, g.hi - g.difference]) * factor,
                        fmt="o", ms=3.5, capsize=2, color=colours[scenario], label=labels[scenario])
        ax.axhline(0, color=PALETTE["ink2"], lw=.6)
        ax.set_xticks(range(4), ["DJF", "MAM", "JJA", "SON"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "paper/figures/Fig6_counterfactual.png", dpi=300)
    plt.close(fig)


def self_check():
    dates = pd.date_range("1951-01-01", "2024-12-01", freq="MS")
    x = pd.Series(10 + .03 * (dates.year - 1961) + np.sin(dates.month), index=dates)
    cf, slopes = remove_trend(x)
    assert np.allclose(slopes, .03)
    assert np.allclose(cf.loc["1951":"1960"], x.loc["1951":"1960"])
    assert abs(np.polyfit(cf.loc["1961":"2024"].index.year, cf.loc["1961":"2024"], 1)[0]) < 1e-10
    z = fixed_reference_index(x, x.loc["1961":"1990"])
    assert np.isfinite(z).all()
    assert fixed_reference_index(x + 100, x.loc["1961":"1990"]).min() > 0
    q, soil, gw, et = simulate([.95, 300, .4, .1], np.ones(60) * 40, np.ones(60) * 30, np.ones(60) * 10)
    assert np.isfinite(q).all() and (q >= 0).all() and (soil >= 0).all()
    print("counterfactual self-check passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--snow-shift", type=float, default=0.0, help="raise both snow thresholds by this many degrees C")
    args = ap.parse_args()
    self_check()
    if args.self_check:
        return
    forcing = pd.read_csv(PROC / "cmfd_basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    parameters = pd.read_csv(PROC / "abcd_params.csv", dtype={"basin": str}).set_index("basin")
    old = pd.read_csv(PROC / "drought_indices_v2.csv", dtype={"basin": str}, parse_dates=["date"]).set_index(["basin", "date"])
    records, checks = [], []
    for bid, c in forcing.groupby("basin"):
        c = c.set_index("date").sort_index()
        attr = json.loads((ROOT / "data/hydromlyr/1_data" / bid / "attributes.json").read_text())
        x, scenarios, slopes = forcing_scenarios(c, attr)
        params = parameters.loc[bid, ["a", "b", "c", "d"]].to_numpy()
        frames, refs = {}, {}
        for scenario, (pet, temperature) in scenarios.items():
            # Raising the snow thresholds by k is equivalent to lowering the snow-module temperature by k.
            snow_t = temperature.values - args.snow_shift
            q, soil, gw, et = simulate(params, x.p.values, pet.values, snow_t)
            snow, melt = snowpack(x.p.values, snow_t)
            frame = pd.DataFrame({"pet": pet, "q": q, "soil": soil, "et": et, "snow": snow, "melt": melt}, index=x.index)
            raw = {"spei": (x.p - pet).rolling(3).sum(), "ssmi": frame.soil.rolling(3).mean(), "sri": frame.q.rolling(3).sum()}
            if scenario == "factual":
                refs = {key: value.loc["1961":"1990"] for key, value in raw.items()}
            if scenario == "factual" and not args.snow_shift:
                saved = old.loc[bid].reindex(x.index)
                errors = {key: float(np.max(np.abs(frame[key] - saved[key])) / max(saved[key].abs().max(), 1)) for key in ("q", "soil", "et")}
                if max(errors.values()) > .01:
                    raise ValueError(f"Factual reproduction exceeds 1% relative maximum error: {bid} {errors}")
                checks.append({"basin": bid, "warming_per_decade": slopes.mean() * 10, **errors})
            for key, value in raw.items():
                frame[key] = fixed_reference_index(value, refs[key])
                frame[f"{key}_drought"] = (frame[key] < -.5).astype(float)
            frame = frame.loc["1961":"2024"].copy()
            frame["basin"], frame["scenario"] = bid, scenario
            # December counts toward the next year's winter.
            frame["year"] = frame.index.year + (frame.index.month == 12)
            frame["season"] = frame.index.month.map(SEASON)
            # Keep complete winters only; the first winter of 1961 lacks December 1960.
            frame = frame[~((frame.season == "DJF") & frame.year.isin([1961, 2025]))]
            frames[scenario] = frame
        records.extend(frames.values())
    monthly = pd.concat(records).rename_axis("date").reset_index()
    if args.snow_shift:
        s = summary(monthly, n_boot=500)[1]
        s.to_csv(PROC / f"counterfactual_summary_snow{args.snow_shift:+.0f}.csv", index=False)
        print(s[(s.period == "1995-2024") & (s.metric == "q")].round(3).to_string(index=False))
        return
    monthly.to_csv(PROC / "counterfactual_monthly.csv", index=False, float_format="%.7g")
    pd.DataFrame(checks).to_csv(PROC / "counterfactual_reproduction.csv", index=False)
    annual, s = summary(monthly)
    annual.to_csv(PROC / "counterfactual_annual.csv", index=False)
    s.to_csv(PROC / "counterfactual_summary.csv", index=False)
    print("Factual reproduction relative errors:\n" + pd.DataFrame(checks).describe().to_string())
    print("\nRecent effects (CF - factual):\n" + s[(s.period == "1995-2024") & s.metric.isin(["q", "ssmi_drought", "sri_drought"])].round(5).to_string(index=False))
    plot(s)


if __name__ == "__main__":
    main()
