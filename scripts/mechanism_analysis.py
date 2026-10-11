"""Continuous atmospheric water-balance pressures and conditional drought responses."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from propagation import events

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
OUT = ROOT / "paper/figures"
SEASON = {m: ("DJF" if m in (12, 1, 2) else "MAM" if m <= 5 else "JJA" if m <= 8 else "SON") for m in range(1, 13)}
TARGETS = {"abcd_agri": "ssmi3", "gleam_agri": "ssmi3_gleam", "abcd_hydro": "sri3", "grun_hydro": "sri3_grun"}
TARGET_NAMES = {"abcd_agri": "ABCD soil moisture", "gleam_agri": "GLEAM soil moisture",
                "abcd_hydro": "ABCD runoff", "grun_hydro": "G-RUN runoff"}
NUMERIC = ["p_deficit_mean", "pet_excess_mean", "soil_pre", "log_duration", "elev_km"]


def read_indices():
    d = pd.read_csv(PROC / "drought_indices_v2.csv", dtype={"basin": str}, parse_dates=["date"])
    d["basin"] = d.basin.str.zfill(4)
    assert not d.duplicated(["basin", "date"]).any()
    return d


def climatology(s, reference):
    r = s.loc[reference[0]:reference[1]]
    return pd.Series(s.index.month.map(r.groupby(r.index.month).mean()), index=s.index, dtype=float)


def anomaly(s, reference=("1961", "2024")):
    r = s.loc[reference[0]:reference[1]]
    mean, sd = r.groupby(r.index.month).mean(), r.groupby(r.index.month).std()
    return (s - s.index.month.map(mean)) / s.index.month.map(sd)


def pressure_parts(dp, de):
    supply = np.maximum(dp, 0).sum()
    demand = np.maximum(de, 0).sum()
    compensation = np.maximum(-dp, 0).sum() + np.maximum(-de, 0).sum()
    gross = supply + demand
    return {"p_deficit_mm": dp.sum(), "pet_excess_mm": de.sum(),
            "supply_pressure_mm": supply, "demand_pressure_mm": demand,
            "compensation_mm": compensation, "gross_pressure_mm": gross,
            "net_deficit_mm": dp.sum() + de.sum(),
            "demand_share": demand / gross if gross > 0 else np.nan}


def response(s, start, end, window=3):
    valid = s.dropna()
    limit = end + pd.DateOffset(months=window)
    if valid.empty or start < valid.index.min() or limit > valid.index.max():
        return np.nan, np.nan, True
    # Missing months inside the exposure/outcome window invalidate ascertainment.
    if s.loc[start:limit].isna().any():
        return np.nan, np.nan, True
    es = events(valid)
    prior = any(e["start"] < start <= e["end"] for e in es)
    hits = [e["start"] for e in es if start <= e["start"] <= limit]
    lag = (min(hits).year - start.year) * 12 + min(hits).month - start.month if hits else np.nan
    return float(bool(hits)) if not prior else np.nan, lag, prior


def build_events(d, reference=("1961", "1990"), window=3, threshold=-0.5, event_index="spei3"):
    import propagation
    old_threshold = propagation.THRESH
    propagation.THRESH = threshold
    rows = []
    try:
        for bid, group in d.groupby("basin", sort=True):
            g = group.set_index("date").sort_index()
            attr = json.loads((ROOT / "data/hydromlyr/1_data" / bid / "attributes.json").read_text())
            dp, de = climatology(g.p, reference) - g.p, g.pet - climatology(g.pet, reference)
            soils = {"abcd": anomaly(g.soil), "gleam": anomaly(g.sm_gleam, ("1980", "2024"))}
            target_events = {k: events(g[c].dropna()) for k, c in TARGETS.items()}
            for e in events(g[event_index].loc["1961":"2024"].dropna()):
                start, end = e["start"], e["end"]
                left = start - pd.DateOffset(months=2)
                pre_left, pre_right = start - pd.DateOffset(months=5), start - pd.DateOffset(months=3)
                x, z = dp.loc[left:end].values, de.loc[left:end].values
                n = len(x)
                r = {"basin": bid, **e, **pressure_parts(x, z), "exposure_months": n,
                     "p_deficit_mean": x.mean(), "pet_excess_mean": z.mean(),
                     "pet_absolute_mean": g.pet.loc[left:end].mean(), "elev_km": attr["elev"] / 1000,
                     "area_km2": attr["area"], "season": SEASON[start.month], "year": start.year,
                     "log_duration": np.log(e["duration"]),
                     "soil_pre": soils["abcd"].loc[pre_left:pre_right].mean(),
                     "soil_pre_gleam": soils["gleam"].loc[pre_left:pre_right].mean()}
                for k, col in TARGETS.items():
                    valid = g[col].dropna()
                    limit = end + pd.DateOffset(months=window)
                    complete = (not valid.empty and start >= valid.index.min() and
                                limit <= valid.index.max() and not g[col].loc[start:limit].isna().any())
                    prior = any(a["start"] < start <= a["end"] for a in target_events[k])
                    hits = [a["start"] for a in target_events[k] if start <= a["start"] <= limit]
                    r[f"y_{k}"] = float(bool(hits)) if complete and not prior else np.nan
                    r[f"prior_{k}"] = prior
                    r[f"censored_{k}"] = not complete
                    r[f"lag_{k}"] = ((min(hits).year - start.year) * 12 + min(hits).month - start.month
                                        if complete and not prior and hits else np.nan)
                rows.append(r)
    finally:
        propagation.THRESH = old_threshold
    f = pd.DataFrame(rows)
    assert np.allclose(f.net_deficit_mm, f.gross_pressure_mm - f.compensation_mm)
    return f


def block_weights(f, rng, block=5):
    """Bootstrap weights: resampled basins times 5-year calendar blocks shared by all basins."""
    years = np.arange(f.year.min(), f.year.max() + 1)
    length = min(block, len(years))
    starts = rng.integers(0, len(years) - length + 1, int(np.ceil(len(years) / length)))
    draw = np.concatenate([years[i:i + length] for i in starts])[:len(years)]
    yw = pd.Series(draw).value_counts()
    basins = f.basin.unique()
    bw = pd.Series(rng.choice(basins, len(basins), replace=True)).value_counts()
    w = f.year.map(yw).fillna(0).to_numpy() * f.basin.map(bw).fillna(0).to_numpy()
    return w / w.mean()


class Design:
    """Fixed low-rank B-spline additive terms, with demand x season and demand x soil."""
    def __init__(self, f, spline=True):
        self.spline = spline
        self.scale = StandardScaler().fit(f[NUMERIC])
        self.basin = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first").fit(f[["basin"]])
        self.smooth = SplineTransformer(n_knots=3, degree=3, include_bias=False,
                                        extrapolation="linear").fit(self.scale.transform(f[NUMERIC])[:, :3])

    def transform(self, f):
        z = self.scale.transform(f[NUMERIC])
        seasons = np.column_stack([(f.season == s).astype(float) for s in ("MAM", "JJA", "SON")])
        if self.spline:
            smooth = self.smooth.transform(z[:, :3])
            width = smooth.shape[1] // 3
            demand = smooth[:, width:2 * width]
            interactions = np.column_stack([demand * seasons[:, i:i + 1] for i in range(3)])
            base = np.column_stack([smooth, z[:, 3:], interactions])
        else:
            base = np.column_stack([z, z[:, 1:2] * seasons])
        return np.column_stack([base, seasons, z[:, 1] * z[:, 2], self.basin.transform(f[["basin"]])])


def fit_model(x, y, c=1, weights=None):
    m = LogisticRegression(C=c, solver="lbfgs", max_iter=1500, tol=1e-6)
    m.fit(x, y, sample_weight=weights)
    if m.n_iter_[0] >= m.max_iter:
        raise RuntimeError("Spline logistic model did not converge")
    return m


def validation(f, target):
    """Year-blocked tuning, then held-out 2000+ evaluation; preprocessing fits training only."""
    train, test = f[f.year <= 1999], f[f.year >= 2000]
    test = test[test.basin.isin(train.basin.unique())]
    if len(train) < 100 or len(test) < 100:
        raise ValueError(f"Insufficient temporal validation data for {target}")
    tune_train, tune_test = train[train.year <= 1989], train[train.year >= 1990]
    if len(tune_train) < 100:
        cut = int(train.year.quantile(.6))
        tune_train, tune_test = train[train.year <= cut], train[train.year > cut]
    tune_test = tune_test[tune_test.basin.isin(tune_train.basin.unique())]
    selected = {}
    scores = []
    baseline = np.full(len(test), train.y.mean())
    scores.append({"target": target, "model": "constant_train_rate", "C": np.nan,
                   "train_n": len(train), "test_n": len(test), "brier": brier_score_loss(test.y, baseline),
                   "log_loss": log_loss(test.y, baseline), "auc": .5,
                   "observed_rate": test.y.mean(), "predicted_rate": train.y.mean()})
    for spline, name in ((False, "linear_logit"), (True, "spline_logit")):
        des = Design(tune_train, spline)
        a, b = des.transform(tune_train), des.transform(tune_test)
        losses = {c: log_loss(tune_test.y, fit_model(a, tune_train.y, c).predict_proba(b)[:, 1]) for c in (.1, 1, 10)}
        c = min(losses, key=losses.get)
        selected[name] = c
        des = Design(train, spline)
        m = fit_model(des.transform(train), train.y, c)
        p = m.predict_proba(des.transform(test))[:, 1]
        scores.append({"target": target, "model": name, "C": c, "train_n": len(train), "test_n": len(test),
                       "brier": brier_score_loss(test.y, p), "log_loss": log_loss(test.y, p),
                       "auc": roc_auc_score(test.y, p), "observed_rate": test.y.mean(), "predicted_rate": p.mean()})
    # Same predictors and train/test split; boosting is only a comparison.
    des = Design(train, False)
    m = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=40,
                                      l2_regularization=10, learning_rate=.05, random_state=0)
    m.fit(des.transform(train), train.y)
    p = m.predict_proba(des.transform(test))[:, 1]
    scores.append({"target": target, "model": "boosting", "C": np.nan, "train_n": len(train), "test_n": len(test),
                   "brier": brier_score_loss(test.y, p), "log_loss": log_loss(test.y, p),
                   "auc": roc_auc_score(test.y, p), "observed_rate": test.y.mean(), "predicted_rate": p.mean()})
    return selected["spline_logit"], scores


def contrasts(f, des):
    pairs, info = [], []
    for season in ("DJF", "MAM", "JJA", "SON"):
        g = f[f.season == season]
        scale = StandardScaler().fit(g[NUMERIC[:3]])
        observed = scale.transform(g[NUMERIC[:3]])
        neighbors = NearestNeighbors(n_neighbors=10).fit(observed)
        radius = np.quantile(neighbors.kneighbors(observed, n_neighbors=11)[0][:, -1], .95)
        for variable in ("pet_excess_mean", "p_deficit_mean", "soil_pre"):
            lo, hi = g[variable].quantile([.25, .75])
            a, b = g.copy(), g.copy()
            a[variable], b[variable] = lo, hi
            da = neighbors.kneighbors(scale.transform(a[NUMERIC[:3]]))[0][:, -1]
            db = neighbors.kneighbors(scale.transform(b[NUMERIC[:3]]))[0][:, -1]
            supported = (da <= radius) & (db <= radius)
            if supported.sum() < 30:
                raise ValueError(f"Insufficient joint support for {season} {variable}")
            pairs.append((des.transform(a[supported]), des.transform(b[supported])))
            info.append({"season": season, "variable": variable, "low": lo, "high": hi,
                         "n": len(g), "supported_n": int(supported.sum()), "support_fraction": supported.mean()})
    return pairs, info


def model_responses(f, n_boot=200, seed=42):
    margins, scores, curves, counts = [], [], [], []
    for target in TARGETS:
        ycol = f"y_{target}"
        source = f.assign(soil_pre=f.soil_pre_gleam) if target == "gleam_agri" else f
        g = source.dropna(subset=NUMERIC + [ycol]).copy().assign(y=lambda x: x[ycol].astype(int)).reset_index(drop=True)
        counts.append({"target": target, "n": len(g), "events": int(g.y.sum()), "rate": g.y.mean(),
                       "prior_excluded": int(f[f"prior_{target}"].sum()),
                       "censored_excluded": int(f[f"censored_{target}"].sum())})
        c, score = validation(g, target)
        scores.extend(score)
        des = Design(g)
        x = des.transform(g)
        m = fit_model(x, g.y, c)
        pairs, info = contrasts(g, des)
        point = np.array([(m.predict_proba(b)[:, 1] - m.predict_proba(a)[:, 1]).mean() for a, b in pairs])
        rng, samples = np.random.default_rng(seed), []
        for k in range(n_boot):
            w = block_weights(g, rng)
            boot = fit_model(x, g.y, c, w)
            samples.append([(boot.predict_proba(b)[:, 1] - boot.predict_proba(a)[:, 1]).mean() for a, b in pairs])
            if (k + 1) % 50 == 0:
                print(f"{target}: bootstrap {k + 1}/{n_boot}", flush=True)
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
        for i, rec in enumerate(info):
            margins.append({"target": target, **rec, "difference": point[i], "lo": lo[i], "hi": hi[i]})
        # Marginal demand curves, displayed only over each season's central 90% exposure range.
        for season in ("DJF", "JJA"):
            s = g[g.season == season]
            a, b = s.pet_excess_mean.quantile([.05, .95])
            for value in np.linspace(a, b, 31):
                frame = s.copy()
                frame["pet_excess_mean"] = value
                curves.append({"target": target, "season": season, "pet_excess_mean": value,
                               "probability": m.predict_proba(des.transform(frame))[:, 1].mean()})
        print(f"{target}: eligible {len(g)}, response {g.y.mean():.1%}, spline C={c}", flush=True)
    for name, records in (("response_margins", margins), ("response_validation", scores),
                          ("response_curves", curves), ("response_risksets", counts)):
        pd.DataFrame(records).to_csv(PROC / f"{name}.csv", index=False)
    return pd.DataFrame(margins), pd.DataFrame(scores)


def annual_summary(f, n_boot=2000):
    # Equal basin weights within each year; no-event basin-years do not define a pressure share.
    a = f.groupby(["basin", "year"]).agg(demand=("demand_pressure_mm", "sum"), gross=("gross_pressure_mm", "sum"))
    a["share"] = a.demand / a.gross
    annual = a.groupby("year").agg(demand_share=("share", "mean"), contributing_basins=("share", "count"))
    years, values = annual.index.to_numpy(), annual.demand_share.to_numpy()
    slope = theilslopes(values, years)[0]
    fit = np.median(values - slope * years) + slope * years
    residual, rng, slopes = values - fit, np.random.default_rng(42), []
    for _ in range(n_boot):
        starts = rng.integers(0, len(years) - 4, int(np.ceil(len(years) / 5)))
        res = np.concatenate([residual[i:i + 5] for i in starts])[:len(years)]
        slopes.append(theilslopes(fit + res, years)[0] * 1000)
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    result = {"slope_percentage_points_per_decade": slope * 1000, "lo": lo, "hi": hi,
              "early_share": annual.loc[1961:1990].demand_share.mean(),
              "late_share": annual.loc[1995:2024].demand_share.mean()}
    return annual, result


PALETTE = {"blue": "#2a78d6", "aqua": "#1baf7a", "orange": "#eb6834", "purple": "#8a5cc2", "ink": "#0b0b0b", "ink2": "#52514e"}
SEASON_COLOURS = {"DJF": PALETTE["blue"], "MAM": PALETTE["aqua"], "JJA": PALETTE["orange"], "SON": PALETTE["purple"]}


def style():
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": PALETTE["ink2"], "legend.frameon": False, "savefig.dpi": 300,
                         "savefig.bbox": "tight"})


def figures(f, margins):
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    annual = pd.read_csv(PROC / "mechanism_annual.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ref, colour in (("1961-1990", PALETTE["blue"]), ("1981-2010", PALETTE["aqua"])):
        g = annual[annual.reference == ref]
        axes[0].plot(g.year, g.demand_share * 100, lw=1.2, color=colour, label=f"Events, {ref.replace('-', '–')} reference")
    checks = PROC / "demand_share_checks_annual.csv"
    if checks.exists():
        g = pd.read_csv(checks).query("check == 'all_months'")
        axes[0].plot(g.year, g.demand_share * 100, lw=1.2, color=PALETTE["orange"], ls="--", label="All months, no event selection")
    axes[0].set(xlabel="Year", ylabel="Demand share of gross pressure (%)")
    axes[0].set_title("(a)", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7, loc="upper left")
    for season in ("DJF", "MAM", "JJA", "SON"):
        g = f[f.season == season]
        axes[1].scatter(g.pet_excess_mean, g.p_deficit_mean, s=2, alpha=.25, color=SEASON_COLOURS[season], label=season, lw=0)
    axes[1].axhline(0, color=PALETTE["ink2"], lw=.5)
    axes[1].axvline(0, color=PALETTE["ink2"], lw=.5)
    axes[1].set(xlabel="Mean PET excess (mm/month)", ylabel="Mean precipitation deficit (mm/month)")
    axes[1].set_title("(b)", loc="left", fontweight="bold")
    axes[1].legend(markerscale=4, fontsize=7, title="Onset season", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "Fig4_continuous_pressures.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, targets, label in zip(axes, (("abcd_agri", "gleam_agri"), ("abcd_hydro", "grun_hydro")), ("(a) Soil moisture", "(b) Runoff")):
        for i, (target, colour) in enumerate(zip(targets, (PALETTE["blue"], PALETTE["orange"]))):
            g = margins[(margins.target == target) & (margins.variable == "soil_pre")].set_index("season").loc[["DJF", "MAM", "JJA", "SON"]]
            ax.errorbar(np.arange(4) + (i - .5) * .18, g.difference * 100,
                        yerr=np.array([g.difference - g.lo, g.hi - g.difference]) * 100,
                        fmt="o", ms=4, capsize=2.5, color=colour, label=TARGET_NAMES[target])
        ax.axhline(0, color=PALETTE["ink2"], lw=.6)
        ax.set_xticks(range(4), ["DJF", "MAM", "JJA", "SON"])
        ax.set_title(label, loc="left", fontweight="bold")
        ax.legend(fontsize=7, loc="upper right" if targets[0] == "abcd_agri" else "lower right")
    axes[0].set_ylabel("Response difference, wet minus dry\nantecedent soil (percentage points)")
    axes[0].set_ylim(-25, 8)
    fig.tight_layout()
    fig.savefig(OUT / "Fig5_storage_buffering.png")
    plt.close(fig)


def self_check():
    r = pressure_parts(np.array([10., -5.]), np.array([8., -2.]))
    assert r["gross_pressure_mm"] == 18 and r["compensation_mm"] == 7 and r["net_deficit_mm"] == 11
    assert np.isclose(r["demand_share"], 8 / 18)
    assert np.isnan(pressure_parts(np.array([-1.]), np.array([-2.]))["demand_share"])
    dates = pd.date_range("2000-01-01", periods=12, freq="MS")
    s = pd.Series([0, -1, -1, 0, 0, 0, -1, -1, 0, 0, 0, 0], index=dates)
    assert np.isnan(response(s, dates[2], dates[3])[0])  # pre-existing drought
    assert response(s, dates[4], dates[5])[0] == 1      # new onset within end+3
    assert np.isnan(response(s, dates[10], dates[11])[0])  # incomplete follow-up
    print("mechanism self-check passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    self_check()
    if args.self_check:
        return
    d = read_indices()
    f = build_events(d)
    f.to_csv(PROC / "mechanism_events.csv", index=False)
    records, annuals = [], []
    for ref in (("1961", "1990"), ("1981", "2010")):
        alt = f if ref[0] == "1961" else build_events(d, ref)
        a, stats = annual_summary(alt)
        label = "-".join(ref)
        annuals.append(a.assign(reference=label).reset_index())
        records.append({"reference": label, **stats})
        print(f"Pressure composition {label}: {stats}", flush=True)
    pd.concat(annuals).to_csv(PROC / "mechanism_annual.csv", index=False)
    pd.DataFrame(records).to_csv(PROC / "mechanism_trends.csv", index=False)
    margins, scores = model_responses(f, args.bootstrap)
    print("\nSeasonal adjusted contrasts:\n" + margins.round(4).to_string(index=False))
    print("\nTemporal validation (train <=1999, test >=2000):\n" + scores.round(4).to_string(index=False))
    figures(f, margins)


if __name__ == "__main__":
    main()
