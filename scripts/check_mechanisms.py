"""Consistency checks on the analysis outputs used by the manuscript."""
import json
import numpy as np
import pandas as pd
from abcd import run
from counterfactual import SCENARIOS, fixed_reference_index
from mechanism_analysis import PROC, NUMERIC, Design, TARGETS, read_indices


def main():
    # Water conservation is checked independently of the stored research outputs.
    rng = np.random.default_rng(7)
    p, pet = rng.uniform(0, 150, 120), rng.uniform(0, 100, 120)
    q, soil, gw, et = run(p, pet, np.full(120, 10.), .95, 300., .4, .1, 50., 20.)
    storage_change = np.diff(np.r_[70., soil + gw])
    assert np.allclose(p - q - et, storage_change, atol=1e-9)
    f = pd.read_csv(PROC / "mechanism_events.csv", dtype={"basin": str}, parse_dates=["start", "end"])
    assert len(f) == 5454 and not f.duplicated(["basin", "start"]).any()
    assert f.exposure_months.eq(f.duration + 2).all()
    assert np.allclose(f.net_deficit_mm, f.p_deficit_mm + f.pet_excess_mm)
    assert np.allclose(f.net_deficit_mm, f.gross_pressure_mm - f.compensation_mm)
    assert f.demand_share.dropna().between(0, 1).all()
    assert not f[NUMERIC].isna().any().any()
    for target in TARGETS:
        eligible = f[f[f"y_{target}"].notna()]
        assert not eligible[f"prior_{target}"].any()
        assert not eligible[f"censored_{target}"].any()
        assert eligible[f"y_{target}"].isin([0, 1]).all()
        hit = eligible[eligible[f"y_{target}"] == 1]
        assert hit[f"lag_{target}"].ge(0).all()
        assert hit[f"lag_{target}"].le(hit.duration + 2).all()
    des = Design(f)
    changed = f.assign(type_k3="arbitrary")
    assert np.array_equal(des.transform(f), des.transform(changed))
    cf = pd.read_csv(PROC / "counterfactual_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    assert set(cf.scenario) == set(SCENARIOS)
    assert cf.groupby("scenario").size().nunique() == 1
    assert np.isfinite(cf[["pet", "q", "soil", "et", "spei", "ssmi", "sri"]]).all().all()
    d = read_indices()
    # The common-reference mapping must preserve ordering and be identical for identical inputs.
    x = d[d.basin == f.basin.iloc[0]].set_index("date").p.rolling(3).sum()
    ref = x.loc["1961":"1990"]
    z = fixed_reference_index(x, ref)
    assert np.allclose(z, fixed_reference_index(x.copy(), ref), equal_nan=True)
    for m in range(1, 13):
        joined = pd.concat([x.rename("x"), z.rename("z")], axis=1).dropna()
        joined = joined[joined.index.month == m].sort_values("x")
        assert joined.z.diff().dropna().ge(-1e-12).all()
    r = pd.read_csv(PROC / "counterfactual_reproduction.csv")
    assert r[["q", "soil", "et"]].max().max() < .01
    metrics = pd.read_csv(PROC / "response_validation.csv")
    assert metrics.brier.between(0, 1).all() and metrics.auc.between(0, 1).all()
    margins = pd.read_csv(PROC / "response_margins.csv")
    assert margins.support_fraction.between(0, 1).all()
    assert (margins.lo <= margins.hi).all()
    sensitivity = pd.read_csv(PROC / "mechanism_robustness.csv")
    assert {"linear_response", "small_basins", "shared_abcd_antecedent_soil", "matched_product_riskset",
            "reference_1981_2010", "window_1", "window_6", "threshold_minus1"} <= set(sensitivity.sensitivity)
    undammed = pd.read_csv(PROC / "mechanism_undammed.csv")
    dam_list = set((PROC.parent / "hydromlyr/natural_basins.txt").read_text().split())
    assert set(undammed.target) == set(TARGETS) and (undammed.lo <= undammed.hi).all()
    assert undammed.basins.max() == len(dam_list & set(f.basin.str.zfill(4)))
    trend_u = pd.read_csv(PROC / "mechanism_undammed_trend.csv").iloc[0]
    assert trend_u.lo <= trend_u.slope_percentage_points_per_decade <= trend_u.hi
    print(json.dumps({"checks": "passed", "meteorological_events": len(f), "basins": f.basin.nunique(),
                      "counterfactual_rows": len(cf), "response_contrasts": len(margins),
                      "sensitivity_contrasts": len(sensitivity), "max_factual_relative_error": float(r[["q", "soil", "et"]].max().max())}, indent=2))


if __name__ == "__main__":
    main()
