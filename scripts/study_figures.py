"""Analysis framework (Fig. 1), study-area map (Fig. 2) and precipitation comparison (Fig. 3)."""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
HYD = ROOT / "data" / "hydromlyr"
OUT = ROOT / "paper" / "figures"
TYPES = ["cold_deficit", "warm_compound", "heat"]
NAMES = {"cold_deficit": "Cold precipitation-deficit", "warm_compound": "Warm compound", "heat": "Heat-driven"}
COL = {"cold_deficit": "#2a78d6", "warm_compound": "#1baf7a", "heat": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ZONES = {"high": "High elevation (mean ≥2500 m)", "mid": "Mid elevation (1000–2500 m)",
         "low": "Low elevation (<1000 m)"}
LARGE_KM2 = 40000  # nested basins above this size are drawn as grey background on maps
ZCOL = {"high": "#4a3aa7", "mid": "#c9a35b", "low": "#9fb7a3"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def zone(elev):
    return "high" if elev >= 2500 else ("mid" if elev >= 1000 else "low")


def panel(ax, letter):
    ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def basins():
    shp = sorted((HYD / "0_basin_boundary").glob("*.shp"))
    g = gpd.GeoDataFrame(pd.concat([gpd.read_file(f).assign(basin=f.stem) for f in shp]), crs="EPSG:4326")
    g = g.dissolve("basin").reset_index()
    attr = {b: json.load(open(HYD / "1_data" / b / "attributes.json")) for b in g.basin}
    g["elev"] = g.basin.map(lambda b: attr[b]["elev"])
    g["area_km2"] = g.basin.map(lambda b: attr[b]["area"])
    g["zone"] = g.elev.map(zone)
    g["undammed"] = g.basin.isin(set((HYD / "natural_basins.txt").read_text().split()))
    return g.sort_values("area_km2", ascending=False)


def add_scale_and_north(ax):
    """Scale bar (approximate at 35 N) and north arrow in the lower right corner."""
    x0, y0 = 113.0, 32.6
    deg = 200 / (111.32 * np.cos(np.radians(35)))
    ax.plot([x0, x0 + deg], [y0, y0], color=INK, lw=1.2)
    ax.plot([x0, x0], [y0 - 0.08, y0 + 0.08], color=INK, lw=0.8)
    ax.plot([x0 + deg, x0 + deg], [y0 - 0.08, y0 + 0.08], color=INK, lw=0.8)
    ax.text(x0 + deg / 2, y0 + 0.2, "200 km", ha="center", va="bottom", fontsize=7, color=INK)
    ax.annotate("N", xy=(117.6, 33.9), xytext=(117.6, 32.6), ha="center", va="center", fontsize=8, color=INK,
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1})


def fig2_framework():
    """Schematic of the three analysis steps."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 40)
    ax.axis("off")
    box = {"boxstyle": "round,pad=0.45", "linewidth": 0.8}
    ax.add_patch(FancyBboxPatch((4, 32), 92, 6, boxstyle="round,pad=0.4", facecolor="#f2f1ed", edgecolor=INK2, lw=0.8))
    ax.text(50, 35, "CMFD forcing, FAO-56 PET, ABCD model, GLEAM and G-RUN  ·  102 catchments, 1961–2024",
            ha="center", va="center", fontsize=7.5, color=INK)
    steps = [(17, "1  Atmospheric pressure", "Precipitation deficit and\nPET excess of each drought", COL["cold_deficit"]),
             (50, "2  New land droughts", "New soil and runoff droughts\nvs. pressures and antecedent soil", COL["warm_compound"]),
             (83, "3  Seasonal pathways", "Temperature trend removed\nfor PET and snow in ABCD", COL["heat"])]
    arrow = {"arrowstyle": "-|>", "color": INK2, "lw": 0.8, "shrinkA": 0, "shrinkB": 0}
    for x, title, text, colour in steps:
        ax.annotate("", xy=(x, 25.6), xytext=(x, 31.2), arrowprops=arrow)
        ax.text(x, 22, title, ha="center", va="center", fontsize=8, fontweight="bold", color="white",
                bbox={**box, "facecolor": colour, "edgecolor": colour})
        ax.text(x, 10, text, ha="center", va="center", fontsize=7.5, color=INK,
                bbox={**box, "facecolor": "white", "edgecolor": colour})
    ax.annotate("", xy=(36.8, 22), xytext=(31.2, 22), arrowprops=arrow)
    ax.text(34, 23.6, "events", ha="center", va="bottom", fontsize=6.5, color=INK2)
    fig.savefig(OUT / "Fig1_framework.png")
    plt.close(fig)


def fig1_study_area(g):
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    large = g[g.area_km2 > LARGE_KM2]
    small = g[g.area_km2 <= LARGE_KM2]
    large.plot(ax=ax, color="#ecebe7", edgecolor="#c3c2b7", linewidth=0.4)
    small.plot(ax=ax, color=small.zone.map(ZCOL), edgecolor="white", linewidth=0.4)
    g.iloc[[0]].boundary.plot(ax=ax, color=INK, linewidth=0.9)
    for z in ("high", "mid", "low"):
        ax.plot([], [], "s", color=ZCOL[z], ms=7, label=f"{ZONES[z]} (n={(g.zone == z).sum()})")
    ax.plot([], [], "s", color="#ecebe7", mec="#c3c2b7", ms=7, label=f"Catchments >{LARGE_KM2 // 1000},000 km² (n={len(large)})")
    g[g.undammed].boundary.plot(ax=ax, color=INK, linewidth=0.35, linestyle=(0, (2, 1.5)))
    ax.plot([], [], color=INK, lw=0.6, ls=(0, (2, 1.5)), label="No dam (GRanD)")
    add_scale_and_north(ax)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(1.25)
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7)
    fig.savefig(OUT / "Fig2_study_area.png")
    plt.close(fig)


def fig2_homogeneity():
    c = pd.read_csv(PROC / "cmfd_basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    c["p"] = c.prec * 86400 * c.date.dt.days_in_month
    e = pd.read_csv(PROC / "drought_indices.csv", dtype={"basin": str}, parse_dates=["date"], usecols=["basin", "date", "p_raw"])
    st = {}
    for f in (HYD / "1_data").glob("*/meteorological.txt"):
        x = pd.read_csv(f, parse_dates=["date"]).set_index("date").pre.resample("YS").agg(["sum", "count"])
        x = x[x["count"] >= 360]["sum"]
        st[f.parent.name] = pd.Series(x.values, index=x.index.year)
    s = pd.DataFrame(st).median(axis=1).loc[2000:2019]
    ca = c.groupby(["basin", c.date.dt.year]).p.sum().groupby(level=1).median().loc[1961:2024]
    ea = e.groupby(["basin", e.date.dt.year]).p_raw.sum().groupby(level=1).median().loc[1961:2024]
    ref = slice(2000, 2019)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for series, lab, col, lw in ((ca, "CMFD v2.0", "#2a78d6", 1.4), (ea, "ERA5-Land", "#e34948", 1.0)):
        an = 100 * (series / series.loc[ref].mean() - 1)
        ax.plot(an.index, an.values, color=col, lw=lw, label=lab)
    sa = 100 * (s / s.mean() - 1)
    ax.plot(sa.index, sa.values, color=INK, lw=0, marker="o", ms=3, label="Station-based (HydroMLYR)")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_ylabel("Annual precipitation anomaly (%)\n(relative to 2000–2019 mean)")
    ax.set_xlabel("Year")
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.set_ylim(-50, 60)
    panel(ax, "a")

    ax = axes[1]
    trends = {"Station": np.polyfit(s.index, s.values, 1)[0] * 10,
              "CMFD": np.polyfit(range(2000, 2020), ca.loc[ref].values, 1)[0] * 10,
              "ERA5-Land": np.polyfit(range(2000, 2020), ea.loc[ref].values, 1)[0] * 10}
    cols = [INK, "#2a78d6", "#e34948"]
    bars = ax.bar(list(trends), list(trends.values()), color=cols, width=0.55)
    for b, v in zip(bars, trends.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + (3 if v > 0 else -3), f"{v:+.0f}".replace("-", "\u2212"), ha="center",
                va="bottom" if v > 0 else "top", color=INK, fontsize=7)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_ylabel("Precipitation trend 2000–2019\n(mm per decade)")
    ax.set_ylim(-70, 70)
    ax.grid(axis="x", visible=False)
    panel(ax, "b")
    fig.tight_layout()
    fig.savefig(OUT / "Fig3_data_homogeneity.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_study_area(basins())
    fig2_framework()
    fig2_homogeneity()


if __name__ == "__main__":
    main()
