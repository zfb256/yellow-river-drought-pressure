"""Regenerate Figs. 3-5 from saved outputs without refitting the models; Figs. 1-2 come from study_figures.py."""
import pandas as pd
from mechanism_analysis import PROC, figures
from counterfactual import plot

if __name__ == "__main__":
    f = pd.read_csv(PROC / "mechanism_events.csv")
    margins = pd.read_csv(PROC / "response_margins.csv")
    figures(f, margins)
    plot(pd.read_csv(PROC / "counterfactual_summary.csv"))
    print("Wrote Figs. 4-6")
