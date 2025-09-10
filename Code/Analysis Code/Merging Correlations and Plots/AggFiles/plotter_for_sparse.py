#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce per-diffusion trend plots and medians from old_t50.csv
with fixed colors:
  - Homogeneous: blue
  - Heterogeneous: orange
  - Sensors: green
  - All: red

Outputs (in the chosen outdir):
  - GT_trend_BFS.png
  - GT_trend_Midpoint.png
  - GT_trend_t50.png
  - GT_perD_medians_by_category.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ----------------------- helpers -----------------------

def norm_params(s: str) -> str:
    s = str(s).strip().replace("__", "+")
    # normalize legacy "+single" to "+sR:single"
    if "+single" in s and "+sR:single" not in s:
        s = s.replace("+single", "+sR:single")
    return s

def to_label(params: str) -> str:
    """
    Turn 'Params' into the row label used in the pivot CSVs.
      - 'fR:0.01+sR:single' -> 'fR=0.01'          (homogeneous)
      - 'fR:0.01+sR:0.001'  -> 'fR=0.01, sR=0.001'
      - already 'fR=..., sR=...' -> unchanged
    """
    s = norm_params(params)
    m_h = re.match(r"^fR:([0-9eE\.\-]+)\+sR:single$", s)
    if m_h:
        return f"fR={m_h.group(1)}"
    m_b = re.match(r"^fR:([0-9eE\.\-]+)\+sR:([0-9eE\.\-]+)$", s)
    if m_b:
        return f"fR={m_b.group(1)}, sR={m_b.group(2)}"
    m_c = re.match(r"^fR=([0-9eE\.\-]+)(?:,\s*sR=([0-9eE\.\-]+))?$", s)
    if m_c:
        return s
    return s

def parse_label(lbl: str):
    """Return (fR, sR) floats; for 'fR=val' treat as homogeneous (fR=sR=val)."""
    m = re.match(r"fR=([0-9eE\.\-]+)(?:,\s*sR=([0-9eE\.\-]+))?$", lbl)
    if not m:
        return None, None
    f = float(m.group(1))
    s = float(m.group(2)) if m.group(2) is not None else f
    return f, s

def categorize(lbl: str) -> str:
    """
    Categories:
      - 'homogeneous'   : fR == sR > 0  (aka 'single')
      - 'heterogeneous' : fR != sR and both > 0
      - 'sensors'       : exactly one of fR or sR is 0
    """
    f, s = parse_label(lbl)
    if f is None:
        return "unknown"
    if (f == 0) ^ (s == 0):
        return "sensors"
    if f == s and f > 0:
        return "homogeneous"
    if f != s and f > 0 and s > 0:
        return "heterogeneous"
    return "unknown"

def perD_median(df: pd.DataFrame, metric: str, D_vals) -> pd.DataFrame:
    """Median per diffusion for All + each category."""
    out = []
    for D in D_vals:
        sub = df[df["Diffusion"] == D]
        row = {"Diffusion": D}
        # overall
        row["All"] = np.nanmedian(pd.to_numeric(sub[metric], errors="coerce"))
        # categories
        for cat in ["homogeneous", "heterogeneous", "sensors"]:
            sc = sub[sub["Category"] == cat]
            row[cat] = np.nanmedian(pd.to_numeric(sc[metric], errors="coerce")) if len(sc) else np.nan
        out.append(row)
    return pd.DataFrame(out)

# Fixed color palette
COLORS = {
    "All": "#d62728",            # red
    "homogeneous": "#1f77b4",    # blue
    "heterogeneous": "#ff7f0e",  # orange
    "sensors": "#2ca02c",        # green
}

def plot_metric(df_med: pd.DataFrame, metric_name: str, y_label: str, out_path: Path):
    """
    One chart per metric, matching the previous look, with fixed colors:
      - 7x4.5 inches, dpi=140
      - 4 lines (All + 3 categories), markers, grid, legend, ScalarFormatter on x.
    """
    x = df_med["Diffusion"].values
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)

    ax.plot(x, df_med["All"].values,           marker="o", label="All",           color=COLORS["All"])
    ax.plot(x, df_med["homogeneous"].values,   marker="o", label="Homogeneous",   color=COLORS["homogeneous"])
    ax.plot(x, df_med["heterogeneous"].values, marker="o", label="Heterogeneous", color=COLORS["heterogeneous"])
    ax.plot(x, df_med["sensors"].values,       marker="o", label="Sensors",       color=COLORS["sensors"])

    ax.set_xlabel("Diffusion (D)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{metric_name} vs. Diffusion — per-D medians")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------- main -----------------------

def main(in_csv: str, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(in_csv)
    df.columns = [c.strip() for c in df.columns]

    # Required columns
    if "Params" not in df.columns or "Diffusion" not in df.columns:
        raise ValueError("CSV must contain at least 'Params' and 'Diffusion' columns.")

    df["Diffusion"] = pd.to_numeric(df["Diffusion"], errors="coerce")

    # Build label + category
    df["Label"] = df["Params"].map(to_label)
    df["Category"] = df["Label"].map(categorize)

    # Keep only the categories of interest
    df = df[df["Category"].isin(["homogeneous", "heterogeneous", "sensors"])].copy()

    # Identify available metrics
    metrics = []
    if "BFS_Composite" in df.columns:       metrics.append(("BFS_Composite", "Median BFS_Composite", out / "GT_trend_BFS.png"))
    if "Midpoint_Composite" in df.columns:  metrics.append(("Midpoint_Composite", "Median Midpoint_Composite", out / "GT_trend_Midpoint.png"))
    if "time50_Midpoint" in df.columns:     metrics.append(("time50_Midpoint", "Median t50 (steps)", out / "GT_trend_t50.png"))

    if not metrics:
        raise ValueError("No expected metric columns found (BFS_Composite, Midpoint_Composite, time50_Midpoint).")

    # Diffusion values sorted
    D_vals = sorted(df["Diffusion"].dropna().unique())

    # Build medians per metric and save tidy CSV
    med_tables = {}
    for mcol, _, _ in metrics:
        med_tables[mcol] = perD_median(df, mcol, D_vals)

    tidy = pd.concat([t.assign(Metric=name) for name, t in med_tables.items()], ignore_index=True)
    tidy.to_csv(out / "GT_perD_medians_by_category.csv", index=False)

    # Plots
    for mcol, ylabel, png_path in metrics:
        plot_metric(med_tables[mcol], mcol, ylabel, png_path)

    print("Saved:")
    for _, _, png_path in metrics:
        print(" -", png_path)
    print(" -", out / "GT_perD_medians_by_category.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", default="old_t50.csv", help="Path to old_t50.csv")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()
    main(args.in_csv, args.outdir)
