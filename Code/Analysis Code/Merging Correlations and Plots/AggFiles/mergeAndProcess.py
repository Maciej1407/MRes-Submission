#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intersection-only merge & aggregation for chemotaxis simulations.

python3 MergeAndProcess.py \
  --extra ./GOODDATA/FullSimulationData_PF/ExtraSimData.csv \
  --full ./GOODDATA/FullSimulationData_PF/FullSimulationData_PF.csv \
  --outdir ./GOODDATA/IntersectionFin \
  --make-heatmaps




What this script does (replicates the pipeline we agreed on):
1) Load ExtraSimData.csv and FullSimulationData_PF.csv.
2) Standardise: ensure Diffusion column exists; parse fR/sR from Params if needed.
3) Canonicalise symmetric pairs: fR_sorted = min(fR,sR), sR_sorted = max(fR,sR) (rounded to 1e-12).
4) Build keyset K from Extra: {(Diffusion, fR_sorted, sR_sorted)}.
5) Filter Full to keys in K (intersection only) so it contributes replicates, not new parameters.
6) Append Extra + filtered Full (row-wise).
7) Aggregate (row-wise mean) per key:
      BFS = mean(BFS_Composite)
      Mid = mean(Midpoint_Composite)
      n_total / n_extra / n_full
8) Save:
   - combined_raw_intersection.csv (preserves ALL original columns + added fields)
   - aggregated_intersection.csv
   - coverage_summary.csv
   - symmetric_duplicates_in_full.csv
   - full_replicate_check.csv
   - (optional) per-D heatmaps for BFS & Midpoint (+ CSV pivot grids)
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Optional plotting (only used when --make-heatmaps is passed)
try:
    import matplotlib
    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
PARAM_RE = re.compile(rf"fR\s*[:=]\s*({NUM_RE})\s*\+\s*sR\s*[:=]\s*({NUM_RE})", re.I)

def parse_fr_sr(text: str):
    """Parse fR and sR from 'Params'-like strings."""
    s = str(text)
    m = PARAM_RE.search(s)
    if m:
        return float(m.group(1)), float(m.group(2))
    # fallbacks: find any two numbers
    nums = re.findall(NUM_RE, s)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return np.nan, np.nan

def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns: Diffusion, fR, sR, BFS_Composite, Midpoint_Composite, and add fR_sorted/sR_sorted."""
    out = df.copy()

    # Diffusion column
    if "Diffusion" not in out.columns:
        for alt in ("diffusion", "D", "diffusion_BFS", "diffusion_Mid"):
            if alt in out.columns:
                out = out.rename(columns={alt: "Diffusion"})
                break

    # Parse fR/sR from Params if missing
    if not {"fR", "sR"}.issubset(out.columns):
        if "Params" not in out.columns:
            raise KeyError("Need 'fR'/'sR' or a 'Params' column to parse rates.")
        frsr = out["Params"].apply(parse_fr_sr)
        out["fR"] = [a for a, b in frsr]
        out["sR"] = [b for a, b in frsr]

    # Enforce numeric types
    for c in ("fR", "sR", "Diffusion", "BFS_Composite", "Midpoint_Composite"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Canonical symmetric keys
    out["fR_sorted"] = np.round(np.minimum(out["fR"], out["sR"]), 12)
    out["sR_sorted"] = np.round(np.maximum(out["fR"], out["sR"]), 12)

    return out

def coverage_summary(keys_extra: pd.DataFrame, full_on_extra: pd.DataFrame) -> pd.DataFrame:
    n_keys_extra = len(keys_extra)
    n_keys_full = full_on_extra[["Diffusion", "fR_sorted", "sR_sorted"]].drop_duplicates().shape[0]
    ratio = n_keys_full / n_keys_extra if n_keys_extra else np.nan
    return pd.DataFrame({"n_keys_extra": [n_keys_extra],
                         "n_keys_full_on_extra": [n_keys_full],
                         "coverage_ratio": [ratio]})

def check_full_symmetry(full: pd.DataFrame) -> pd.DataFrame:
    """List canonical keys where Full has both (fR,sR) and (sR,fR) (pure symmetry duplicates)."""
    full = full.copy()
    full["pair_str"] = full["fR"].round(12).astype(str) + "|" + full["sR"].round(12).astype(str)
    g = (full.groupby(["Diffusion", "fR_sorted", "sR_sorted"], as_index=False)
               .agg(n_rows=("pair_str", "size"),
                    pairs=("pair_str", lambda s: sorted(set(s)))))
    return g[g["n_rows"] == 2].sort_values(["Diffusion", "fR_sorted", "sR_sorted"])

def replicate_counts_full(full_on_extra: pd.DataFrame) -> pd.DataFrame:
    """How many Full rows per canonical key (intersection only)."""
    return (full_on_extra.groupby(["Diffusion", "fR_sorted", "sR_sorted"], as_index=False)
                        .size()
                        .rename(columns={"size": "n_full_reps"}))

def aggregate_intersection(pooled: pd.DataFrame) -> pd.DataFrame:
    """Row-wise mean per key; also report source counts."""
    agg = (pooled.groupby(["Diffusion", "fR_sorted", "sR_sorted"], as_index=False)
                  .agg(BFS=("BFS_Composite", "mean"),
                       Mid=("Midpoint_Composite", "mean"),
                       n_total=("BFS_Composite", "size"),
                       n_extra=("source", lambda s: (s == "extra").sum()),
                       n_full =("source", lambda s: (s == "full").sum())))
    return agg

# ---------- Heatmap helpers ----------
def _log_edges_from_centers(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lg = np.log10(v)
    if v.size == 1:
        pad = 0.25
        edges = np.array([lg[0] - pad, lg[0] + pad])
    else:
        mids = 0.5 * (lg[:-1] + lg[1:])
        first = lg[0] - (mids[0] - lg[0])
        last  = lg[-1] + (lg[-1] - mids[-1])
        edges = np.concatenate([[first], mids, [last]])
    return 10.0 ** edges

def _fmt_plain(x: float, max_dec: int = 12) -> str:
    s = f"{float(x):.{max_dec}f}".rstrip("0").rstrip(".")
    return s if s else "0"

def plot_heatmaps(
    df,
    value_col="BFS_Composite",
    outdir="Out_Maps",
    decimals=12,
    vmin=None, vmax=None,
    global_xlim=None, global_ylim=None,
    figsize=6.0, dpi=180,
    tick_fontsize=8, rotate_xticks=90,
    uniform_cells=False,           # <<<<<<<<<< add this
):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Ensure needed columns exist and numeric
    need = {"Diffusion","fR","sR",value_col}
    w = df[list(need)].copy()
    for c in need: w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[(w["fR"] > 0) & (w["sR"] > 0)].dropna(subset=list(need))

    # Round to merge exact duplicates; average duplicates
    w["fR_r"] = w["fR"].round(decimals)
    w["sR_r"] = w["sR"].round(decimals)
    agg = w.groupby(["Diffusion","fR_r","sR_r"], as_index=False)[value_col].mean()

    # Color scale
    if vmin is None or vmax is None:
        vals = pd.to_numeric(agg[value_col], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
        if len(vals):
            if vmin is None: vmin = float(vals.min())
            if vmax is None: vmax = float(vals.max())
            if vmin == vmax: vmin -= 1e-12; vmax += 1e-12

    saved = []
    for D, sub in agg.groupby("Diffusion", sort=True):
        f_vals = np.sort(sub["fR_r"].unique())
        s_vals = np.sort(sub["sR_r"].unique())
        grid = (sub.pivot(index="fR_r", columns="sR_r", values=value_col)
                  .reindex(index=f_vals, columns=s_vals))

        # Save CSV grid
        slug = str(D).replace(".", "p")
        csv_path = Path(outdir) / f"{value_col}_grid_D{slug}.csv"
        grid.to_csv(csv_path)

        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)

        if uniform_cells:
            # --- UNIFORM SQUARE CELLS (index space) ---
            # Build integer edges 0..n for a regular lattice
            nx, ny = len(s_vals), len(f_vals)
            x_edges = np.arange(nx+1)
            y_edges = np.arange(ny+1)

            pc = ax.pcolormesh(x_edges, y_edges, grid.values, shading="flat", vmin=vmin, vmax=vmax)

            # Ticks at cell centers with your original numeric labels
            ax.set_xticks(np.arange(nx) + 0.5)
            ax.set_xticklabels([_fmt_plain(v) for v in s_vals], rotation=rotate_xticks, fontsize=tick_fontsize)
            ax.set_yticks(np.arange(ny) + 0.5)
            ax.set_yticklabels([_fmt_plain(v) for v in f_vals], fontsize=tick_fontsize)

            # Equal squares, no log scaling (labels still the real values)
            ax.set_xlim(0, nx); ax.set_ylim(0, ny)
            ax.set_aspect("equal", adjustable="box")

            # Optional light grid for aesthetics:
            # ax.set_xticks(np.arange(nx+1), minor=True)
            # ax.set_yticks(np.arange(ny+1), minor=True)
            # ax.grid(which="minor", color="k", alpha=0.08, linewidth=0.5)

        else:
            # --- ORIGINAL (geometric spacing; cell area follows numeric spacing) ---
            x_edges = _log_edges_from_centers(grid.columns.values)
            y_edges = _log_edges_from_centers(grid.index.values)
            pc = ax.pcolormesh(x_edges, y_edges, grid.values, shading="flat", vmin=vmin, vmax=vmax)

            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xticks(s_vals)
            ax.set_xticklabels([_fmt_plain(v) for v in s_vals], rotation=rotate_xticks, fontsize=tick_fontsize)
            ax.set_yticks(f_vals)
            ax.set_yticklabels([_fmt_plain(v) for v in f_vals], fontsize=tick_fontsize)
            ax.set_aspect("equal", adjustable="box")

        fig.colorbar(pc, ax=ax, label=value_col)
        ax.set_xlabel("sR"); ax.set_ylabel("fR")
        ax.set_title(f"{value_col} — fR vs sR (Diffusion={D})")
        ax.minorticks_off()
        fig.tight_layout()

        png_path = Path(outdir) / f"{value_col}_grid_D{slug}.png"
        fig.savefig(png_path); plt.close(fig)
        saved.append((str(csv_path), str(png_path)))

    return saved


def main(extra_path: str, full_path: str, outdir: str, make_heatmaps: bool = False):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    # 1) Load & standardise
    extra = standardise(pd.read_csv(extra_path)).assign(source="extra")
    full  = standardise(pd.read_csv(full_path)).assign(source="full")

    # 2) Note symmetry in Full (for transparency)
    symm = check_full_symmetry(full)
    symm.to_csv(out / "symmetric_duplicates_in_full.csv", index=False)

    # 3) Intersection keyset from Extra
    keys_extra = extra[["Diffusion", "fR_sorted", "sR_sorted"]].drop_duplicates().copy()

    # 4) Filter Full to Extra’s keys (intersection only)
    full_on_extra = full.merge(keys_extra, on=["Diffusion", "fR_sorted", "sR_sorted"], how="inner")

    # 5) Coverage + Full replicate counts
    coverage = coverage_summary(keys_extra, full_on_extra)
    coverage.to_csv(out / "coverage_summary.csv", index=False)

    rep_counts = replicate_counts_full(full_on_extra)
    rep_counts.to_csv(out / "full_replicate_check.csv", index=False)

    # 6) Combined RAW (preserve ALL original cols), then intersect
    #    (Union of columns; rows are only Extra + Full_on_Extra)
    combined_raw = pd.concat([extra, full_on_extra], ignore_index=True, sort=False)
    combined_raw.to_csv(out / "combined_raw_intersection.csv", index=False)

    # 7) Aggregated (row-wise mean per key)
    agg = aggregate_intersection(combined_raw)
    agg.to_csv(out / "aggregated_intersection.csv", index=False)

    # 8) Optional heatmaps
    if make_heatmaps:
        hm_base = out / "Heatmaps"
        plot_heatmaps(agg.rename(columns={
                                            "BFS": "BFS_Composite",
                                            "fR_sorted": "fR",
                                            "sR_sorted": "sR" }),

                      value_col="BFS_Composite",
                      outdir=hm_base / "BFS",
                      uniform_cells=True)
        plot_heatmaps(agg.rename(columns={"Mid": "Midpoint_Composite",
                                          "fR_sorted": "fR",
                                          "sR_sorted": "sR" }),
                      value_col="Midpoint_Composite",
                      outdir=hm_base / "Midpoint",
                      uniform_cells=True)

    # Console summary
    print("=== DONE ===")
    print(f"Coverage: {coverage.to_dict('records')[0]}")
    print(f"Combined RAW (intersection): {len(combined_raw)} rows")
    print(f"Aggregated (intersection): {len(agg)} keys")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--extra", required=True, help="Path to ExtraSimData.csv")
    p.add_argument("--full",  required=True, help="Path to FullSimulationData_PF.csv")
    p.add_argument("--outdir", default="Intersection_Final", help="Output directory")
    p.add_argument("--make-heatmaps", action="store_true", help="Generate heatmaps and CSV grids")
    args = p.parse_args()
    main(args.extra, args.full, args.outdir, make_heatmaps=args.make_heatmaps)
