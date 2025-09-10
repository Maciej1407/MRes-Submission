#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intersection-only merge & aggregation for chemotaxis simulations.

Examples
--------
# Epoch-level inputs (has an Epoch column):
python3 MergeAndProcess3.py \
  --extra ./GOODDATA/epoch_merge/ExtraSimData.csv \
  --full  ./GOODDATA/epoch_merge/FullSimulationData_PF_Epochs.csv \
  --outdir ./GOODDATA/IntersectionFin/epochRes \
  --epoch-res

# Aggregate (original behaviour; no Epoch in keys):
python3 MergeAndProcess2.py \
  --extra ./GOODDATA/IntersectionFin/ExtraSimData.csv \
  --full  ./GOODDATA/IntersectionFin/FullSimulationData_PF.csv \
  --outdir ./GOODDATA/IntersectionFin \
  --make-heatmaps
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Optional plotting (only used when --make-heatmaps and not --epoch-res)
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
    """
    Ensure columns: Diffusion, fR, sR (parse from Params if needed),
    and add canonical symmetric keys fR_sorted/sR_sorted.
    If an Epoch column exists (any casing/variant), normalise it to numeric 'Epoch'.
    """
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
        out["fR"] = [a for a, _ in frsr]
        out["sR"] = [b for _, b in frsr]

    # Enforce numeric types if present
    for c in ("fR", "sR", "Diffusion", "BFS_Composite", "Midpoint_Composite"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Canonical symmetric keys
    out["fR_sorted"] = np.round(np.minimum(out["fR"], out["sR"]), 12)
    out["sR_sorted"] = np.round(np.maximum(out["fR"], out["sR"]), 12)

    # Normalise Epoch column if present
    if "Epoch" not in out.columns:
        for alt in ("epoch", "Epoch_BFS", "Epoch_Mid"):
            if alt in out.columns:
                out = out.rename(columns={alt: "Epoch"})
                break
    if "Epoch" in out.columns:
        out["Epoch"] = pd.to_numeric(out["Epoch"], errors="coerce")

    return out


def check_full_symmetry(full: pd.DataFrame) -> pd.DataFrame:
    """List canonical keys where Full has both (fR,sR) and (sR,fR) (pure symmetry duplicates)."""
    full = full.copy()
    full["pair_str"] = full["fR"].round(12).astype(str) + "|" + full["sR"].round(12).astype(str)
    g = (
        full.groupby(["Diffusion", "fR_sorted", "sR_sorted"], as_index=False)
            .agg(n_rows=("pair_str", "size"),
                 pairs=("pair_str", lambda s: sorted(set(s))))
    )
    return g[g["n_rows"] == 2].sort_values(["Diffusion", "fR_sorted", "sR_sorted"])


def aggregate_intersection(pooled: pd.DataFrame, keys=("Diffusion", "fR_sorted", "sR_sorted"), t50_agg: str = "mean") -> pd.DataFrame:
    """
    Row-wise aggregation per `keys`, reporting:
      - BFS = mean(BFS_Composite)
      - Mid = mean(Midpoint_Composite)
      - t50* = <t50_agg>(time50_Midpoint*) if present
      - n_total / n_extra / n_full
    """
    tmp = pooled.copy()

    # Coerce relevant columns to numeric for safe aggregation
    for c in ("BFS_Composite", "Midpoint_Composite"):
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # Detect and coerce any time50 columns
    t50_cols = [c for c in tmp.columns if c.startswith("time50_Midpoint")]
    for c in t50_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # Means
    agg_specs = {"BFS_Composite": "mean", "Midpoint_Composite": "mean"}
    for c in t50_cols:
        agg_specs[c] = t50_agg

    means = (
        tmp.groupby(list(keys), as_index=False)
           .agg(agg_specs)
           .rename(columns={
               "BFS_Composite": "BFS",
               "Midpoint_Composite": "Mid",
               **{c: c.replace("time50_Midpoint", "t50") for c in t50_cols}
           })
    )

    # Replicate counts
    counts = (
        tmp.groupby(list(keys), as_index=False)
           .agg(n_total=("source", "size"),
                n_extra=("source", lambda s: (s == "extra").sum()),
                n_full =("source", lambda s: (s == "full").sum()))
    )

    out = means.merge(counts, on=list(keys), how="left")

    # Stable column order: keys + BFS/Mid + t50* + counts
    front = list(keys) + ["BFS", "Mid"]
    t50_renamed = [c.replace("time50_Midpoint", "t50") for c in t50_cols]
    tail = ["n_total", "n_extra", "n_full"]
    ordered = [c for c in (front + t50_renamed + tail) if c in out.columns]
    return out.loc[:, ordered]


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
    uniform_cells=False,
):
    if plt is None:
        return []

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure needed columns exist and numeric
    need = {"Diffusion", "fR", "sR", value_col}
    w = df[list(need)].copy()
    for c in need:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[(w["fR"] > 0) & (w["sR"] > 0)].dropna(subset=list(need))

    # Round to merge exact duplicates; average duplicates
    w["fR_r"] = w["fR"].round(decimals)
    w["sR_r"] = w["sR"].round(decimals)
    agg = w.groupby(["Diffusion", "fR_r", "sR_r"], as_index=False)[value_col].mean()

    # Color scale
    if vmin is None or vmax is None:
        vals = pd.to_numeric(agg[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals):
            if vmin is None:
                vmin = float(vals.min())
            if vmax is None:
                vmax = float(vals.max())
            if vmin == vmax:
                vmin -= 1e-12
                vmax += 1e-12

    saved = []
    for D, sub in agg.groupby("Diffusion", sort=True):
        f_vals = np.sort(sub["fR_r"].unique())
        s_vals = np.sort(sub["sR_r"].unique())
        grid = (
            sub.pivot(index="fR_r", columns="sR_r", values=value_col)
               .reindex(index=f_vals, columns=s_vals)
        )

        # Save CSV grid
        slug = str(D).replace(".", "p")
        csv_path = outdir / f"{value_col}_grid_D{slug}.csv"
        grid.to_csv(csv_path)

        fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)

        if uniform_cells:
            nx, ny = len(s_vals), len(f_vals)
            x_edges = np.arange(nx + 1)
            y_edges = np.arange(ny + 1)
            pc = ax.pcolormesh(x_edges, y_edges, grid.values, shading="flat", vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(nx) + 0.5)
            ax.set_xticklabels([_fmt_plain(v) for v in s_vals], rotation=rotate_xticks, fontsize=tick_fontsize)
            ax.set_yticks(np.arange(ny) + 0.5)
            ax.set_yticklabels([_fmt_plain(v) for v in f_vals], fontsize=tick_fontsize)
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.set_aspect("equal", adjustable="box")
        else:
            x_edges = _log_edges_from_centers(grid.columns.values)
            y_edges = _log_edges_from_centers(grid.index.values)
            pc = ax.pcolormesh(x_edges, y_edges, grid.values, shading="flat", vmin=vmin, vmax=vmax)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(s_vals)
            ax.set_xticklabels([_fmt_plain(v) for v in s_vals], rotation=rotate_xticks, fontsize=tick_fontsize)
            ax.set_yticks(f_vals)
            ax.set_yticklabels([_fmt_plain(v) for v in f_vals], fontsize=tick_fontsize)
            ax.set_aspect("equal", adjustable="box")

        fig.colorbar(pc, ax=ax, label=value_col)
        ax.set_xlabel("sR")
        ax.set_ylabel("fR")
        ax.set_title(f"{value_col} — fR vs sR (Diffusion={D})")
        ax.minorticks_off()
        fig.tight_layout()

        png_path = outdir / f"{value_col}_grid_D{slug}.png"
        fig.savefig(png_path)
        plt.close(fig)
        saved.append((str(csv_path), str(png_path)))

    return saved


def main(extra_path: str, full_path: str, outdir: str, make_heatmaps: bool = False, epoch_res: bool = False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Load & standardise
    extra = standardise(pd.read_csv(extra_path)).assign(source="extra")
    full = standardise(pd.read_csv(full_path)).assign(source="full")

    # 2) Symmetry report (for transparency)
    symm = check_full_symmetry(full)
    symm.to_csv(out / "symmetric_duplicates_in_full.csv", index=False)

    # 3) Choose key set (with or without Epoch)
    key_cols = ["Diffusion", "fR_sorted", "sR_sorted"] + (["Epoch"] if epoch_res else [])

    # 4) Build intersection keyset from Extra and filter Full to it
    keys_extra = extra[key_cols].drop_duplicates().copy()
    full_on_extra = full.merge(keys_extra, on=key_cols, how="inner")

    # 5) Coverage & Full replicate counts (by chosen keys)
    def _coverage(keys_extra_df, full_on_extra_df):
        n_keys_extra = len(keys_extra_df)
        n_keys_full = full_on_extra_df[key_cols].drop_duplicates().shape[0]
        ratio = n_keys_full / n_keys_extra if n_keys_extra else np.nan
        return pd.DataFrame({"n_keys_extra": [n_keys_extra],
                             "n_keys_full_on_extra": [n_keys_full],
                             "coverage_ratio": [ratio]})

    def _rep_counts(full_on_extra_df):
        return (
            full_on_extra_df.groupby(key_cols, as_index=False)
                            .size()
                            .rename(columns={"size": "n_full_reps"})
        )

    coverage = _coverage(keys_extra, full_on_extra)
    coverage.to_csv(out / "coverage_summary.csv", index=False)

    rep_counts = _rep_counts(full_on_extra)
    rep_counts.to_csv(out / "full_replicate_check.csv", index=False)

    # 6) Combined RAW (preserve ALL original columns; this keeps symmetric orientations if present)
    combined_raw = pd.concat([extra, full_on_extra], ignore_index=True, sort=False)
    combined_raw.to_csv(out / "combined_raw_intersection.csv", index=False)

    # 7) Aggregated on chosen keys (collapses symmetric duplicates via fR_sorted/sR_sorted)
    agg = aggregate_intersection(combined_raw, keys=tuple(key_cols))
    agg_name = "aggregated_intersectionepochs.csv" if epoch_res else "aggregated_intersection.csv"
    agg.to_csv(out / agg_name, index=False)

    # 8) Optional heatmaps (only meaningful in non-epoch mode)
    if make_heatmaps and not epoch_res:
        hm_base = out / "Heatmaps"
        plot_heatmaps(
            agg.rename(columns={"BFS": "BFS_Composite", "fR_sorted": "fR", "sR_sorted": "sR"}),
            value_col="BFS_Composite",
            outdir=hm_base / "BFS",
            uniform_cells=True,
        )
        plot_heatmaps(
            agg.rename(columns={"Mid": "Midpoint_Composite", "fR_sorted": "fR", "sR_sorted": "sR"}),
            value_col="Midpoint_Composite",
            outdir=hm_base / "Midpoint",
            uniform_cells=True,
        )

    # Console summary
    print("=== DONE ===")
    print(f"Coverage: {coverage.to_dict('records')[0]}")
    print(f"Combined RAW (intersection): {len(combined_raw)} rows")
    print(f"Aggregated ({'epoch' if epoch_res else 'per-pair'}) keys: {len(agg)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--extra", required=True, help="Path to ExtraSimData.csv")
    p.add_argument("--full", required=True, help="Path to FullSimulationData_PF.csv (or *_Epochs.csv for epoch mode)")
    p.add_argument("--outdir", default="Intersection_Final", help="Output directory")
    p.add_argument("--makesheatmaps", action="store_true", help="Generate heatmaps and CSV grids (non-epoch mode only)")
    p.add_argument("--epoch-res", action="store_true", help="Treat inputs as epoch-level tables; include Epoch in keys")
    args = p.parse_args()

    main(args.extra, args.full, args.outdir, make_heatmaps=False, epoch_res=args.epoch_res)
