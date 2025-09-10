#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlate performance vs attractant metrics from ALREADY-MERGED CSVs.


python3 AggFiles/corr_from_merged.py \
  --perf-agg ./GOODDATA/IntersectionFin/aggregated_intersection.csv \
  --attr-summary-agg ./GOODDATA/IntersectionFin/summary_aggregated_by_key.csv \
  --attr-subpops-agg ./GOODDATA/IntersectionFin/subpops_aggregated_by_key.csv \
  --outdir ./GOODDATA/IntersectionFin/CorrelationFiles



Inputs:
  --perf-agg           CSV with columns at least: Diffusion, (fR_sorted,sR_sorted) or (fR,sR),
                       and performance metrics (preferably 'BFS' and 'Mid'; will accept
                       'BFS_Composite'/'Midpoint_Composite' and map to BFS/Mid).
  --attr-summary-agg   CSV with columns at least: Diffusion, (fR_sorted,sR_sorted) or (fR,sR),
                       and attractant columns like 't25_cons_frac', 't50_cons_frac' (kept as-is).
  --attr-subpops-agg   (optional) CSV with imbalance metrics like 'sub_entropy_bits','sub_hhi',
                       'sub_gini','share_gap' keyed by Diffusion and rates.

Outputs (in --outdir):
  - analysis_merged_intersection.csv
  - analysis_merged_intersection_IQR.csv
  - correlations_by_diffusion_raw.csv
  - correlations_by_diffusion_IQR.csv
  - perD_sample_sizes.csv
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats

# ---------- helpers ----------
NUM_RE   = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
PARAM_RE = re.compile(rf"fR\s*[:=]\s*({NUM_RE})\s*\+\s*sR\s*[:=]\s*({NUM_RE})", re.I)

def parse_fr_sr(text: str):
    s = str(text)
    m = PARAM_RE.search(s)
    if m:
        return float(m.group(1)), float(m.group(2))
    nums = re.findall(NUM_RE, s)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return np.nan, np.nan

def ensure_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Diffusion, fR_sorted, sR_sorted numeric keys exist (compute from fR/sR if needed)."""
    out = df.copy()

    # Normalise Diffusion name if needed
    if "Diffusion" not in out.columns:
        for alt in ("diffusion", "D", "diffusion_BFS", "diffusion_Mid"):
            if alt in out.columns:
                out = out.rename(columns={alt: "Diffusion"})
                break

    # Ensure fR/sR exist or parse from Params
    if not {"fR","sR"}.issubset(out.columns):
        if "Params" in out.columns:
            frsr = out["Params"].apply(parse_fr_sr)
            out["fR"] = [a for a,b in frsr]
            out["sR"] = [b for a,b in frsr]

    # If only sorted keys are present, that's fine; otherwise compute them
    if {"fR_sorted","sR_sorted"}.issubset(out.columns):
        pass
    elif {"fR","sR"}.issubset(out.columns):
        out["fR_sorted"] = np.round(np.minimum(pd.to_numeric(out["fR"], errors="coerce"),
                                               pd.to_numeric(out["sR"], errors="coerce")), 12)
        out["sR_sorted"] = np.round(np.maximum(pd.to_numeric(out["fR"], errors="coerce"),
                                               pd.to_numeric(out["sR"], errors="coerce")), 12)
    else:
        raise KeyError("Need either (fR_sorted,sR_sorted) or (fR,sR) (or a Params column) to build keys.")

    # Cast numerics
    for c in ("Diffusion","fR_sorted","sR_sorted","fR","sR"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def standardise_perf_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Accept BFS / Mid or BFS_Composite / Midpoint_Composite
    if "BFS" not in out.columns and "BFS_Composite" in out.columns:
        out = out.rename(columns={"BFS_Composite":"BFS"})
    if "Mid" not in out.columns and "Midpoint_Composite" in out.columns:
        out = out.rename(columns={"Midpoint_Composite":"Mid"})
    # Cast numerics if present
    for c in ("BFS","Mid"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def iqr_filter_perD(df: pd.DataFrame) -> pd.DataFrame:
    """Within each diffusion, keep rows inside Tukey fences for both fR_sorted and sR_sorted."""
    kept = []
    for D, sub in df.groupby("Diffusion"):
        def bounds(a):
            a = np.asarray(a, dtype=float)
            q1 = np.nanpercentile(a, 25)
            q3 = np.nanpercentile(a, 75)
            iqr = q3 - q1
            return q1 - 1.5*iqr, q3 + 1.5*iqr
        lo_f, hi_f = bounds(sub["fR_sorted"].values)
        lo_s, hi_s = bounds(sub["sR_sorted"].values)
        mask = (sub["fR_sorted"] >= lo_f) & (sub["fR_sorted"] <= hi_f) & \
               (sub["sR_sorted"] >= lo_s) & (sub["sR_sorted"] <= hi_s)
        kept.append(sub[mask])
    return pd.concat(kept, ignore_index=True) if kept else df.copy()

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR q-values aligned to pvals (NaNs preserved)."""
    p = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(p)
    p_use = p[mask]
    m = len(p_use)
    q = np.full_like(p, np.nan, dtype=float)
    if m == 0:
        return q
    order = np.argsort(p_use)
    ranks = np.arange(1, m+1)
    adj   = p_use[order] * (m / ranks)
    # monotone
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    qvals = np.empty_like(p_use)
    qvals[order] = adj
    q[mask] = qvals
    return q

def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-D Pearson & Spearman between performance and attractant metrics."""
    perf_vars   = [c for c in ["BFS","Mid"] if c in df.columns]
    # Keep your names as-is (they are times in your data even if 'frac' appears)
    attract_cols = [c for c in ["t25_cons_frac","t50_cons_frac","t25_time","t50_time"] if c in df.columns]
    imb_cols     = [c for c in ["sub_entropy_bits","sub_hhi","sub_gini","share_gap"] if c in df.columns]
    targets = attract_cols + imb_cols

    rows = []
    for D, sub in df.groupby("Diffusion"):
        for pv in perf_vars:
            for tv in targets:
                x = pd.to_numeric(sub[pv], errors="coerce")
                y = pd.to_numeric(sub[tv], errors="coerce")
                mask = x.notna() & y.notna()
                x = x[mask]; y = y[mask]
                n = len(x)
                if n < 3:
                    rows.append({"D":D, "perf":pv, "attr":tv, "N":n,
                                 "pearson_r":np.nan, "pearson_p":np.nan,
                                 "spearman_rho":np.nan, "spearman_p":np.nan})
                    continue
                pr, pp = stats.pearsonr(x, y)
                sr, sp = stats.spearmanr(x, y)
                rows.append({"D":D, "perf":pv, "attr":tv, "N":n,
                             "pearson_r":pr, "pearson_p":pp,
                             "spearman_rho":sr, "spearman_p":sp})
    out = pd.DataFrame(rows)
    # BH–FDR on Spearman within each D
    q_list = []
    for D, sub in out.groupby("D"):
        qvals = bh_fdr(sub["spearman_p"].values)
        q_list.append(pd.Series(qvals, index=sub.index))
    out["spearman_q"] = pd.concat(q_list).sort_index().values if q_list else np.nan
    return out

# ---------- main ----------
def main(perf_agg_path, attr_summary_agg_path, attr_subpops_agg_path, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    # Load and standardise keys/columns
    perf = ensure_keys(pd.read_csv(perf_agg_path))
    perf = standardise_perf_columns(perf)

    summ = ensure_keys(pd.read_csv(attr_summary_agg_path))
    # do NOT rename t25/t50 columns here; keep as-is

    if attr_subpops_agg_path:
        subp = ensure_keys(pd.read_csv(attr_subpops_agg_path))
    else:
        subp = None

    # Join: inner perf + summary, then left subpops if provided
    keycols = ["Diffusion","fR_sorted","sR_sorted"]
    analysis = perf.merge(summ, on=keycols, how="inner")
    if subp is not None and not subp.empty:
        analysis = analysis.merge(subp, on=keycols, how="left")

    # Save merged analysis table (raw)
    (out / "analysis_merged_intersection.csv").write_text(
        analysis.to_csv(index=False), encoding="utf-8"
    )

    # IQR filter per D on fR_sorted/sR_sorted
    analysis_iqr = iqr_filter_perD(analysis)
    (out / "analysis_merged_intersection_IQR.csv").write_text(
        analysis_iqr.to_csv(index=False), encoding="utf-8"
    )

    # Correlations
    corr_raw = corr_table(analysis)
    corr_iqr = corr_table(analysis_iqr)

    corr_raw.to_csv(out / "correlations_by_diffusion_raw.csv", index=False)
    corr_iqr.to_csv(out / "correlations_by_diffusion_IQR.csv", index=False)

    # Sample sizes per D (raw vs IQR)
    perD_N = (analysis.groupby("Diffusion").size().rename("N_all")
              .to_frame().join(analysis_iqr.groupby("Diffusion").size().rename("N_IQR"), how="left")
              .reset_index())
    perD_N.to_csv(out / "perD_sample_sizes.csv", index=False)

    print("=== DONE ===")
    print(f"Saved to: {out.resolve()}")
    print("Files: analysis_merged_intersection{,_IQR}.csv, correlations_by_diffusion_{raw,IQR}.csv, perD_sample_sizes.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-agg",           required=True, help="Merged performance CSV (e.g., intersection_combined_agg.csv)")
    ap.add_argument("--attr-summary-agg",   required=True, help="Merged attractant summary CSV (e.g., summary_aggregated_by_key.csv)")
    ap.add_argument("--attr-subpops-agg",   default="",    help="Merged attractant subpops CSV (optional)")
    ap.add_argument("--outdir",             default="AttrPerf_Corrs_FromMerged", help="Output directory")
    args = ap.parse_args()
    main(args.perf_agg, args.attr_summary_agg, args.attr_subpops_agg or None, args.outdir)
