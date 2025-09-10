#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attractant-only merge (intersection on Extra keyset), symmetry handled, row-wise numeric mean.

Usage:
python3 AggFiles/merge_attractant_only.py \
  --extra-summary ./GOODDATA/FullSimulationDataExtra/AttBalanceData/CS_AttractantMetrics_summary_S500C50_Maze_Final.csv \
  --full-summary  ./GOODDATA/FullSimulationData_PF/AttBalanceData/AttractantBalanceSummary.csv \
  --extra-subpops ./GOODDATA/FullSimulationDataExtra/AttBalanceData/CS_AttractantMetrics_subpops_S500C50_Maze_Final.csv \
  --full-subpops  ./GOODDATA/FullSimulationData_PF/AttBalanceData/AttractantBalanceSubpops.csv \
  --outdir        ./GOODDATA/IntersectionFin




Outputs (in --outdir):
  - summary_combined_raw.csv
  - summary_aggregated_by_key.csv
  - subpops_combined_raw.csv
  - subpops_aggregated_by_key.csv
  - coverage_summary.csv
  - notes.txt
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

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

def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Diffusion, fR, sR (parse from Params if needed); add fR_sorted/sR_sorted (canonical)."""
    out = df.copy()

    # Normalise Diffusion name if needed
    if "Diffusion" not in out.columns:
        for alt in ("diffusion", "D", "diffusion_BFS", "diffusion_Mid"):
            if alt in out.columns:
                out = out.rename(columns={alt: "Diffusion"})
                break

    # Ensure fR/sR
    if not {"fR", "sR"}.issubset(out.columns):
        if "Params" in out.columns:
            frsr = out["Params"].apply(parse_fr_sr)
            out["fR"] = [a for a, b in frsr]
            out["sR"] = [b for a, b in frsr]
        else:
            # try common variants (best-effort)
            fr_guess = next((c for c in out.columns if c.lower().strip() in {"fr","f_r","f-rate","frate"}), None)
            sr_guess = next((c for c in out.columns if c.lower().strip() in {"sr","s_r","s-rate","srate"}), None)
            if fr_guess and sr_guess:
                out = out.rename(columns={fr_guess: "fR", sr_guess: "sR"})

    # Numeric casts where present
    for c in ("fR","sR","Diffusion"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Canonical symmetric keys
    if {"fR","sR"}.issubset(out.columns):
        out["fR_sorted"] = np.round(np.minimum(out["fR"], out["sR"]), 12)
        out["sR_sorted"] = np.round(np.maximum(out["fR"], out["sR"]), 12)

    return out

def intersect_append_aggregate(extra_df: pd.DataFrame, full_df: pd.DataFrame, domain_tag: str):
    """
    Intersection on Extra keyset; append Extra + filtered Full; aggregate numeric means per key.
    Returns: combined_raw, aggregated_by_key, coverage_dict
    """
    e = standardise(extra_df).assign(source="extra")
    f = standardise(full_df ).assign(source="full")

    need = {"Diffusion","fR_sorted","sR_sorted"}
    if not need.issubset(e.columns):
        missing = sorted(need - set(e.columns))
        raise KeyError(f"[{domain_tag}] Extra is missing {missing} after standardisation.")

    keys = e[list(need)].drop_duplicates()
    f_on = f.merge(keys, on=list(need), how="inner")

    cov = {
        "domain": domain_tag,
        "n_keys_extra": len(keys),
        "n_keys_full_on_extra": f_on[list(need)].drop_duplicates().shape[0],
    }
    cov["coverage_ratio"] = (cov["n_keys_full_on_extra"] / cov["n_keys_extra"]) if cov["n_keys_extra"] else np.nan

    # Combined raw: preserve ALL original columns (union)
    combined_raw = pd.concat([e, f_on], ignore_index=True, sort=False)

    # Aggregated per key: row-wise numeric mean
    group_cols = ["Diffusion","fR_sorted","sR_sorted"]
    num_cols   = combined_raw.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols   = [c for c in num_cols if c not in group_cols]
    aggregated = (combined_raw[group_cols + agg_cols]
                  .groupby(group_cols, as_index=False)
                  .mean())

    return combined_raw, aggregated, cov

def main(extra_summary, full_summary, extra_subpops, full_subpops, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    # Load files
    e_sum = pd.read_csv(extra_summary)
    f_sum = pd.read_csv(full_summary)
    e_sub = pd.read_csv(extra_subpops)
    f_sub = pd.read_csv(full_subpops)

    # Merge summary
    sum_raw, sum_agg, cov_sum = intersect_append_aggregate(e_sum, f_sum, "summary")
    sum_raw.to_csv(out / "summary_combined_raw.csv", index=False)
    sum_agg.to_csv(out / "summary_aggregated_by_key.csv", index=False)

    # Merge subpops
    sub_raw, sub_agg, cov_sub = intersect_append_aggregate(e_sub, f_sub, "subpops")
    sub_raw.to_csv(out / "subpops_combined_raw.csv", index=False)
    sub_agg.to_csv(out / "subpops_aggregated_by_key.csv", index=False)

    # Coverage
    pd.DataFrame([cov_sum, cov_sub]).to_csv(out / "coverage_summary.csv", index=False)

    # Notes
    notes = f"""Attractant CSV merge (intersection on Extra/CS keyset).
- Standardised columns (Diffusion, fR, sR; parsed from Params when needed).
- Canonical symmetric keys: fR_sorted=min(fR,sR), sR_sorted=max(fR,sR) (1e-12 rounding).
- Keyset from Extra; Full filtered to those keys (intersection-only).
- Appended rows (Extra + filtered Full) and aggregated per key via row-wise numeric mean.
Outputs:
  summary_combined_raw.csv
  summary_aggregated_by_key.csv
  subpops_combined_raw.csv
  subpops_aggregated_by_key.csv
  coverage_summary.csv
"""
    (out / "notes.txt").write_text(notes, encoding="utf-8")

    print("=== DONE ===")
    print(f"Saved to: {out.resolve()}")
    print("Files: summary_* , subpops_* , coverage_summary.csv, notes.txt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-summary", required=True, help="CS_* summary CSV (Extra)")
    ap.add_argument("--full-summary",  required=True, help="Full summary CSV")
    ap.add_argument("--extra-subpops", required=True, help="CS_* subpops CSV (Extra)")
    ap.add_argument("--full-subpops",  required=True, help="Full subpops CSV")
    ap.add_argument("--outdir",        default="AttrMerge_Only", help="Output directory")
    args = ap.parse_args()
    main(args.extra_summary, args.full_summary, args.extra_subpops, args.full_subpops, args.outdir)
