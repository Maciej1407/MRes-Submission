#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_json_cli.py
-------------------

Builds aggregate and (optionally) epoch-level metrics from BFS/Midpoint JSONs.

Usage (aggregate only):
  python process_json_cli.py \
    --bfs GOODDATA/FullSimulationData_P1/BFS_Scores_S500C50_FILL.json \
    --mid GOODDATA/FullSimulationData_P1/Midpoint_Metrics_S500C50_FILL.json \
    --gen-path GOODDATA/FullSimulationData_P1/TEST \
    --generate-pop-csv \
    --generate-cat-csv \
    --box-plots \
    --box-plots-path CSVs/Test/BoxPlots

Usage (also save epoch-resolution table):
  python process_json_cli.py \
    --bfs GOODDATA/FullSimulationData_P1/BFS_Scores_S500C50_FILL.json \
    --mid GOODDATA/FullSimulationData_P1/Midpoint_Metrics_S500C50_FILL.json \
    --gen-path GOODDATA/FullSimulationData_P1/TEST \
    --return-epochs
"""

import os, re, math, json, gzip, argparse, warnings
import numpy as np
import pandas as pd

# -----------------------
# Optional plotting deps
# -----------------------
try:
    import plotly.express as px  # for pretty boxplots if available
except Exception:
    px = None
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except Exception:
    sns = None
    plt = None


# =======================
# Helpers you provided
# =======================
def plot_by_category(df_work=None, outdir="", metrics=None, **_):
    """
    Create boxplots by category and save PNGs.
    - Uses Plotly if available; else falls back to seaborn/matplotlib (if available).
    - Accepts extra kwargs (ignored) for compatibility.
    Returns list of saved file paths.
    """
    os.makedirs(outdir, exist_ok=True)

    name_map = {"equal": "homogeneous", "hetero": "heterogeneous"}
    df = df_work.copy()
    df["category_plot"] = df["category"].map(name_map)

    if metrics is None:
        # Try a reasonable default set if not provided
        metrics = [c for c in ["BFS_Composite", "Midpoint_Composite", "time50_Midpoint"] if c in df.columns]
    else:
        metrics = [m for m in metrics if m in df.columns]

    saved = []
    cat_order_plot = [nm for k, nm in name_map.items() if nm in df["category_plot"].unique()]
    color_map_plot = {"homogeneous": "#1f77b4", "heterogeneous": "#ff7f0e"}

    for m in metrics:
        png_path = os.path.join(outdir, f"box_{m}_by_category.png")
        try:
            if px is None:
                raise RuntimeError("Plotly not available")
            fig = px.box(
                df, x="category_plot", y=m, color="category_plot",
                category_orders={"category_plot": cat_order_plot},
                color_discrete_map=color_map_plot,
                points="all",
                hover_data=[col for col in ["Params", "Diffusion"] if col in df.columns],
                title=f"{m} by category"
            )
            fig.update_layout(xaxis_title="Category", yaxis_title=m, legend_title="Category")
            # Try static export; if kaleido is missing, fallback below
            fig.write_image(png_path)  # requires kaleido
        except Exception:
            if (sns is None) or (plt is None):
                warnings.warn("Plotly static export failed and seaborn/matplotlib not available; skipping plot.")
                continue
            plt.figure(figsize=(6, 4))
            order = cat_order_plot
            palette = [color_map_plot.get(c, None) for c in order]
            sns.boxplot(data=df, x="category_plot", y=m, order=order, palette=palette)
            sns.stripplot(data=df, x="category_plot", y=m, order=order, color="k", alpha=0.4, jitter=True)
            plt.title(f"{m} by category")
            plt.xlabel("Category")
            plt.ylabel(m)
            plt.tight_layout()
            plt.savefig(png_path, dpi=200)
            plt.close()
        saved.append(png_path)

    return saved


def label_sort_key(lbl: str):
    """
    Returns a tuple for sorting:
      (is_hetero, fR_value, sR_value_or_-inf)
    """
    if ',' not in lbl:
        # homogeneous: "fR=0.01"
        fR = float(lbl.split('=')[1])
        return (0, fR, -np.inf)
    else:
        # heterogeneous: "fR=0.01, sR=0.001"
        fR_str, sR_str = lbl.split(',')
        fR = float(fR_str.split('=')[1])
        sR = float(sR_str.split('=')[1])
        return (1, fR, sR)


def parse_rates(param_key: str):
    fR = None
    sR = None
    is_het = True
    for part in param_key.split("__"):
        if part.startswith("fR:"):
            fR = float(part.split(":", 1)[1])
        elif part.startswith("sR:"):
            if "single" in part:
                is_het = False
            else:
                sR = float(part.split(":", 1)[1])
    return ([fR, sR], True) if is_het else ([fR], False)


def load_json(path_or_obj):
    """
    Load JSON from a path (.json or .json.gz) OR return dict if input already a dict.
    """
    if isinstance(path_or_obj, dict):
        return path_or_obj
    if path_or_obj is None:
        raise ValueError("load_json: path_or_obj is None")

    path = str(path_or_obj)
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                d = json.load(f)
        else:
            with open(path, "r") as f:
                d = json.load(f)
        print(f"[OK] Loaded {path} with {len(d)} top-level keys")
        return d
    except Exception as e:
        print(f"[ERROR] Loading {path}: {e}")
        return {}


def simplify_label(params_str):
    # params_str now looks like "fR:0.01+sR:single" or "fR:0.01+sR:0.001"
    parts = params_str.split('+')
    # Always have an fR part
    fR_val = parts[0].split(':')[1]
    # Check if there's an sR part
    if len(parts) == 1 or 'single' in parts[1]:
        return f"fR={fR_val}"
    else:
        sR_val = parts[1].split(':')[1]
        return f"fR={fR_val}, sR={sR_val}"


def extract_rate_tokens(param_key: str):
    """
    Return the exact rate tokens as strings as they appear in the param key,
    preserving formatting (e.g., '0.001', not float-str()).
    Also returns a boolean that mirrors the 'single' flag logic:
    - True  → heterogeneous (two groups expected)
    - False → homogeneous/single (one group)
    """
    f_tok = None
    s_tok = None
    is_het = True
    for part in param_key.split("__"):
        if part.startswith("fR:"):
            f_tok = part.split(":", 1)[1]
        elif part.startswith("sR:"):
            if "single" in part:
                is_het = False
            else:
                s_tok = part.split(":", 1)[1]
    tokens = [t for t in (f_tok, s_tok) if t is not None]
    return tokens, is_het


def parse_snapshot_time(s: str):
    """
    Extract integer timestep from keys like 'cells_200.feather'.
    Returns None if parsing fails.
    """
    # Fast path:
    try:
        return int(s.split(".", 1)[0].split("_", 1)[1])
    except Exception:
        pass
    # Fallback with regex:
    m = re.search(r'cells_(\d+)\.feather$', s)
    return int(m.group(1)) if m else None


def first_crossing_time(pairs):
    """
    Given a list of (t, value) sorted by t, return earliest t with value >= 0.5.
    Returns np.nan if no crossing occurs.
    """
    for t, v in pairs:
        if v is not None and v >= 0.5:
            return t
    return np.nan


# ---------------------------------------------------------------
# UPDATED analyze_parameters: adds subpopulation metrics
# ---------------------------------------------------------------
def analyze_parameters(data, metric_name, epsilon=1e-6):
    results = []
    for i, (param_key, diff_dict) in enumerate(data.items(), 1):
        print(f"\n----\nParam #{i}: {param_key}")
        rates, is_het = parse_rates(param_key)              # <- keep existing behavior
        rate_tokens, _is_het_tokens = extract_rate_tokens(param_key)  # exact string tokens for naming/lookup
        # Treat equality as homogeneous ONLY for subpop-filling logic (do NOT change pop fairness logic)
        is_homog_case = (not is_het) or (len(set(rate_tokens)) == 1)

        print(f"  Rates (floats): {rates}, heterogeneous={is_het}")
        print(f"  Rate tokens: {rate_tokens} (used for keys & column naming)")

        for diff_key, epochs in diff_dict.items():
            D = float(diff_key.split("_", 1)[1])

            # Epoch-wise aggregates
            pop_epoch_maxes = []
            pop_time50s     = []  # only used for Midpoint

            # Subpopulation containers keyed by exact token (e.g., '0.001')
            sub_epoch_maxes = {tok: [] for tok in rate_tokens}
            sub_time50s     = {tok: [] for tok in rate_tokens}  # only used for Midpoint

            for epoch_key, cells in epochs.items():
                if not cells:
                    continue

                # Gather time series within this epoch
                pop_series = []
                sub_series = {tok: [] for tok in rate_tokens}

                for snap_key, snap in cells.items():
                    t = parse_snapshot_time(snap_key)
                    if t is None:
                        continue

                    # population value at t
                    if 'pop' in snap:
                        pop_series.append((t, snap['pop']))

                    # subpop values at t by exact token
                    for tok in rate_tokens:
                        k = f"c-{tok}"
                        if k in snap:
                            sub_series[tok].append((t, snap[k]))

                # If we have any population values, compute epoch-level summaries
                if pop_series:
                    pop_series.sort(key=lambda p: p[0])
                    pop_vals = [v for _, v in pop_series if v is not None]
                    if pop_vals:
                        pop_epoch_maxes.append(max(pop_vals))
                    if metric_name.lower() == "midpoint":
                        pop_time50s.append(first_crossing_time(pop_series))

                # Subpopulation epoch-level summaries
                for tok in rate_tokens:
                    seq = sub_series[tok]
                    if seq:
                        seq.sort(key=lambda p: p[0])
                        vals = [v for _, v in seq if v is not None]
                        if vals:
                            sub_epoch_maxes[tok].append(max(vals))
                        if metric_name.lower() == "midpoint":
                            sub_time50s[tok].append(first_crossing_time(seq))

            if not pop_epoch_maxes:
                print(f"    D={D}: no valid epochs → skipped")
                continue

            # -------------------------
            # Population-level summaries
            # -------------------------
            overall      = float(np.mean(pop_epoch_maxes))
            consistency  = 1.0 - (float(np.max(pop_epoch_maxes)) - float(np.min(pop_epoch_maxes)))

            if is_het:
                # fairness via CV over subpop means (keep original behavior)
                avgs = []
                for tok in rate_tokens:
                    vals = sub_epoch_maxes.get(tok, [])
                    if vals:
                        avgs.append(float(np.mean(vals)))
                if avgs:
                    mu = float(np.mean(avgs))
                    sd = float(np.std(avgs))
                    fairness = max(0.0, 1.0 - sd / (mu + epsilon))
                    hg_disp  = (min(avgs) / (max(avgs) + epsilon)) if len(avgs) > 1 else 0.0
                else:
                    fairness = 0.0
                    hg_disp  = 0.0
            else:
                # homogeneous/single: use epoch variability
                arr = np.array(pop_epoch_maxes, dtype=float)
                mu  = float(arr.mean())
                sd  = float(arr.std())
                fairness = max(0.0, 1.0 - sd / (mu + epsilon))
                hg_disp  = 0.0

            composite = overall * fairness

            row = {
                'Params': param_key.replace("__", "+").replace("fR:", "fR:").replace("sR:", "sR:"),
                'Diffusion': D,
                f'{metric_name}_Overall':      overall,
                f'{metric_name}_Consistency':  consistency,
                f'{metric_name}_Fairness':     fairness,
                'hg_disparity':                hg_disp,
                f'{metric_name}_Composite':    composite,
            }

            # time50 at population level (Midpoint only)
            if metric_name.lower() == "midpoint":
                row['time50_Midpoint'] = float(np.nanmean(pop_time50s)) if len(pop_time50s) else np.nan

            # -------------------------
            # Subpopulation-level adds
            # -------------------------
            # We DO NOT compute fairness/composite/hg for subpops (inter-group metrics).
            # For homogeneous cases (equal rates or 'single'), set subpop = pop values.
            for tok in rate_tokens:
                # Overall / Consistency
                if is_homog_case:
                    sub_overall     = overall
                    sub_consistency = consistency
                else:
                    vals = sub_epoch_maxes.get(tok, [])
                    if vals:
                        sub_overall     = float(np.mean(vals))
                        sub_consistency = 1.0 - (float(np.max(vals)) - float(np.min(vals)))
                    else:
                        sub_overall     = np.nan
                        sub_consistency = np.nan

                row[f'{metric_name}_Overall_c{tok}']     = sub_overall
                row[f'{metric_name}_Consistency_c{tok}'] = sub_consistency

                # time50 for Midpoint subpops
                if metric_name.lower() == "midpoint":
                    if is_homog_case:
                        sub_t50 = row['time50_Midpoint'] if 'time50_Midpoint' in row else np.nan
                    else:
                        t50s = sub_time50s.get(tok, [])
                        sub_t50 = float(np.nanmean(t50s)) if len(t50s) else np.nan
                    row[f'time50_Midpoint_c{tok}'] = sub_t50

            print(f"    D={D}: Overall={overall:.3f}, Fairness={fairness:.3f}, hg_disp={hg_disp:.3f} "
                  f"+ subpop columns: {', '.join([f'c{tok}' for tok in rate_tokens])}")

            results.append(row)

    df = pd.DataFrame(results)
    print(f"\n>> {metric_name} analysis produced {len(df)} rows "
          f"(with subpopulation columns where applicable)")
    return df


# =======================
# process_json (with epochs)
# =======================
def process_json(pathBFS=None, pathMid=None,
                 generate_pop_csv=False,
                 generate_cat_csv=False,
                 gen_path="",
                 box_plots=False,
                 box_plots_path="",
                 skip_json_load=False,
                 loaded_df=None,
                 return_epochs=False):
    """
    Returns:
      - by default: aggregated DataFrame (unchanged from your workflow)
      - if return_epochs=True: (aggregated_df, epoch_level_df)
    """
    if gen_path:
        os.makedirs(gen_path, exist_ok=True)
    if box_plots_path:
        os.makedirs(box_plots_path, exist_ok=True)

    # --- Load or use provided df (aggregate path uses analyze_parameters) ---
    if not skip_json_load:
        bfs_data = load_json(pathBFS)
        mid_data = load_json(pathMid)

        mid_df = analyze_parameters(mid_data, 'Midpoint')
        bfs_df = analyze_parameters(bfs_data, 'BFS')

        full_df = pd.merge(
            bfs_df, mid_df,
            on=['Params', 'Diffusion'],
            suffixes=('_BFS', '_Mid')
        )
        # keep your original hetero & labels logic so nothing else changes
        full_df['is_heterogeneous'] = ~full_df['Params'].str.contains("single")
        print("Unique Params:", full_df['Params'].unique())
        full_df['Label'] = full_df['Params'].apply(simplify_label)

    else:
        full_df = loaded_df.copy()
        full_df['is_heterogeneous'] = ~full_df['Params'].str.contains("single")
        full_df['Label'] = full_df['Params'].apply(simplify_label)

    # --- Optional population CSVs ---
    if generate_pop_csv:
        values = ["BFS_Composite", "Midpoint_Composite", "time50_Midpoint"]
        for v in values:
            if v not in full_df.columns:
                continue
            pop_pivot = full_df.pivot(index="Label", columns="Diffusion", values=v)
            sorted_labels = sorted(pop_pivot.index, key=label_sort_key)
            pop_pivot = pop_pivot.reindex(sorted_labels)
            pop_pivot.to_csv(os.path.join(gen_path, f"{v}.csv"))

    # --- Optional category CSVs & boxplots ---
    if generate_cat_csv:
        num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

        def parse_params(label: str):
            s = str(label)
            m_eq = re.search(rf"fR\s*=\s*sR\s*=\s*({num})", s, re.I)
            if m_eq:
                v = float(m_eq.group(1)); return v, v
            m_fr = re.search(rf"fR\s*[:=]\s*({num})", s, re.I)
            m_sr = re.search(rf"sR\s*[:=]\s*({num})", s, re.I)
            if m_fr and m_sr:
                return float(m_fr.group(1)), float(m_sr.group(1))
            m_single = re.search(rf"single\s*[:=]?\s*({num})?", s, re.I)
            if m_single:
                if m_single.group(1):
                    v = float(m_single.group(1)); return v, v
                anynum = re.search(num, s)
                if anynum:
                    v = float(anynum.group(0)); return v, v
            if m_fr and not m_sr:
                v = float(m_fr.group(1)); return v, v
            if m_sr and not m_fr:
                v = float(m_sr.group(1)); return v, v
            return np.nan, np.nan

        df_work = full_df.copy()
        fR, sR = zip(*df_work["Params"].map(parse_params))
        df_work["fR"] = pd.to_numeric(fR, errors="coerce")
        df_work["sR"] = pd.to_numeric(sR, errors="coerce")
        df_work = df_work.dropna(subset=["fR","sR"]).copy()
        df_work = df_work.loc[~((df_work["fR"]==0) & (df_work["sR"]==0))].copy()
        df_work["Diffusion"] = pd.to_numeric(df_work["Diffusion"], errors="coerce")

        cond_equal   = (df_work["fR"]>0) & (df_work["sR"]>0) & (df_work["fR"]==df_work["sR"])
        cond_mix     = ((df_work["fR"]==0) ^ (df_work["sR"]==0))
        cond_hetero  = (df_work["fR"]>0) & (df_work["sR"]>0) & (df_work["fR"]!=df_work["sR"])
        df_work["category"] = np.select([cond_equal, cond_mix, cond_hetero], ["equal","mix","hetero"], default=np.nan)
        df_work = df_work.dropna(subset=["category"]).copy()

        metrics = [c for c in ["BFS_Composite","Midpoint_Composite","time50_Midpoint"] if c in df_work.columns]
        D_all = sorted(df_work["Diffusion"].dropna().unique().tolist())

        for cat in ["equal","hetero","mix"]:
            sub_cat = df_work[df_work["category"]==cat]
            if sub_cat.empty:
                continue
            for m in metrics:
                pvt = sub_cat.pivot_table(index="Label", columns="Diffusion", values=m, aggfunc="median")
                pvt = pvt.reindex(columns=D_all)
                pvt.to_csv(os.path.join(gen_path, f"{cat}_{m.lower()}_pivot.csv"))

        if box_plots:
            label_map = {"equal": "homogeneous", "hetero": "heterogeneous", "mix": "sensor"}
            df_plot = df_work.copy()
            df_plot["category"] = df_plot["category"].map(label_map)

            metrics_available = set(df_plot.columns)
            plot_metrics = []
            plot_metrics.append("BFS_Composite" if "BFS_Composite" in metrics_available else
                                ("BFS_Overall" if "BFS_Overall" in metrics_available else None))
            plot_metrics.append("Midpoint_Composite" if "Midpoint_Composite" in metrics_available else
                                ("Midpoint_Overall" if "Midpoint_Overall" in metrics_available else None))
            plot_metrics.append("time50_Midpoint" if "time50_Midpoint" in metrics_available else None)
            plot_metrics = [m for m in plot_metrics if m is not None]

            if plot_metrics:
                saved_files = plot_by_category(
                    df_work=df_plot,
                    outdir=box_plots_path,
                    metrics=plot_metrics,
                    category_col="category",
                    diffusion_col="Diffusion",
                    cats=["heterogeneous","homogeneous","sensor"],
                    dpi=220,
                    fmt="png"
                )
                print(f"Boxplots saved: {len(saved_files)}")

    # ------------------------------
    # Epoch-level table (optional)
    # ------------------------------
    epoch_df = None
    if not skip_json_load and return_epochs:
        def _num_from_key(s: str):
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            return float(m.group(0)) if m else np.nan

        def _epoch_metrics(data, metric_name):
            rows = []
            for param_key, diff_dict in data.items():
                rate_tokens, is_het = extract_rate_tokens(param_key)
                is_homog_case = (not is_het) or (len(set(rate_tokens)) == 1)
                params_norm = param_key.replace("__", "+").replace("fR:", "fR:").replace("sR:", "sR:")
                for diff_key, epochs in diff_dict.items():
                    D = _num_from_key(diff_key)
                    for epoch_key, cells in epochs.items():
                        # parse epoch index if possible
                        try:
                            epoch_idx = int(epoch_key.split("_", 1)[1])
                        except Exception:
                            epoch_idx = None

                        if not cells:
                            continue

                        pop_series = []
                        sub_series = {tok: [] for tok in rate_tokens}

                        for snap_key, snap in cells.items():
                            t = parse_snapshot_time(snap_key)
                            if t is None:
                                continue
                            if 'pop' in snap:
                                pop_series.append((t, snap['pop']))
                            for tok in rate_tokens:
                                k = f"c-{tok}"
                                if k in snap:
                                    sub_series[tok].append((t, snap[k]))

                        if not pop_series:
                            continue

                        pop_series.sort(key=lambda p: p[0])
                        pop_vals = [v for _, v in pop_series if v is not None]

                        # epoch-level summaries within this epoch
                        pop_overall = float(np.nanmax(pop_vals)) if pop_vals else np.nan
                        pop_consistency = (1.0 - (float(np.nanmax(pop_vals)) - float(np.nanmin(pop_vals)))) if pop_vals else np.nan

                        sub_overall_map, sub_cons_map = {}, {}
                        for tok in rate_tokens:
                            seq = sub_series[tok]
                            if seq:
                                seq.sort(key=lambda p: p[0])
                                vals = [v for _, v in seq if v is not None]
                                if vals:
                                    sub_overall_map[tok] = float(np.nanmax(vals))
                                    sub_cons_map[tok] = 1.0 - (float(np.nanmax(vals)) - float(np.nanmin(vals)))
                                else:
                                    sub_overall_map[tok] = np.nan
                                    sub_cons_map[tok] = np.nan
                            else:
                                sub_overall_map[tok] = np.nan
                                sub_cons_map[tok] = np.nan

                        # fairness per epoch
                        if is_homog_case:
                            arr = np.array(pop_vals, dtype=float)
                            mu = float(np.nanmean(arr))
                            sd = float(np.nanstd(arr))
                            fairness = max(0.0, 1.0 - sd / (mu + 1e-6)) if pop_vals else np.nan
                            hg_disp = 0.0
                        else:
                            avgs = [v for v in sub_overall_map.values() if v == v]
                            if avgs:
                                mu = float(np.mean(avgs))
                                sd = float(np.std(avgs))
                                fairness = max(0.0, 1.0 - sd / (mu + 1e-6))
                                hg_disp = (min(avgs) / (max(avgs) + 1e-6)) if len(avgs) > 1 else 0.0
                            else:
                                fairness = 0.0
                                hg_disp = 0.0

                        row = {
                            'Params': params_norm,
                            'Diffusion': D,
                            'Epoch': epoch_idx,
                            f'{metric_name}_Overall':     pop_overall,
                            f'{metric_name}_Consistency': pop_consistency,
                            f'{metric_name}_Fairness':    fairness,
                            'hg_disparity':               hg_disp,
                            f'{metric_name}_Composite':   pop_overall * fairness,
                        }

                        if metric_name.lower() == "midpoint":
                            row['time50_Midpoint'] = float(first_crossing_time(pop_series))

                        # subpop columns
                        for tok in rate_tokens:
                            row[f'{metric_name}_Overall_c{tok}']     = sub_overall_map.get(tok, np.nan)
                            row[f'{metric_name}_Consistency_c{tok}'] = sub_cons_map.get(tok, np.nan)
                            if metric_name.lower() == "midpoint":
                                seq = sub_series[tok]
                                t50 = first_crossing_time(seq) if seq else np.nan
                                row[f'time50_Midpoint_c{tok}'] = (float(t50) if (t50 is not None and t50 == t50) else np.nan)

                        rows.append(row)
            return pd.DataFrame(rows)

        bfs_data = load_json(pathBFS)
        mid_data = load_json(pathMid)
        bfs_epochs = _epoch_metrics(bfs_data, 'BFS')
        mid_epochs = _epoch_metrics(mid_data, 'Midpoint')

        epoch_df = pd.merge(
            bfs_epochs, mid_epochs,
            on=['Params','Diffusion','Epoch'],
            suffixes=('_BFS','_Mid')
        )

        epoch_df['is_heterogeneous'] = ~epoch_df['Params'].str.contains("single")
        epoch_df['Label'] = epoch_df['Params'].apply(simplify_label)

        # Save as a convenience if a gen_path is provided
        if gen_path:
            out_path = os.path.join(gen_path, "EpochLevel.csv")
            epoch_df.to_csv(out_path, index=False)
            print("Epoch-level DataFrame saved to:", out_path)

    # --- Return shape: backwards compatible by default ---
    if return_epochs and (epoch_df is not None):
        return full_df, epoch_df
    return full_df


# =======================
# CLI
# =======================
def _parse_args():
    p = argparse.ArgumentParser(description="Process BFS & Midpoint JSONs into aggregate and (optional) epoch-level CSVs.")
    p.add_argument("--bfs", required=True, help="Path to BFS JSON (.json or .json.gz)")
    p.add_argument("--mid", required=True, help="Path to Midpoint JSON (.json or .json.gz)")
    p.add_argument("--gen-path", default="Out", help="Directory to save outputs (pivots, EpochLevel.csv, Aggregated.csv)")
    p.add_argument("--generate-pop-csv", action="store_true", help="Write population pivot CSVs (BFS_Composite, Midpoint_Composite, time50_Midpoint)")
    p.add_argument("--generate-cat-csv", action="store_true", help="Write category pivot CSVs")
    p.add_argument("--box-plots", action="store_true", help="Create category boxplots (requires plotly or seaborn/mpl)")
    p.add_argument("--box-plots-path", default="", help="Directory for boxplots (defaults to <gen-path>/boxplots if not provided)")
    p.add_argument("--return-epochs", action="store_true", help="Also build & save epoch-level table (EpochLevel.csv)")
    p.add_argument("--aggregated-name", default="Aggregated.csv", help="Filename for aggregated CSV in gen-path")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.gen_path, exist_ok=True)
    box_dir = args.box_plots_path or os.path.join(args.gen_path, "boxplots")
    if args.box_plots:
        os.makedirs(box_dir, exist_ok=True)

    res = process_json(
        pathBFS=args.bfs,
        pathMid=args.mid,
        generate_pop_csv=args.generate_pop_csv,
        generate_cat_csv=args.generate_cat_csv,
        gen_path=args.gen_path,
        box_plots=args.box_plots,
        box_plots_path=box_dir,
        return_epochs=args.return_epochs
    )

    # Save aggregated DF
    if args.return_epochs:
        agg_df, epoch_df = res
        out_agg = os.path.join(args.gen_path, args.aggregated_name)
        agg_df.to_csv(out_agg, index=False)
        print("Aggregated DataFrame saved to:", out_agg)
        # EpochLevel.csv already saved inside process_json if gen_path present
    else:
        agg_df = res
        out_agg = os.path.join(args.gen_path, args.aggregated_name)
        agg_df.to_csv(out_agg, index=False)
        print("Aggregated DataFrame saved to:", out_agg)


if __name__ == "__main__":
    main()
