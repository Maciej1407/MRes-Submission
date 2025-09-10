"""
Usage:
python3 AggFiles/thesis_corr_and_plots_v2.py \
  --perf GOODDATA/IntersectionFin/aggregated_intersection.csv \
  --summary GOODDATA/IntersectionFin/summary_aggregated_by_key.csv \
  --outdir GOODDATA/IntersectionFin/CorrOutputs_IQR

"""


#!/usr/bin/env python3
# Save as: thesis_corr_and_plots_v2.py
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from math import atanh, tanh

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def fisher_ci(r, n):
    if n is None or n < 4 or np.isnan(r): return (np.nan, np.nan)
    z = atanh(r); se = 1/np.sqrt(n-3)
    return (tanh(z-1.96*se), tanh(z+1.96*se))

def BH_fdr(pvals):
    p = np.asarray(pvals, float); m = np.sum(~np.isnan(p))
    q = np.full_like(p, np.nan, float)
    if m == 0: return q
    nn = np.where(~np.isnan(p))[0]
    ord_idx = np.argsort(p[nn])
    ranks = np.arange(1, len(nn)+1, dtype=float)
    q_nn = p[nn][ord_idx] * m / ranks
    # monotone
    for i in range(len(q_nn)-2, -1, -1):
        q_nn[i] = min(q_nn[i], q_nn[i+1])
    q[nn[ord_idx]] = np.clip(q_nn, 0, 1)
    return q

def iqr_filter_perD(df):
    keep = []
    for D, sub in df.groupby('Diffusion'):
        def fences(s):
            q1, q3 = s.quantile([0.25, 0.75]); i = q3-q1
            return q1 - 1.5*i, q3 + 1.5*i
        fr = sub['fR_sorted'].astype(float)
        sr = sub['sR_sorted'].astype(float)
        lof,hif = fences(fr); los,his = fences(sr)
        mask = (fr>=lof)&(fr<=hif)&(sr>=los)&(sr<=his)
        keep.append(sub[mask])
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[[]]

def corr_pair(x, y):
    m = x.notna() & y.notna(); n = int(m.sum())
    if n < 3: return np.nan, np.nan, np.nan, np.nan, n
    r, rp   = pearsonr(x[m], y[m])
    rho, pp = spearmanr(x[m], y[m])
    return r, rp, rho, pp, n

def scatter_annotated(df, x, y, color_by, title, xlabel, ylabel, out_png):
    plt.figure(figsize=(6,4))
    for val in sorted(df[color_by].unique()):
        sub = df[df[color_by]==val]
        plt.scatter(sub[x], sub[y], alpha=0.7, label=f'{color_by}={val}')
    r, rp, rho, rhop, n = corr_pair(df[x], df[y])
    note = f'Pearson r={r:.2f} (p={rp:.1e})\nSpearman ρ={rho:.2f} (p={rhop:.1e})\nN={n}'
    plt.text(0.98, 0.02, note, transform=plt.gca().transAxes,
             ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    plt.legend(title='Diffusion'); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def save_bar_with_ci(df, xcol, ycol, lo, hi, title, ylabel, out_png, ylim=(-0.9,0.3)):
    xs = np.arange(len(df)); heights = df[ycol].values
    yerr = np.vstack([heights - df[lo].values, df[hi].values - heights])
    plt.figure(figsize=(6,4))
    plt.bar(xs, heights, yerr=yerr, capsize=5, width=0.6, alpha=0.85)
    plt.xticks(xs, df[xcol].astype(str).values)
    plt.ylabel(ylabel); plt.xlabel(xcol); plt.title(title)
    if ylim: plt.ylim(*ylim)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def main(perf_path, summary_path, outdir):
    ensure_dir(outdir)
    perf = pd.read_csv(perf_path)[['Diffusion','fR_sorted','sR_sorted','BFS','Mid']].copy()
    summ = pd.read_csv(summary_path).copy()
    merged = pd.merge(summ, perf, on=['Diffusion','fR_sorted','sR_sorted'], how='inner')

    # harmonise share-gap column name
    if 'sub_gini' not in merged.columns and 'share_gap' in merged.columns:
        merged['sub_gini'] = merged['share_gap']

    # -----------------------
    # (A) POOLED — IQR first
    # -----------------------
    filtered_all = iqr_filter_perD(merged)

    # pooled (IQR-filtered) scatters
    if 't25_cons_frac' in filtered_all.columns:
        scatter_annotated(filtered_all.dropna(subset=['t25_cons_frac','Mid']),
                          't25_cons_frac', 'Mid', 'Diffusion',
                          'Midpoint vs time to 25% depletion (pooled across D; IQR-filtered)',
                          'Time to 25% depletion (steps)', 'Midpoint Composite',
                          os.path.join(outdir, 'POOLED_IQR_scatter_t25_vs_mid.png'))
        scatter_annotated(filtered_all.dropna(subset=['t25_cons_frac','BFS']),
                          't25_cons_frac', 'BFS', 'Diffusion',
                          'BFS vs time to 25% depletion (pooled across D; IQR-filtered)',
                          'Time to 25% depletion (steps)', 'BFS Composite',
                          os.path.join(outdir, 'POOLED_IQR_scatter_t25_vs_bfs.png'))
    if 't50_cons_frac' in filtered_all.columns:
        scatter_annotated(filtered_all.dropna(subset=['t50_cons_frac','Mid']),
                          't50_cons_frac', 'Mid', 'Diffusion',
                          'Midpoint vs time to 50% depletion (pooled across D; IQR-filtered)',
                          'Time to 50% depletion (steps)', 'Midpoint Composite',
                          os.path.join(outdir, 'POOLED_IQR_scatter_t50_vs_mid.png'))
        scatter_annotated(filtered_all.dropna(subset=['t50_cons_frac','BFS']),
                          't50_cons_frac', 'BFS', 'Diffusion',
                          'BFS vs time to 50% depletion (pooled across D; IQR-filtered)',
                          'Time to 50% depletion (steps)', 'BFS Composite',
                          os.path.join(outdir, 'POOLED_IQR_scatter_t50_vs_bfs.png'))
    if 'sub_gini' in filtered_all.columns:
        scatter_annotated(filtered_all.dropna(subset=['sub_gini','Mid']),
                          'sub_gini', 'Mid', 'Diffusion',
                          'Midpoint vs sub-population gini (pooled across D; IQR-filtered)',
                          'Sub-population gini (share-gap)', 'Midpoint Composite',
                          os.path.join(outdir, 'POOLED_IQR_scatter_gini_vs_mid.png'))
        scatter_annotated(filtered_all.dropna(subset=['sub_gini','BFS']),
                          'sub_gini', 'BFS', 'Diffusion',
                          'BFS vs sub-population gini (pooled across D; IQR-filtered)',
                          'Sub-population gini (share-gap)', 'BFS Composite',
                          os.path.join(outdir, 'POOLED_IQR_scatter_gini_vs_bfs.png'))

    # also dump pooled correlations (on IQR-filtered pooled)
    pooled_rows = []
    for xcol,ycol,tag in [
        ('t25_cons_frac','Mid','t25_vs_mid'),
        ('t50_cons_frac','Mid','t50_vs_mid'),
        ('sub_gini','Mid','gini_vs_mid'),
        ('t25_cons_frac','BFS','t25_vs_bfs'),
        ('t50_cons_frac','BFS','t50_vs_bfs'),
        ('sub_gini','BFS','gini_vs_bfs'),
    ]:
        if xcol in filtered_all.columns and ycol in filtered_all.columns:
            r, rp, rho, rhop, n = corr_pair(filtered_all[xcol], filtered_all[ycol])
            pooled_rows.append({'pair':tag,'pearson_r':r,'pearson_p':rp,'spearman_rho':rho,'spearman_p':rhop,'N':n})
    if pooled_rows:
        pd.DataFrame(pooled_rows).to_csv(os.path.join(outdir,'POOLED_IQR_correlations.csv'), index=False)

    # ----------------------------------------------
    # (B) PER-D TABLES + BAR PLOTS (IQR-filtered per D)
    # ----------------------------------------------
    filt = filtered_all  # already IQR-filtered per D

    # t25/t50 vs Mid
    rec_mid = []
    for D, sub in filt.groupby('Diffusion'):
        r25, rp25, rho25, rhop25, n25 = (np.nan,)*5
        r50, rp50, rho50, rhop50, n50 = (np.nan,)*5
        if 't25_cons_frac' in sub.columns:
            r25, rp25, rho25, rhop25, n25 = corr_pair(sub['t25_cons_frac'], sub['Mid'])
        if 't50_cons_frac' in sub.columns:
            r50, rp50, rho50, rhop50, n50 = corr_pair(sub['t50_cons_frac'], sub['Mid'])
        rec_mid.append({'D':D,'rho_t25_mid':rho25,'p_t25_mid':rhop25,'N_t25_mid':n25,
                             'rho_t50_mid':rho50,'p_t50_mid':rhop50,'N_t50_mid':n50})
    mid_df = pd.DataFrame(rec_mid).sort_values('D')
    mid_df['q_t25_mid'] = BH_fdr(mid_df['p_t25_mid'].values)
    if 'p_t50_mid' in mid_df: mid_df['q_t50_mid'] = BH_fdr(mid_df['p_t50_mid'].values)
    mid_df.to_csv(os.path.join(outdir,'perD_mid_vs_timing.csv'), index=False)

    # t25/t50 vs BFS
    rec_bfs = []
    for D, sub in filt.groupby('Diffusion'):
        r25, rp25, rho25, rhop25, n25 = (np.nan,)*5
        r50, rp50, rho50, rhop50, n50 = (np.nan,)*5
        if 't25_cons_frac' in sub.columns:
            r25, rp25, rho25, rhop25, n25 = corr_pair(sub['t25_cons_frac'], sub['BFS'])
        if 't50_cons_frac' in sub.columns:
            r50, rp50, rho50, rhop50, n50 = corr_pair(sub['t50_cons_frac'], sub['BFS'])
        rec_bfs.append({'D':D,'rho_t25_bfs':rho25,'p_t25_bfs':rhop25,'N_t25_bfs':n25,
                             'rho_t50_bfs':rho50,'p_t50_bfs':rhop50,'N_t50_bfs':n50})
    bfs_df = pd.DataFrame(rec_bfs).sort_values('D')
    bfs_df['q_t25_bfs'] = BH_fdr(bfs_df['p_t25_bfs'].values)
    if 'p_t50_bfs' in bfs_df: bfs_df['q_t50_bfs'] = BH_fdr(bfs_df['p_t50_bfs'].values)
    bfs_df.to_csv(os.path.join(outdir,'perD_bfs_vs_timing.csv'), index=False)

    # gini vs Mid/BFS
    rec_g_mid, rec_g_bfs = [], []
    if 'sub_gini' in filt.columns:
        for D, sub in filt.groupby('Diffusion'):
            r_m, rp_m, rho_m, rhop_m, n_m = corr_pair(sub['sub_gini'], sub['Mid'])
            r_b, rp_b, rho_b, rhop_b, n_b = corr_pair(sub['sub_gini'], sub['BFS'])
            rec_g_mid.append({'D':D,'rho_gini_mid':rho_m,'p_gini_mid':rhop_m,'N_gini_mid':n_m})
            rec_g_bfs.append({'D':D,'rho_gini_bfs':rho_b,'p_gini_bfs':rhop_b,'N_gini_bfs':n_b})
    gmid = pd.DataFrame(rec_g_mid).sort_values('D'); gbfs = pd.DataFrame(rec_g_bfs).sort_values('D')
    if not gmid.empty:
        gmid['q_gini_mid'] = BH_fdr(gmid['p_gini_mid'].values)
        gmid.to_csv(os.path.join(outdir,'perD_gini_vs_mid.csv'), index=False)
    if not gbfs.empty:
        gbfs['q_gini_bfs'] = BH_fdr(gbfs['p_gini_bfs'].values)
        gbfs.to_csv(os.path.join(outdir,'perD_gini_vs_bfs.csv'), index=False)

    # bar plots (Spearman ρ with 95% CI)
    if not mid_df.empty and mid_df['rho_t25_mid'].notna().any():
        tmp = mid_df.copy()
        ci = [fisher_ci(r, n) for r,n in zip(tmp['rho_t25_mid'], tmp['N_t25_mid'])]
        tmp['lo'], tmp['hi'] = [c[0] for c in ci], [c[1] for c in ci]
        save_bar_with_ci(tmp, 'D', 'rho_t25_mid', 'lo', 'hi',
                         'Spearman ρ: t25 vs Midpoint (IQR-filtered)',
                         'ρ (95% CI)',
                         os.path.join(outdir, 'bar_t25_vs_mid.png'))
    if 'rho_t50_mid' in mid_df and mid_df['rho_t50_mid'].notna().any():
        tmp = mid_df.dropna(subset=['rho_t50_mid']).copy()
        ci = [fisher_ci(r, n) for r,n in zip(tmp['rho_t50_mid'], tmp['N_t50_mid'])]
        tmp['lo'], tmp['hi'] = [c[0] for c in ci], [c[1] for c in ci]
        save_bar_with_ci(tmp, 'D', 'rho_t50_mid', 'lo', 'hi',
                         'Spearman ρ: t50 vs Midpoint (IQR-filtered)',
                         'ρ (95% CI)',
                         os.path.join(outdir, 'bar_t50_vs_mid.png'))
    if not bfs_df.empty and bfs_df['rho_t25_bfs'].notna().any():
        tmp = bfs_df.copy()
        ci = [fisher_ci(r, n) for r,n in zip(tmp['rho_t25_bfs'], tmp['N_t25_bfs'])]
        tmp['lo'], tmp['hi'] = [c[0] for c in ci], [c[1] for c in ci]
        save_bar_with_ci(tmp, 'D', 'rho_t25_bfs', 'lo', 'hi',
                         'Spearman ρ: t25 vs BFS (IQR-filtered)',
                         'ρ (95% CI)',
                         os.path.join(outdir, 'bar_t25_vs_bfs.png'))
    if 'rho_t50_bfs' in bfs_df and bfs_df['rho_t50_bfs'].notna().any():
        tmp = bfs_df.dropna(subset=['rho_t50_bfs']).copy()
        ci = [fisher_ci(r, n) for r,n in zip(tmp['rho_t50_bfs'], tmp['N_t50_bfs'])]
        tmp['lo'], tmp['hi'] = [c[0] for c in ci], [c[1] for c in ci]
        save_bar_with_ci(tmp, 'D', 'rho_t50_bfs', 'lo', 'hi',
                         'Spearman ρ: t50 vs BFS (IQR-filtered)',
                         'ρ (95% CI)',
                         os.path.join(outdir, 'bar_t50_vs_bfs.png'))
    if not gmid.empty and gmid['rho_gini_mid'].notna().any():
        tmp = gmid.copy()
        ci = [fisher_ci(r, n) for r,n in zip(tmp['rho_gini_mid'], tmp['N_gini_mid'])]
        tmp['lo'], tmp['hi'] = [c[0] for c in ci], [c[1] for c in ci]
        save_bar_with_ci(tmp, 'D', 'rho_gini_mid', 'lo', 'hi',
                         'Spearman ρ: sub_gini vs Midpoint (IQR-filtered)',
                         'ρ (95% CI)',
                         os.path.join(outdir, 'bar_gini_vs_mid.png'))
    if not gbfs.empty and gbfs['rho_gini_bfs'].notna().any():
        tmp = gbfs.copy()
        ci = [fisher_ci(r, n) for r,n in zip(tmp['rho_gini_bfs'], tmp['N_gini_bfs'])]
        tmp['lo'], tmp['hi'] = [c[0] for c in ci], [c[1] for c in ci]
        save_bar_with_ci(tmp, 'D', 'rho_gini_bfs', 'lo', 'hi',
                         'Spearman ρ: sub_gini vs BFS (IQR-filtered)',
                         'ρ (95% CI)',
                         os.path.join(outdir, 'bar_gini_vs_bfs.png'))

    print(f'Done. Outputs in: {outdir}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--perf', required=True, help='aggregated_intersection.csv')
    ap.add_argument('--summary', required=True, help='summary_aggregated_by_key.csv')
    ap.add_argument('--outdir', required=True, help='output directory')
    args = ap.parse_args()
    main(args.perf, args.summary, args.outdir)

