import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Paths (adjust if needed)
# --------------------------
SPARE_RUN_FILE = os.environ.get("SPARSE_INPUT", "collectedData/old_t50.csv")
SAVE_DIR = os.environ.get("SPARSE_FIG_OUT", "SparseFiguresColored")
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------
# Helpers
# --------------------------
NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def parse_params_pair(params: str):
    """
    Parse 'Params' strings like:
      'fR:0.1+sR:single', 'fR:0.01+sR:0.0', 'fR=1e-3+sR=1e-3', 'single:0.05' (legacy)
    Returns (fR: float, sR: float) where 'single' → duplicates the other rate.
    Unknowns → (np.nan, np.nan).
    """
    s = str(params)
    # fR and sR may be 'single' or numeric
    m_fr = re.search(rf"fR\s*[:=]\s*({NUM_RE}|single)", s, flags=re.I)
    m_sr = re.search(rf"sR\s*[:=]\s*({NUM_RE}|single)", s, flags=re.I)

    fr_str = m_fr.group(1) if m_fr else None
    sr_str = m_sr.group(1) if m_sr else None

    def _to_float(x):
        if x is None:
            return np.nan
        x = x.strip().lower()
        if x == "single":
            return "single"
        try:
            return float(x)
        except Exception:
            return np.nan

    fr = _to_float(fr_str)
    sr = _to_float(sr_str)

    # Legacy "single" without explicit fr/sr: try to find any number
    if (isinstance(fr, float) and np.isnan(fr)) and (isinstance(sr, float) and np.isnan(sr)):
        m_single = re.search(rf"single\s*[:=]\s*({NUM_RE})", s, flags=re.I)
        if m_single:
            v = float(m_single.group(1))
            return v, v

    # If one is 'single', copy the other
    if fr == "single" and isinstance(sr, (int, float)):
        fr = float(sr)
    if sr == "single" and isinstance(fr, (int, float)):
        sr = float(fr)

    # If still 'single', or still nan, try to infer any sole number
    if fr == "single" and (not isinstance(sr, (int, float))):
        m_any = re.search(NUM_RE, s)
        if m_any:
            v = float(m_any.group(0))
            fr = sr = v
    if sr == "single" and (not isinstance(fr, (int, float))):
        m_any = re.search(NUM_RE, s)
        if m_any:
            v = float(m_any.group(0))
            fr = sr = v

    return float(fr) if pd.notna(fr) else np.nan, float(sr) if pd.notna(sr) else np.nan


def classify_category(fr: float, sr: float) -> str:
    """Return one of: 'Homogeneous', 'Heterogeneous', 'Pair containing sensor'."""
    if not (np.isfinite(fr) and np.isfinite(sr)):
        return "Heterogeneous"
    # Sensor pair if either is exactly zero
    if np.isclose(fr, 0.0) or np.isclose(sr, 0.0):
        return "Pair containing sensor"
    if np.isclose(fr, sr):
        return "Homogeneous"
    return "Heterogeneous"


def ensure_category_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "category" in df.columns and df["category"].notna().any():
        # Normalize capitalization to our three labels
        mapping = {
            "homogeneous": "Homogeneous",
            "heterogeneous": "Heterogeneous",
            "pair containing sensor": "Pair containing sensor",
            "sensor": "Pair containing sensor",
            "mix": "Pair containing sensor",
        }
        df["category"] = df["category"].astype(str).str.strip().map(lambda x: mapping.get(x.lower(), x))
        return df

    # Build from Params
    fr_list, sr_list = [], []
    for p in df["Params"].astype(str):
        fr, sr = parse_params_pair(p)
        fr_list.append(fr); sr_list.append(sr)
    df["fR_val"] = fr_list
    df["sR_val"] = sr_list
    df["category"] = [classify_category(fr, sr) for fr, sr in zip(fr_list, sr_list)]
    return df


def save_category_boxplots(df: pd.DataFrame, metrics: list, fname_prefix: str):
    """Make and save a single row of boxplots (one per metric) over category."""
    cat_order = ["Homogeneous", "Heterogeneous", "Pair containing sensor"]
    palette = {"Homogeneous": "blue", "Heterogeneous": "orange", "Pair containing sensor": "green"}

    # Filter rows that have category in our order (ignore unexpected categories)
    df = df[df["category"].isin(cat_order)].copy()
    if df.empty:
        print("No data after filtering categories for plotting.")
        return []

    # Only keep rows where the metric is present and numeric
    existing = [m for m in metrics if m in df.columns]
    if not existing:
        print("None of the requested metrics exist in the dataframe:", metrics)
        return []

    n = len(existing)
    # Figure width scales with number of metrics
    plt.figure(figsize=(5*n, 4))

    for i, m in enumerate(existing, start=1):
        plt.subplot(1, n, i)
        # Remove inf/nan
        sub = df[[m, "category"]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            plt.text(0.5, 0.5, f"No data for {m}", ha="center"); plt.axis("off"); continue
        sns.boxplot(x="category", y=m, data=sub, order=cat_order, palette=palette, showfliers=False)
        if m.lower().startswith("time") or "t50" in m.lower():
            plt.ylabel("t50 (Midpoint)")
            plt.title("Sparse sweep: t50 by category")
        else:
            plt.ylabel(m.replace("_", " "))
            pretty = " ".join(m.split("_"))
            plt.title(f"Sparse sweep: {pretty} by category")
        plt.xlabel("Category"); plt.xticks(rotation=20)

    plt.tight_layout()
    out = os.path.join(SAVE_DIR, f"{fname_prefix}_category_boxplots.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)
    return [out]


def save_byDiffusion_boxplots(df: pd.DataFrame, metrics: list, fname_prefix: str):
    """For each metric, make one figure: x=category, y=metric, faceted by Diffusion (one subplot per D)."""
    df = df.copy()
    if "Diffusion" not in df.columns:
        print("No 'Diffusion' column; skipping by-Diffusion plots.")
        return []

    cat_order = ["Homogeneous", "Heterogeneous", "Pair containing sensor"]
    palette = {"Homogeneous": "blue", "Heterogeneous": "orange", "Pair containing sensor": "green"}

    # Ensure numeric D order
    df["Diffusion"] = pd.to_numeric(df["Diffusion"], errors="coerce")
    D_vals = sorted(df["Diffusion"].dropna().unique().tolist())
    if not D_vals:
        print("No valid Diffusion values; skipping by-Diffusion plots.")
        return []

    saved = []
    for m in metrics:
        if m not in df.columns:
            continue
        # Layout
        n = len(D_vals)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(5*cols, 4*rows))
        for i, D in enumerate(D_vals, start=1):
            plt.subplot(rows, cols, i)
            sub = df[df["Diffusion"] == D]
            sub = sub[sub["category"].isin(cat_order)][[m, "category"]].replace([np.inf, -np.inf], np.nan).dropna()
            if sub.empty:
                plt.text(0.5, 0.5, f"No data\nD={D}", ha="center"); plt.axis("off"); continue
            sns.boxplot(x="category", y=m, data=sub, order=cat_order, palette=palette, showfliers=False)
            plt.title(f"D={D}")
            plt.xlabel("Category")
            ylabel = "t50 (Midpoint)" if (m.lower().startswith("time") or "t50" in m.lower()) else m.replace("_"," ")
            plt.ylabel(ylabel)
            plt.xticks(rotation=20)

        plt.suptitle(f"{m.replace('_',' ')} — by Category for each Diffusion", y=1.02, fontsize=12)
        plt.tight_layout()
        out = os.path.join(SAVE_DIR, f"{fname_prefix}_{m}_byDiffusion.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", out)
        saved.append(out)
    return saved


if not os.path.exists(SPARE_RUN_FILE):
    raise FileNotFoundError(f"Input file not found: {SPARE_RUN_FILE}")

sp = pd.read_csv(SPARE_RUN_FILE)


sp = ensure_category_column(sp)


metrics_all = []
if "BFS_Composite" in sp.columns:
    metrics_all.append("BFS_Composite")
if "Midpoint_Composite" in sp.columns:
    metrics_all.append("Midpoint_Composite")

t50_candidates = [c for c in sp.columns if c.lower() in ("time50_midpoint", "t50_midpoint", "time50")]
if t50_candidates:
    metrics_all.append(t50_candidates[0])


save_category_boxplots(sp, metrics_all, fname_prefix="fig1_stage1")

save_byDiffusion_boxplots(sp, metrics_all, fname_prefix="fig2_stage1")

print(f"Figures saved to {SAVE_DIR}")
