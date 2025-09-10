#!/usr/bin/env python
import sys
import os
import re
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque, defaultdict
import pyarrow.feather as feather
import math
# ======= Configuration =======

# ======= Dynamic score-grid builder =======
def build_score_grid(grid_size):
    # Walls setup
    w = np.zeros((grid_size, grid_size))
    w[:, -1] = -1
    w[1, :]  = -1
    w[-1, :] = -1
    w[:, 1]  = -1
    # internal walls
    w[0:int(0.8*grid_size), int(0.2*grid_size):int(0.25*grid_size)] = -1
    w[int(0.2*grid_size):grid_size, int(0.4*grid_size):int(0.45*grid_size)] = -1
    w[0:int(0.8*grid_size), int(0.6*grid_size):int(0.65*grid_size)] = -1
    w[int(0.2*grid_size):grid_size, int(0.8*grid_size):int(0.85*grid_size)] = -1
    return compute_continuous_score(w)


def compute_continuous_score(grid):
    rows, cols = grid.shape
    distance = np.full((rows, cols), -1, dtype=int)
    #target = (int(0.01 * rows), int(0.01 * cols))  # 1% from top-left corner
    target = (2,2)
    if grid[target] == -1:
        raise ValueError("Target is blocked!")
        exit(1)
    distance[target] = 0
    queue = deque([target])
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    while queue:
        r,c = queue.popleft()
        for dr,dc in dirs:
            nr,nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr,nc] != -1 and distance[nr,nc] == -1:
                    distance[nr,nc] = distance[r,c] + 1
                    queue.append((nr,nc))

    # find max reachable
    dvals = distance[distance>=0]
    max_dist = dvals.max() if dvals.size else 0

    # map linearly: 0→4, max_dist→0, unreachable→0
    # np.interp will give float64
    score = np.interp(distance,
                      xp=[0,        max_dist],
                      fp=[4.0,      0.0],
                      left=0.0,    # distance < 0 → 0
                      right=0.0)   # distance > max_dist → 0 (shouldn’t happen)
    return score


# ======= Midpoint predicate =======
def passes_midpoint(x, y, grid_size):
    return (y < 0.4 * grid_size) or (y < 0.6 * grid_size and x < 0.5 * grid_size)
    #return (y < 200) or (y < 300 and x < 250)    
    #return (y < 100) or (y < 150 and x < 125)

# ======= Scoring functions =======
def get_cell_score(x, y):
    xi = int(round(x))
    yi = int(round(y))
    xi = np.clip(xi, 0, GRID_SIZE - 1)
    yi = np.clip(yi, 0, GRID_SIZE - 1)
    return float(SCORE_GRID[yi, xi])

# ======= File processors =======
def process_bfs_file(path):
    df = feather.read_feather(path, columns=['x','y','consumption_rate'])
    if df.empty: return None
    xi = np.rint(df['x']).astype(np.int64).clip(0, GRID_SIZE-1)
    yi = np.rint(df['y']).astype(np.int64).clip(0, GRID_SIZE-1)
    score = SCORE_GRID[yi.to_numpy(), xi.to_numpy()]
    out = {'pop': float(score.mean()), 'pop_std': float(score.std(ddof=1))}
    means = pd.Series(score).groupby(df['consumption_rate']).mean()
    stds  = pd.Series(score).groupby(df['consumption_rate']).std(ddof=1)
    for k,v in means.items(): out[f'c-{k}'] = float(v)
    for k,v in stds.items():  out[f'c-{k}_std'] = float(v)
    return out

def process_midpoint_file(path, grid_size):
    df = feather.read_feather(path, columns=['x','y','consumption_rate'])
    if df.empty: return None
    passed = ((df['y'] < 0.4*grid_size) |
              ((df['y'] < 0.6*grid_size) & (df['x'] < 0.5*grid_size))
             ).astype(float)
    out = {'pop': float(passed.mean())}
    for k,v in passed.groupby(df['consumption_rate']).mean().items():
        out[f'c-{k}'] = float(v)
    return out



def _clean_number(x):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None                 # becomes JSON null
    return xf if math.isfinite(xf) else None

def process_position_file(path):
    df = feather.read_feather(path, columns=['id', 'x', 'y', 'consumption_rate'])
    if df.empty:
        return None

    out = {}
    for row in df.itertuples(index=False):
        cell_id = int(row.id)
        out[f"id_{cell_id}"] = {
            "x_pos": _clean_number(row.x),
            "y_pos": _clean_number(row.y),
            "cell_type": _clean_number(row.consumption_rate),
        }
    return out

def _dump_strict_json(obj, out_path):
    d = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fo:
            json.dump(obj, fo,
                      allow_nan=False,          # refuse NaN/Inf
                      separators=(",", ":"),    # compact
                      ensure_ascii=False)
            fo.flush(); os.fsync(fo.fileno())
        os.replace(tmp, out_path)
    except Exception:
        try: os.remove(tmp)
        finally: raise



# ======= Directory traversal =======
def analyze(root_dir, metric):
    results = {}
    root = Path(root_dir)
    all_params = list(root.glob('fR*__sR*'))
    print("[INFO] found {} parameter combos under {}".format(len(all_params), root_dir))
    for cr_dir in all_params:
        cr_key = cr_dir.name
        for diff_dir in cr_dir.glob('DiffusionRate_*'):
            diff_key = diff_dir.name
            for epoch_dir in sorted(diff_dir.glob('epoch_*'),
                                    key=lambda d: int(d.name.split('_')[-1])):
                cells = epoch_dir / 'Cells'
                # BEFORE
                files = sorted(cells.glob('cells_*.feather'), key=lambda p: int(re.search(r'cells_(\d+)', p.name).group(1)))

                print("[INFO] {}: {}/{}/{}: {} files".format(
                    metric, cr_key, diff_key, epoch_dir.name, len(files)))
                epoch_metrics = {}
                for fpath in files:
                    if metric == 'bfs':
                        m = process_bfs_file(fpath)
                    elif metric == 'midpoint':
                        m = process_midpoint_file(fpath, GRID_SIZE)
                    elif metric == 'positions':
                        m = process_position_file(fpath)
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    if m is not None:
                        epoch_metrics[fpath.name] = m
                results.setdefault(cr_key, {}) \
                       .setdefault(diff_key, {})[epoch_dir.name] = epoch_metrics
                
    return results

# ======= Main =======
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python metrics_processor_fin.py <root_directory> <grid_size> <n_cells> <label>")
        sys.exit(1)

    root = sys.argv[1]
    GRID_SIZE = int(sys.argv[2])
    N_CELLS = int(sys.argv[3])
    label = str(sys.argv[4])
    SCORE_GRID = build_score_grid(GRID_SIZE)
    tag = f"S{GRID_SIZE}C{N_CELLS}"

    print("Computing BFS scores...")
    bfs = analyze(root, 'bfs')
    _dump_strict_json(bfs, f'BFS_Scores_{tag}_{label}.json')

    print("Computing midpoint fractions...")
    mid = analyze(root, 'midpoint')
    _dump_strict_json(mid, f'Midpoint_Metrics_{tag}_{label}.json')

    print("Computing cell positions...")
    positions = analyze(root, 'positions')
    _dump_strict_json(positions, f'Cell_Positions_{tag}_{label}.json')

    print(f"[DONE] Metrics saved to BFS_Scores_{tag}_{label}.json, Midpoint_Metrics_{tag}_{label}.json and Cell_Positions_{tag}_{label}.json")
