"""Example usage:
create_simulation_animation(
    data_path="SimulationData_Sample/Maze/fR:0.001__sR:0.0005/DiffusionRate_0.25/epoch_0",
    output_file="interest.mp4",
    fps=15,
    dpi=100  # Resolution
)
"""

import os, re, queue, threading
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import pyarrow.feather as feather

# --------------------------
# Filename helpers
# --------------------------
def _basename_noext(path):
    return os.path.splitext(os.path.basename(path))[0]

def extract_time_step_from_grid(path):
    """Extract integer timestep from filenames like attractant4000 (with or without extension)."""
    base = _basename_noext(path)
    m = re.search(r'attractant(\d+)$', base)
    return int(m.group(1)) if m else None

def extract_time_step_from_cells(path):
    """Extract integer timestep from filenames like cells4000 (with or without extension)."""
    base = _basename_noext(path)
    m = re.search(r'cells(\d+)$', base)
    return int(m.group(1)) if m else None

# --------------------------
# Fast readers
# --------------------------
def read_grid_file(file_path):
    """Read feather file with a 'nums' column -> 1D np.array (memory-mapped)."""
    if file_path is None:
        return None
    try:
        tbl = feather.read_table(file_path, memory_map=True)
        return tbl["nums"].to_numpy()
    except Exception as e:
        print(f"[grid] Error {file_path}: {e}")
        return None

def read_cell_file(file_path):
    """Read feather file to pandas (expects x/xloc & y/yloc; optional consumption_rate/cell_type as 'rate')."""
    if file_path is None:
        return pd.DataFrame(columns=["x", "y", "rate"])
    try:
        df = feather.read_table(file_path, memory_map=True).to_pandas()
        lower = {c.lower(): c for c in df.columns}
        xcol = lower.get("xloc", lower.get("x"))
        ycol = lower.get("yloc", lower.get("y"))
        rcol = lower.get("consumption_rate", lower.get("cell_type"))
        if xcol is None or ycol is None:
            raise ValueError("Expected x/xloc and y/yloc columns.")
        use_cols = [xcol, ycol] + ([rcol] if rcol else [])
        out = df[use_cols].rename(columns={xcol: "x", ycol: "y", (rcol or "rate"): "rate"})
        return out
    except Exception as e:
        print(f"[cells] Error {file_path}: {e}")
        return pd.DataFrame(columns=["x", "y", "rate"])

# --------------------------
# Epoch discovery
# --------------------------
def _has_epoch_layout(path):
    return os.path.isdir(os.path.join(path, "Grid")) and os.path.isdir(os.path.join(path, "Cells"))

def _list_epoch_dirs(root):
    """Return a sorted list of epoch_* dirs, or [root] if root itself has Grid/Cells."""
    if _has_epoch_layout(root):
        return [root]
    candidates = [p for p in glob(os.path.join(root, "epoch_*")) if os.path.isdir(p)]
    def _epoch_num(p):
        m = re.search(r'epoch_(\d+)$', p)
        return int(m.group(1)) if m else -1
    candidates.sort(key=_epoch_num)
    return candidates

# --------------------------
# Build a full timeline (no early cut-off)
# --------------------------
def _build_timeline(epoch_dirs, frame_skip):
    """
    Returns:
      steps:    list of (epoch_idx, step_int)
      g_paths:  list of grid file paths or None (same length as steps)
      c_paths:  list of cell file paths or None
      labels:   list of strings for overlay
      grid_shape_hint: (Nx, Ny) from the first available grid
      first_cells_df: first available cells df (for rate grouping)
    """
    steps = []
    g_paths = []
    c_paths = []
    labels = []

    grid_shape_hint = None
    first_cells_df = None

    for e_idx, e_dir in enumerate(epoch_dirs):
        g_files = glob(os.path.join(e_dir, "Grid", "attractant*"))
        c_files = glob(os.path.join(e_dir, "Cells", "cells*"))

        grid_map = {}
        for p in g_files:
            t = extract_time_step_from_grid(p)
            if t is not None:
                grid_map[t] = p
                if grid_shape_hint is None:
                    arr = read_grid_file(p)
                    if arr is not None:
                        n = int(np.sqrt(len(arr)))  # expect square
                        grid_shape_hint = (n, n)

        cell_map = {}
        for p in c_files:
            t = extract_time_step_from_cells(p)
            if t is not None:
                cell_map[t] = p
                if first_cells_df is None:
                    df0 = read_cell_file(p)
                    if not df0.empty:
                        first_cells_df = df0

        # Union of steps ensures start->end coverage, even if one side is missing
        all_steps = sorted(set(grid_map) | set(cell_map))
        if frame_skip > 1:
            all_steps = all_steps[::frame_skip]

        for t in all_steps:
            steps.append((e_idx, t))
            g_paths.append(grid_map.get(t))  # may be None
            c_paths.append(cell_map.get(t))  # may be None
            labels.append(f"Epoch {e_idx} • Step {t}")

        print(f"[align] epoch_{e_idx}: frames={len(all_steps)} "
              f"(min={all_steps[0] if all_steps else '—'}, max={all_steps[-1] if all_steps else '—'}) "
              f"grid-only={len(set(grid_map)-set(cell_map))} cells-only={len(set(cell_map)-set(grid_map))}")

    if grid_shape_hint is None:
        raise RuntimeError("Could not determine grid size (no readable grid files).")
    if first_cells_df is None:
        raise RuntimeError("No readable cell files found.")

    return steps, g_paths, c_paths, labels, grid_shape_hint, first_cells_df

# --------------------------
# Optional prefetch (I/O)
# --------------------------
class Prefetcher:
    """Background prefetch of (grid_array_or_None, cells_df_or_empty) tuples for frame indices."""
    def __init__(self, grid_files, cell_files, NxNy, capacity=4):
        self.grid_files = grid_files
        self.cell_files = cell_files
        self.NxNy = NxNy
        self.N = len(grid_files)
        self.q = queue.Queue(maxsize=capacity)
        self.stop = False
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def _worker(self):
        Nx, Ny = self.NxNy
        for i in range(self.N):
            if self.stop:
                break
            g_path = self.grid_files[i]
            c_path = self.cell_files[i]
            g = read_grid_file(g_path)
            if g is not None:
                try:
                    g = g.reshape((Nx, Ny)).T
                except Exception:
                    g = None
            c = read_cell_file(c_path)
            self.q.put((i, g, c), block=True)
        self.q.put((None, None, None))  # sentinel

    def get(self):
        return self.q.get()

    def close(self):
        self.stop = True
        try:
            while not self.q.empty():
                self.q.get_nowait()
        except queue.Empty:
            pass

# --------------------------
# Main animation
# --------------------------
def create_simulation_animation(
    data_path,
    output_file="simulation_animation.mp4",
    fps=15,
    dpi=100,
    vmax=1.0,
    cell_palette=("red", "green"),  # high-contrast over plasma/magma
    cmap_name="plasma",
    distinct_markers=True,
    frame_skip=1,       # e.g., 2 to use every 2nd frame
    use_prefetch=True,  # threaded I/O prefetch
    codec="libx264",
    crf=20,
    preset="veryfast",
):
    """
    Create an animation that spans from the very first to the very last available step.
    Missing frames on either side (Grid/Cells) are forward-filled so there's no early cut-off.
    """
    # Discover epoch directories (supports path/to/epoch_0 or path/to/.../ with epoch_* subdirs)
    epoch_dirs = _list_epoch_dirs(data_path)
    if not epoch_dirs:
        print("No epoch directories or Grid/Cells found at:", data_path)
        return

    # Build a single timeline across all epochs (in order)
    steps, grid_files, cell_files, labels, (Nx, Ny), first_cells = _build_timeline(epoch_dirs, frame_skip)
    N = len(steps)
    if N == 0:
        print("No frames to render.")
        return
    print(f"[timeline] total frames: {N}  (epochs: {len(epoch_dirs)})")

    # First available grid (for initial image)
    first_grid_1d = None
    for gp in grid_files:
        if gp is not None:
            first_grid_1d = read_grid_file(gp)
            if first_grid_1d is not None:
                break
    if first_grid_1d is None:
        print("Failed to read any grid.")
        return
    first_grid = first_grid_1d.reshape((Nx, Ny)).T

    # Rate grouping based on first available cells
    has_rate = ("rate" in first_cells.columns) and first_cells["rate"].notna().any()
    rates = (sorted(first_cells["rate"].dropna().unique().tolist()) if has_rate else ["all"])

    # --- Figure & artists (blit-ready)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, Nx); ax.set_ylim(0, Ny); ax.set_aspect("equal")

    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    cmap = get_cmap(cmap_name)
    im = ax.imshow(first_grid, origin="lower", extent=[0, Nx, 0, Ny],
                   cmap=cmap, norm=norm, animated=True, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attractant Concentration")

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "*", "h", "<", ">"]
    color_cycle = list(cell_palette)

    # Pick a "first cells" frame to seed scatters
    seed_cells = None
    for cp in cell_files:
        df = read_cell_file(cp)
        if not df.empty:
            seed_cells = df
            break
    if seed_cells is None:
        seed_cells = first_cells

    group_artists = {}
    legend_handles = []

    if has_rate:
        for idx, r in enumerate(rates):
            sub = seed_cells[seed_cells["rate"] == r]
            color = color_cycle[idx % len(color_cycle)]
            marker = marker_cycle[idx % len(marker_cycle)] if distinct_markers else "o"
            label = f"rate={r}" if isinstance(r, str) else f"rate={r:g}"
            scat = ax.scatter(
                sub["x"], sub["y"],
                s=20, c=color, marker=marker,
                alpha=1.0, edgecolors="black", linewidths=0.3,
                zorder=3, animated=True, label=label
            )
            group_artists[r] = scat
            legend_handles.append(scat)
    else:
        scat = ax.scatter(
            seed_cells["x"], seed_cells["y"],
            s=20, c=color_cycle[0],
            alpha=1.0, edgecolors="black", linewidths=0.3,
            zorder=3, animated=True, label="cells"
        )
        group_artists["all"] = scat
        legend_handles.append(scat)

    if legend_handles:
        ax.legend(loc="upper right", framealpha=0.8)

    # Overlay text: epoch & step
    time_text = ax.text(
        0.02, 0.98, labels[0],
        transform=ax.transAxes, va="top", ha="left",
        color="w", fontsize=12,
        bbox=dict(facecolor="0.1", alpha=0.35, edgecolor="none"),
        animated=True
    )

    artists_static = [im] + list(group_artists.values()) + [time_text]

    # Prefetch
    prefetch = Prefetcher(grid_files, cell_files, (Nx, Ny), capacity=6) if use_prefetch else None
    frame_cache = {}
    prev_grid = first_grid
    prev_cells = seed_cells

    def get_frame(i):
        if prefetch:
            while True:
                j, g, c = prefetch.get()
                if j is None:
                    return None, None
                frame_cache[j] = (g, c)
                if i in frame_cache:
                    g2, c2 = frame_cache.pop(i)
                    return g2, c2
        else:
            g_path = grid_files[i]
            c_path = cell_files[i]
            g = read_grid_file(g_path)
            g = g.reshape((Nx, Ny)).T if g is not None else None
            c = read_cell_file(c_path)
            return g, c

    def init():
        im.set_array(first_grid)
        if has_rate:
            for r, scat in group_artists.items():
                sub = seed_cells[seed_cells["rate"] == r]
                scat.set_offsets(np.c_[sub["x"].to_numpy(), sub["y"].to_numpy()])
        else:
            group_artists["all"].set_offsets(np.c_[seed_cells["x"].to_numpy(), seed_cells["y"].to_numpy()])
        time_text.set_text(labels[0])
        return artists_static

    def update(i):
        nonlocal prev_grid, prev_cells
        g, c = get_frame(i)

        # Forward-fill missing sides to guarantee continuous progression
        if g is not None:
            im.set_array(g)
            prev_grid = g
        else:
            im.set_array(prev_grid)

        if c is not None and len(c):
            prev_cells = c
        c_use = prev_cells

        if has_rate:
            for r, scat in group_artists.items():
                sub = c_use[c_use["rate"] == r]
                if len(sub):
                    scat.set_offsets(np.c_[sub["x"].to_numpy(), sub["y"].to_numpy()])
                else:
                    scat.set_offsets(np.empty((0, 2)))
        else:
            group_artists["all"].set_offsets(np.c_[c_use["x"].to_numpy(), c_use["y"].to_numpy()])

        time_text.set_text(labels[i])
        return artists_static

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=N,
        interval=1000 / fps,
        blit=True,
        cache_frame_data=False,
    )

    writer = FFMpegWriter(
        fps=fps,
        codec=codec,
        bitrate=None,
        extra_args=["-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", preset],
    )
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    anim.save(output_file, writer=writer, dpi=dpi)
    plt.close(fig)
    if prefetch:
        prefetch.close()
    print(f"Animation saved to {output_file}")
