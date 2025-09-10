import os
import pyarrow as pa
import pyarrow.feather as feather
import numpy as np

class SimulationSaver:
    def __init__(self, root_directory, env_name):
        self.root_dir = root_directory
        self.env_name = env_name
        self.path = self.root_dir          # alias for backward compatibility

        self._create_directories()
        
        # ─── Attractant mass-balance logger ───────────────────────────────────────────
    def save_attractant_balance(self, records):
        import pandas as pd, os
        if not records:
            return
        pd.DataFrame.from_records(records).to_csv(
            os.path.join(self.path, "AttractantBalance.csv"),
            index=False
        )

    # ──────────────────────────────────────────────────────────────────────────────

    def _create_directories(self):
        """Create required directory structure"""
        os.makedirs(self.root_dir, exist_ok=True)
        self.migrasome_dir = os.path.join(self.root_dir, "Migrasomes")
        os.makedirs(self.migrasome_dir, exist_ok=True)
        
    def save_migrasomes(self, migrasomes, timestep):
        """Save migrasome data for a specific timestep"""
        if not migrasomes:
            return
            
        # Prepare data arrays
        xs, ys, creation_times, durations = [], [], [], []
        for mig in migrasomes:
            xs.append(mig[0])
            ys.append(mig[1])
            creation_times.append(mig[2])
            durations.append(mig[3])
        
        # Create Arrow table
        table = pa.Table.from_arrays(
            [xs, ys, creation_times, durations],
            names=["x", "y", "creation_time", "duration"]
        )
        
        # Save with timestep in filename
        filename = f"migrasomes_{timestep}.feather"
        feather.write_feather(table, os.path.join(self.migrasome_dir, filename))
        
    def save_attractant(self, grid, timestep):
        """Save attractant grid state"""
        dir_path = os.path.join(self.root_dir, "Attractant")
        os.makedirs(dir_path, exist_ok=True)
        flattened = grid.flatten()
        table = pa.Table.from_arrays([flattened], names=["concentration"])
        filename = f"attractant_{timestep}.feather"
        feather.write_feather(table, os.path.join(dir_path, filename))
        
    def save_cells(self, cells, timestep):
        """Save cell data"""
        dir_path = os.path.join(self.root_dir, "Cells")
        os.makedirs(dir_path, exist_ok=True)
        
        # Prepare cell data
        ids, xs, ys, consumptions = [], [], [], []
        for cell in cells:
            ids.append(cell.id)
            xs.append(cell.xlocation)
            ys.append(cell.ylocation)
            consumptions.append(cell.consumptionRate)
        
        # Create Arrow table
        table = pa.Table.from_arrays(
            [ids, xs, ys, consumptions],
            names=["id", "x", "y", "consumption_rate"]
        )
        
        filename = f"cells_{timestep}.feather"
        feather.write_feather(table, os.path.join(dir_path, filename))