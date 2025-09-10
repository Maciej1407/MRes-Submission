#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dmv, mkm
"""

import numba
import numpy as np
import math

"""
# ---------------------------------------------------------------------------
# Numerical integrator: use Forward-Euler when D is small enough, otherwise
# DuFort–Frankel.
EULER_THRESHOLD = 0.25
# ---------------------------------------------------------------------------


Not used: MIGRASOME_RADIUS = 1  # in grid‐cell units
"""

#This is used for one step after initialization, as dufort-frankel requires two steps of initial conditions

#!/usr/bin/env python3

import numpy as np
import math

V_MAX = 0.0004


#MIGRASOME_RADIUS = 1
EULER_THRESHOLD = 0.25  # Threshold for choosing between Euler and DuFort–Frankel
# ---------------------------------------------------------------------------
# Vectorised explicit Forward-Euler integrator
# ---------------------------------------------------------------------------
@numba.jit(nopython=True, parallel=True)
def step_diffusion_forwardeuler(grid, grid_walls, D):
    grid_new = np.copy(grid)
    if D > 0.25:
        D = 0.25
    rows, cols = grid.shape
    for i in numba.prange(1, rows - 1):
        for j in range(1, cols - 1):
            if grid_walls[i, j]:
                continue
            # Calculate contributions using arithmetic instead of conditionals
            up = grid[i, j+1] * (1 - grid_walls[i, j+1]) + grid[i, j] * grid_walls[i, j+1]
            down = grid[i, j-1] * (1 - grid_walls[i, j-1]) + grid[i, j] * grid_walls[i, j-1]
            left = grid[i+1, j] * (1 - grid_walls[i+1, j]) + grid[i, j] * grid_walls[i+1, j]
            right = grid[i-1, j] * (1 - grid_walls[i-1, j]) + grid[i, j] * grid_walls[i-1, j]
            laplacian = (left + right + up + down - 4 * grid[i, j])
            grid_new[i, j] = grid[i, j] + D * laplacian
    return grid_new

@numba.jit(nopython=True, parallel=True)
def step_diffusion_dufortfrankel(grid, old_grid, grid_walls, D):
    grid_new = np.zeros_like(grid)
    rows, cols = grid.shape
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if grid_walls[i, j]:
                grid_new[i, j] = grid[i, j]  # Preserve wall values
                continue
                
            # Get neighbors with boundary checks
            up = grid[i, j+1] if j+1 < cols and not grid_walls[i, j+1] else grid[i, j]
            down = grid[i, j-1] if j-1 >= 0 and not grid_walls[i, j-1] else grid[i, j]
            left = grid[i+1, j] if i+1 < rows and not grid_walls[i+1, j] else grid[i, j]
            right = grid[i-1, j] if i-1 >= 0 and not grid_walls[i-1, j] else grid[i, j]
            
            # Correct DuFort-Frankel formula
            numerator = (1 - 4*D) * old_grid[i, j] + 2*D * (up + down + left + right)
            grid_new[i, j] = numerator / (1 + 4*D)
    
    return grid_new, grid


@numba.jit(nopython=True, parallel=True)
def step_decay(grid, halflife):
    decay_factor = math.exp(-1 / halflife)
    rows, cols = grid.shape
    for i in numba.prange(1, rows - 1):
        for j in range(1, cols - 1):
            grid[i, j] *= decay_factor
    return grid

#Same effect as decay function above, but returns the amount broken down
#Can be useful if you want a ligand to break down into a different ligand
@numba.jit(nopython=True)
def step_decay_withreturn(grid,halflife):
    Nx = grid.shape[0]-1
    Ny = grid.shape[1]-1
    decaysites = np.zeros(Nx*Ny,dtype=np.int64)
    decayvalues = np.zeros(Nx*Ny)
    sitecount = 0
    for i in range(1, Nx):
        for j in range(1, Ny):
            pastvalue = grid[i,j]
            if pastvalue>0:
                grid[i,j] = grid[i,j]* math.exp(-1/halflife)
                decaysites[sitecount] = j*Nx+i
                decayvalues[sitecount] = pastvalue - grid[i,j]
    return(grid,decaysites[0:sitecount],decayvalues[0:sitecount])

#Remove some ligand with a df in a specific area
#Used to represent consumption or extracellular breakdown around cells

@numba.jit(nopython=True)
def step_consume(grid,coveredsquares,consumptiondf):
    totalconsumed = 0
    Nx = grid.shape[0]
    for location in coveredsquares:
        i = location%Nx
        j = math.floor(location / Nx)
        localconcentration = grid[i,j]
        if localconcentration>0:
            grid[i,j]= V_MAX*(localconcentration/(localconcentration + consumptiondf))  
            totalconsumed += (localconcentration-grid[i,j])
            if grid[i,j]<0:
                print("Oopsie! negative value")
                grid[i,j]=0
    return grid, totalconsumed




@numba.jit(nopython=True)
def step_consume(grid, coveredsquares, K):
    rows, cols = grid.shape
    total_removed = 0.0
    for loc in coveredsquares:
        i = loc // cols
        j = loc %  cols
        local = grid[i, j]
        if local > 0.0:
            removed = V_MAX * (local / (local + K))
            if removed > local:
                removed = local
            grid[i, j] = local - removed
            total_removed += removed
    return grid, total_removed

#As above, but returns how much has been consumed/broken down
#Can be used to have a ligand break down into another ligand without moving
@numba.jit(nopython=True)
def step_consume_returnsites(grid, coveredsquares, K):
    Nx = grid.shape[0]
    nsites = len(coveredsquares)
    consumevalues = np.zeros(nsites)
    for idx in range(nsites):
        loc = coveredsquares[idx]
        i   = loc % Nx      # unchanged
        j   = loc // Nx     # unchanged
        local = grid[i, j]
        if local > 0.0:
            # compute how much to remove
            removed = V_MAX * (local / (local + K))
            if removed > local:      # guard against tiny negatives
                removed = local
            
            grid[i, j] = local - removed
            consumevalues[idx] = removed
        else:
            consumevalues[idx] = 0.0
    return grid, consumevalues 



#Produce a ligand in a specific area, usually under a cell
@numba.jit(nopython=True)
def step_produce(grid,coveredsquares,production):
    Nx = grid.shape[0]
    sitecount = 0
    for site in coveredsquares:
        i = site%Nx
        j = math.floor(site / Nx)
        addedconcentration = (production[sitecount])
        grid[i,j] += addedconcentration
        sitecount +=1
    return grid

#Sense the amount of ligand locally
@numba.jit(nopython=True)
def sense_jitted(grid,coveredsquares):
    totalsensed= 0
    Nx = grid.shape[0]
    for location in coveredsquares:
        i = location%Nx
        j = math.floor(location / Nx)
        totalsensed += grid[i,j]
    return totalsensed

class ligand:
    def __init__(self,xsize,ysize,halflife,diffuserate,grid_walls,name, scheme = "auto"):
        self.xsize = xsize
        self.ysize = ysize
        self.halflife = halflife
        self.diffuserate = diffuserate
        self.grid_prev = np.zeros((xsize, ysize))  # initial condition (e.g., zero everywhere), may be set to something 
        self.grid = np.zeros((xsize, ysize)) #Will be set by the first euler step
        self.name = name
        self.scheme = scheme.lower()
        self.timestep = 0
        self.secreted_cumulative = 0.0
        self.max_value = 1e10
        
        if scheme == "auto":
            if diffuserate <= EULER_THRESHOLD:
                self.scheme = "euler"
            else:
                self.scheme = "dufort"
        else:
            self.scheme = scheme
            
        self.timestep = 0
        self.secreted_cumulative = 0.0    #These methods are mostly wrappers for functions that have been sped up with 

        schemeMap = {
            "euler": self.diffuse_euler,
            "dufort": self.diffuse_dufort
        }

    # Diffusion auto selector
    def diffuse(self, grid_walls):
        schemeMap = {
            "euler": self.diffuse_euler,
            "dufort": self._dufort_safe
        }
        schemeMap[self.scheme](grid_walls)

    #Done as a first step of diffusion after initialisation

    def diffuse_euler(self,grid_walls, D=None):
        if D is None:
            D = self.diffuserate          # default behaviour
        new_grid = step_diffusion_forwardeuler(self.grid, grid_walls, D)
        self.grid_prev = self.grid
        self.grid = new_grid

    #Used for diffusion in later steps
    def diffuse_dufort(self, grid_walls, D=None):
        if D is None:
            D = self.diffuserate          # default behaviour
        self.grid, self.grid_prev = step_diffusion_dufortfrankel(self.grid, self.grid_prev, grid_walls, D = self.diffuserate)
    

    def _dufort_safe(self, grid_walls):
        """Split large diffusion rates into stable chunks."""
        if self.diffuserate > 1.0:
            n, f = divmod(self.diffuserate, 1.0)   # integer + fraction
            for _ in range(int(n)):
                self.diffuse_dufort(grid_walls, D=1.0)
            if f > 1e-12:
                self.diffuse_dufort(grid_walls, D=f)
        else:
            self.diffuse_dufort(grid_walls)         # regular step


    #Decay the ligand using its halflife. Either return what has been decayed or not (slight overhead for returning)
    def decay(self,returndetail = False):
        if returndetail:
            self.grid, decaysites,decayvalues = step_decay_withreturn(self.grid,self.halflife)
            return decaysites,decayvalues
        else:
            self.grid = step_decay(self.grid,self.halflife)
            
    #Decay the ligand using a different grid, being careful not to go below 0
    #Could be sped up with numba if used intensively in the future
    def decaywithgrid(self,decaygrid,kd):
        self.grid = self.grid - kd*(self.grid * decaygrid)
        for i in range(1, self.grid.shape[0]-1):
            for j in range(1, self.grid.shape[1]-1):
                if self.grid[i,j]<0:
                    self.grid[i,j]=0
                #print(self.grid[i,j])
            
    #Remove the ligand with a specific df from the squares, either returning the values removed or not
    #Slight overhead for reporting back
    def consume(self,coveredsquares, consumptiondf, returndetail=False):
        if returndetail:
                self.grid, consumedvalues = step_consume_returnsites(
                    self.grid, coveredsquares, consumptiondf)
                
                return consumedvalues
        else:
            self.grid, totalconsumed = step_consume(
                self.grid, coveredsquares, consumptiondf
            )
            return totalconsumed
    
    #Sense the amount of this ligand in this area
    def sense(self,coveredsquares):
        return sense_jitted(self.grid,coveredsquares)
        
    #Put a single amount of ligand into the system at the specified locations
    def produce_singlevalue(self,coveredsquares,production):
        productionvalues = np.full(len(coveredsquares),production/len(coveredsquares))
        self.grid = step_produce(self.grid,coveredsquares,productionvalues)
        
    #Put the specified amount of ligand into the system for specific locations
    def produce_multivalues(self,coveredsquares,productionvalues):
        self.grid = step_produce(self.grid,coveredsquares,productionvalues)


def receive_migrasomes(env, locations):
    """
    Add migrasomes to the environment. Stores them as (x, y, time_remaining).
    """
    if not hasattr(env, "active_migrasomes"):
        env.active_migrasomes = []
    for (x, y) in locations:
        env.active_migrasomes.append((x, y, 2000))  # 300 = signal duration


def update_migrasomes_leg(env):
    """
    Each timestep, emit chemoattractant from active migrasomes and remove expired ones.
    """
    if not hasattr(env, "active_migrasomes"):
        env.active_migrasomes = []

    new_migrasomes = []
    for (x, y, time_left) in env.active_migrasomes:
        if time_left > 0:
            i, j = int(x), int(y)
            # secrete over a small radius
            for dx in range(-MIGRASOME_RADIUS, MIGRASOME_RADIUS + 1):
                for dy in range(-MIGRASOME_RADIUS, MIGRASOME_RADIUS + 1):
                    ii, jj = i + dx, j + dy
                    if 0 <= ii < env.grid.shape[0] and 0 <= jj < env.grid.shape[1]:
                        env.grid[ii, jj] += 0.004
                        if env.grid[ii, jj] > 1.0:
                            env.grid[ii, jj] = 1.0
            new_migrasomes.append((x, y, time_left - 1))

    env.active_migrasomes = new_migrasomes

# migrasome representation and secretion logic, not used in thesis
# ─────────────────────────────────────────────────────────────────────────────
def update_migrasomes(env, current_time):
    """
    Advance all active migrasomes belonging to the given ligand *env*
    (which is a ligand instance), secrete attractant into env.grid,
    and keep track of the total released mass via
    env.secreted_cumulative.
    """
    if not hasattr(env, "active_migrasomes"):
        env.active_migrasomes = []

    new_migrasomes = []
    for mig in env.active_migrasomes:
        x, y, creation_time, duration = mig
        age = current_time - creation_time
        if age >= duration:
            continue  # migrasome expired

        # Gaussian secretion profile, peaks at 1/3 lifespan
        peak_time = duration / 3.0
        std_dev   = duration / 6.0
        base_secretion = 0.003
        gaussian_factor = math.exp(-0.5 * ((age - peak_time) / std_dev) ** 2)
        secretion_rate  = base_secretion * gaussian_factor

        i, j = int(x), int(y)
        for dx in range(-MIGRASOME_RADIUS, MIGRASOME_RADIUS + 1):
            for dy in range(-MIGRASOME_RADIUS, MIGRASOME_RADIUS + 1):
                ii, jj = i + dx, j + dy
                if 0 <= ii < env.grid.shape[0] and 0 <= jj < env.grid.shape[1]:
                    env.grid[ii, jj] += secretion_rate
                    env.secreted_cumulative += secretion_rate
                    if env.grid[ii, jj] > 1.0:
                        env.grid[ii, jj] = 1.0

        new_migrasomes.append((x, y, creation_time, duration))

    env.active_migrasomes = new_migrasomes
# ─────────────────────────────────────────────────────────────────────────────


# Update migrasome creation
def receive_migrasomes(env, locations, current_time, duration=1500):
    if not hasattr(env, "active_migrasomes"):
        env.active_migrasomes = []
    
    for (x, y) in locations:
        # Store creation time and duration with position
        env.active_migrasomes.append((x, y, current_time, duration))
