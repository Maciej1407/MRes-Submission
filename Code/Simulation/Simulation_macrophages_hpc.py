#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
authors: dmv, mkm
"""

import os
import time
import random
import argparse
import numpy as np
import math

import mazelayouts
import environment
import cell
import collisionfunctions
import SimulationSaver
"""
Not used in thesis, fun test:

from environment import receive_migrasomes, update_migrasomes


MIGRASOME_DISTANCE_THRESHOLD    = 10     # was 150
MIGRASOME_VELOCITY_THRESHOLD    = 0.002   # was 0.04
MIGRASOME_PERSISTENCE_THRESHOLD = 0.8  # was 0.95
MIGRASOME_COOLDOWN = 2000           """


# Logging
LOG_FILE = "simulation_status_p.log"
def log_status(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

SIM_PARAM_LOG = "Simulation_parameters.log"
def log_params(message):
    with open(SIM_PARAM_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")



# Map name to func
HashMapEnvs = {
    "Maze": mazelayouts.makewalls,
    "Empty": mazelayouts.makeemptycontainer,
    "Dish": mazelayouts.makeemptydish,
    "Open": mazelayouts.makeopensite,
    "Bridge": mazelayouts.makebridge,
    "Vessel": mazelayouts.makebloodvessel
}


#METRICS_LOG_FILE = "migrasome_metrics_2.log"

def log_migrasome_metrics(t, cells):
    with open(METRICS_LOG_FILE, "a") as f:
        f.write(f"\n--- Time Step {t} ---\n")
        for cell in cells:
            # Compute metrics for each cell
            net_dx = cell.xlocation - cell.migrasome_start_pos[0]
            net_dy = cell.ylocation - cell.migrasome_start_pos[1]
            net_disp = math.hypot(net_dx, net_dy)
            elapsed = t - cell.migrasome_start_time if t > cell.migrasome_start_time else 1
            velocity = net_disp / elapsed

            if len(cell.migrasome_history) > 0:
                avg_dx = sum(d[0] for d in cell.migrasome_history) / len(cell.migrasome_history)
                avg_dy = sum(d[1] for d in cell.migrasome_history) / len(cell.migrasome_history)
                avg_mag = math.hypot(avg_dx, avg_dy)
                cur_dx, cur_dy = cell.migrasome_history[-1]
                cur_mag = math.hypot(cur_dx, cur_dy)

                cos_sim = (
                    (avg_dx * cur_dx + avg_dy * cur_dy) / (avg_mag * cur_mag)
                    if avg_mag > 0 and cur_mag > 0 else 0.0
                )
            else:
                cos_sim = 0.0

            eligible = (
                net_disp >= MIGRASOME_DISTANCE_THRESHOLD and
                velocity >= MIGRASOME_VELOCITY_THRESHOLD and
                cos_sim >= MIGRASOME_PERSISTENCE_THRESHOLD and
                cell.migrasome_cooldown <= 0
            )

            f.write(
                f"Cell {cell.id:03d} | disp={net_disp:7.2f} | v={velocity:6.4f} | "
                f"cos={cos_sim:5.2f} | cooldown={cell.migrasome_cooldown:4d} | eligible={eligible}\n"
            )






def sim_run(layout, save_path,
            steps, Nx, Ny, maxcells,
            Consumption_rate_pair, diffusion_rate,
            plotting, cellsize, mitogenfactor, basemitosis, basedeath,
            collision, celldistancefactor, persistence, movedistance,
            cellfuzzing, attractantweighing,
            metabolitehalflife, metaboliteproductionratio, metaboliteconsumption,
            dmethod = "auto",
            enable_migrasomes = False):
    

    log_status(f"Starting simulation: Environment={layout}, "
               f"Consumption={Consumption_rate_pair}, "
               f"Diffusion={diffusion_rate}, Path={save_path}")
    start_time = time.time()
    
    log_params(f"Nx={Nx}, Ny={Ny}, Cells Count ={maxcells}, path={save_path}, ")

    #celldistancefactor
    
    # Build folder

    os.makedirs(save_path, exist_ok=True)
    mydatasaver = SimulationSaver.SimulationSaver(save_path, layout)
    plotinterval = plotting

    # Environment
    if layout not in HashMapEnvs:
        raise ValueError(f"Unknown layout {layout}")

    grid_walls = HashMapEnvs[layout](Nx, Ny)
    grid_walls_bool = (grid_walls > 0).astype(bool)    


    split = int(maxcells//2)

    # Consumption rates
    if Consumption_rate_pair[1] == "single":
        consumptionRates = [Consumption_rate_pair[0] for _ in range(maxcells)]
    else: 
        consumptionRates = ([Consumption_rate_pair[0] for _ in range(split)] +
                            [Consumption_rate_pair[1] for _ in range(split, maxcells)] )
    
    unique_rates = sorted(set(consumptionRates))       # e.g. [0.0001, 1.0]
    rate_cumulative = {r: 0.0 for r in unique_rates}   # running totals per sub-pop
    
    cellcollision = collision
    celldistance = cellsize*celldistancefactor
    
    alloccupiedsites = np.array([])
    wallsites = np.array([])
    #print("Adding Walls")

    
    wall_indices = np.argwhere(grid_walls > 0)
    linear_indices = (wall_indices[:, 1] * Nx + wall_indices[:, 0]).astype(np.int64)
    alloccupiedsites = np.sort(linear_indices)
    wallsites = np.copy(alloccupiedsites)
    
    
    attractantdiffusion = diffusion_rate
    

    attractanthalflife = 5000000
    

    ligands = {}
    attractant_halflife = 5e10
    ligands["attractant"] = environment.ligand(
        Nx, Ny, attractant_halflife, diffusion_rate,
        grid_walls_bool, "attractant", scheme=dmethod
    )

    non_wall = (grid_walls < 1)

    for ligand in ligands.values():
        ligand.diffuse_euler(grid_walls_bool)

    A_init = ligands["attractant"].grid_prev.sum()     # mass at t = 0
    A_secr = 0.0                         # cumulative attractant secreted
    A_cons = 0.0                                       # cumulative consumed
    balance_records = []        

    interior = np.zeros_like(non_wall, dtype=bool)
    interior[1:-1, 1:-1] = True
    mask = non_wall & interior

    ligands["attractant"].grid_prev[mask] = 1
    ligands["attractant"].grid[mask] = 1  # CRITICAL MISSING LINE

    print("Environment filled")

    # Place cells (using your original logic)
    cells = list()

    cellcollision = 0
    
    if cellcollision ==1:
        cellxlocations = np.zeros(maxcells*10) #Make a guess for how much space we'll need
        cellylocations = np.zeros(maxcells*10)

    for i in range(0,maxcells):
        unplaced = True
        attempt = 0
        while unplaced:
            #test if new cell would not overlap with any other cells or walls
            xlocation =random.randint(int(0.05*Nx), int(0.95*Nx))
            ylocation =random.randint(int(0.85*Ny),int(0.95*Ny))
            consumption_rate = consumptionRates[i]
            unique_rates = sorted(set(consumptionRates))       # e.g. [0.0001, 1.0]
            rate_cumulative = {r: 0.0 for r in unique_rates}   # running totals per sub-pop

            newcell = cell.cell(xlocation, ylocation, cellsize, cellcollision, cellfuzzing, Nx, Ny, consumption_rate, i)
            if cellcollision == 0:
                cells.append(newcell)
                newcell.definesurfacesquares()
                break
            
            if cellcollision == 1:
                if collisionfunctions.check_collision_distance(i,xlocation,ylocation,cellxlocations,cellylocations,celldistance):
                    cellxlocations[i] = xlocation
                    cellylocations[i] = ylocation
                    cells.append(newcell)
                    newcell.definesurfacesquares()
                    break
                
            if cellcollision == 2:
                newoccupiedsites = newcell.definesurfacesquares()
                if len(np.intersect1d(alloccupiedsites,newoccupiedsites))==0: #This means we've placed it at an available stop
                    alloccupiedsites = np.union1d(alloccupiedsites,newcell.definesurfacesquares())
                    cells.append(newcell)
                    break
                
            #This is only triggered if cellcollision was >0 and we failed to place the cell
            attempt +=1
            if attempt > 10:
                del newcell
             #   print("Unable to place cell!")
                break

  #  print("Cells placed")
    #Increment the counter with one, so that any new cells that are created will get a new number            
    maxcells = maxcells + 1


    ##Run the main loop of our simulation
  #  print("Simulation Start")
    for t in range(steps):
        start = time.time()
        
        for ligand in ligands.values():
            ligand.diffuse(grid_walls_bool) # optimal FEM selected automatically


        #Degrade ligands
        #ligands["attractant"].decay()
        #ligands["metab1"].decay()
        step_consumed = 0
        startmove = time.time()
        #shuffle the list so we adress them in a random order
        random.shuffle(cells)
        #Have cells sense chemoattractant, set their target accordingly, consume and produce ligands, and move
        for thiscell in cells:
            #sensing
            thiscell.sense_multiple_attractants(ligands,[attractantweighing],persistence)
            
            #consuming
            totalconsumed = 0.0
            if thiscell.consumptionRate > 0:
                consumedpersite = ligands["attractant"].consume(
                    thiscell.occupiedsites,
                    thiscell.consumptionRate,
                    returndetail=True
                )
                totalconsumed = sum(consumedpersite)

            step_consumed += totalconsumed
            rate_cumulative[thiscell.consumptionRate] += totalconsumed
            #if metabolite1consumption >0:
                #ligands["metab1"].consume(thiscell.occupiedsites, metabolite1consumption,returndetail=False)
            
            #producing
            #ligands["metab1"].produce_singlevalue(thiscell.occupiedsites,totalconsumed*metaboliteproductionratio)
        
            #collision checking
            if thiscell.collision ==0:
                thiscell.move_simple(movedistance,grid_walls, t)
            if thiscell.collision ==1:
                cellxlocations[thiscell.id], cellylocations[thiscell.id] = thiscell.move_coarse(movedistance, Nx, Ny, grid_walls, cellxlocations, cellylocations, celldistance)
            if thiscell.collision ==2:
                alloccupiedsites = thiscell.move_fine(movedistance,Nx,Ny,alloccupiedsites)
                
            #Consider mitosis based on local attractant consumed
            if random.random() <(basemitosis+mitogenfactor*totalconsumed):
                    if thiscell.collision == 0:
                        cells,maxcells = thiscell.divide_simple(cells,maxcells)
                    if thiscell.collision == 1:
                        cells,cellxlocations,cellylocations,maxcells = thiscell.divide_coarse(cells,maxcells,cellxlocations,cellylocations,celldistance*2)
                    if thiscell.collision == 2:
                        cells, alloccupiedsites,maxcells = thiscell.divide_fine(alloccupiedsites,cells,maxcells)
            
            #consider dying
            if random.random()<basedeath:
                if thiscell.collision == 1:
                    cellxlocations[thiscell.id] = -1
                    cellylocations[thiscell.id] = -1
                if thiscell.collision == 2:
                    alloccupiedsites = np.setxor1d(alloccupiedsites,thiscell.occupiedsites)
                thiscell.alive = False
            
        #Take out our dead cells
        cells[:] = [thiscell for thiscell in cells if thiscell.alive]
        
        #timekeeping
        endmove = time.time()
        end = time.time()
        
        if enable_migrasomes:
            migrasome_locations = []
            for thiscell in cells:
                migrasome_locations.extend(thiscell.migrasomes_to_drop)
                thiscell.migrasomes_to_drop.clear()

            receive_migrasomes(ligands["attractant"], migrasome_locations, t)
            update_migrasomes(ligands["attractant"], t)
            A_secr = ligands["attractant"].secreted_cumulative

        endmove = time.time()
        end = time.time()


        #check if we need to add more space to the cellocationarray
        if cellcollision == 1:
            if (maxcells*2) > (len(cellxlocations)):
                cellxlocations = np.append(cellxlocations,np.zeros(len(cellxlocations)))
                cellylocations = np.append(cellylocations,np.zeros(len(cellxlocations)))
        
        A_cons += step_consumed
        A_grid = ligands["attractant"].grid.sum()
        balance_error = A_init + A_secr - A_cons - A_grid

        
        #Save data every this many steps using feather
        saveinterval = 200
        if t % saveinterval == 0:
        # Save attractant grid
            row = {
                "t": t,
                "agrid": A_grid,
                "dA_step": step_consumed,
                "A_cons": A_cons,        
                "A_secr": A_secr,    # ← new line
                "error": balance_error,
            }
            for r in unique_rates:
                row[f"c-{r}"] = rate_cumulative[r]   # cumulative per sub-pop
            balance_records.append(row)
            
            #mydatasaver.save_attractant(ligands["attractant"].grid, t)
            
            # Save cell data
            mydatasaver.save_cells(cells, t)
            mydatasaver.save_attractant_balance(balance_records)

            # Save migrasomes if they exist
            if enable_migrasomes and hasattr(ligands["attractant"], "active_migrasomes"):
                mydatasaver.save_migrasomes(ligands["attractant"].active_migrasomes, t)


    log_status(f"Completed {save_path} in {time.time()-start_time:.2f}s")


# ─── ARGPARSE & ENTRYPOINT ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="HPC Macrophage Chemotaxis")
parser.add_argument("-env", required=True, choices=HashMapEnvs.keys())
parser.add_argument("-fR", required=True, type=float)
parser.add_argument("-sR", required=True)  # can be "single" or float
parser.add_argument("-diff", "--diffusion", required=True, type=float)
parser.add_argument("-ep", "--epoch", required=True, type=int)
parser.add_argument("-root", default="SimulationResults_500")

# Simulation params
parser.add_argument("-Simulation_time", type=int, default=400000)
parser.add_argument("-Nx", type=int, default=500)
parser.add_argument("-Ny", type=int, default=500)
parser.add_argument("-cells", type=int, default=50)
parser.add_argument("-plotting", type=int, default=0)
parser.add_argument("-mig", type=int, default=0, choices=[0, 1],
                    help="Enable migrasomes (1=on, 0=off)")
parser.add_argument("-dmethod",
    choices=["auto", "dufort", "euler"],
    default="auto",
    help="Diffusion integrator: auto (default), dufort, or euler")



# Cell props
parser.add_argument("-cellsize", type=int, default=3)
parser.add_argument("-mitogenfactor", type=float, default=0)
parser.add_argument("-basemitosis", type=int, default=0)
parser.add_argument("-basedeath", type=float, default=0)

# Movement
parser.add_argument("-collision", type=int, default=0)
parser.add_argument("-celldistancefactor", type=float, default=2)
parser.add_argument("-persistence", type=float, default=0.7)
parser.add_argument("-movedistance", type=float, default=0.2)
parser.add_argument("-cellfuzzing", type=float, default=0.1)

# Chemotaxis
parser.add_argument("-attractantweighing", type=float, default=1.0)
parser.add_argument("-metabolitehalflife", type=int, default=1000)
parser.add_argument("-metaboliteproductionratio", type=float, default=0.2)
parser.add_argument("-metaboliteconsumption", type=float, default=0.0)

args = parser.parse_args()
enable_migrasomes = (args.mig == 1)
dmethod = args.dmethod.lower()

# Convert sR if needed
if args.sR.lower() == "single":
    sR_pair = "single"
else:
    sR_pair = float(args.sR)

# Build the save path
save_dir = os.path.join(
    args.root, args.env,
    f"fR:{args.fR}__sR:{sR_pair}",
    f"DiffusionRate_{args.diffusion}",
    f"epoch_{args.epoch}"
)

# Run it
sim_run(
    layout                  = args.env,
    save_path               = save_dir,
    steps                   = args.Simulation_time,
    Nx                      = args.Nx,
    Ny                      = args.Ny,
    maxcells                = args.cells,
    Consumption_rate_pair   = (args.fR, sR_pair),
    diffusion_rate          = args.diffusion,
    plotting                = args.plotting,
    cellsize                = args.cellsize,
    mitogenfactor           = args.mitogenfactor,
    basemitosis             = args.basemitosis,
    basedeath               = args.basedeath,
    collision               = args.collision,
    celldistancefactor      = args.celldistancefactor,
    persistence             = args.persistence,
    movedistance            = args.movedistance,
    cellfuzzing             = args.cellfuzzing,
    attractantweighing      = args.attractantweighing,
    metabolitehalflife      = args.metabolitehalflife,
    metaboliteproductionratio = args.metaboliteproductionratio,
    metaboliteconsumption   = args.metaboliteconsumption,
    enable_migrasomes       = enable_migrasomes,
    dmethod                =  dmethod,
)
