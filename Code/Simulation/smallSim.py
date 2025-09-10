#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:39:00 2024

@authors: dmv, mkm
"""
#This is a simulation of macrophages crossing the bridge of an Insall chamber

#these are libraries from the virtual environment
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import datetime
import numba
import argparse
from collections import deque
#these are supporting files in the local folder
import mazelayouts
import environment
import cell
import collisionfunctions
# At the top with other imports:
import SimulationSaver  # Add this import
import math

#MIGRASOME_DISTANCE_THRESHOLD    = 10     # was 150
#MIGRASOME_VELOCITY_THRESHOLD    = 0.002   # was 0.04
#MIGRASOME_PERSISTENCE_THRESHOLD = 0.8  # was 0.95
#MIGRASOME_COOLDOWN = 2000           

import os
#from environment import receive_migrasomes, update_migrasomes




# Define the log file name
LOG_FILE = "simulation_status.log"

METRICS_LOG_FILE = "migrasome_metrics_2.log"

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


# Function to append a message to the log file
#def log_status(message):
    #with open(LOG_FILE, "a") as f:
        #f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")



HashMapEnvs = {

    "Maze": mazelayouts.makewalls,
    "Empty": mazelayouts.makeemptycontainer,
    "Dish": mazelayouts.makeemptydish,
    "Open": mazelayouts.makeopensite,
    "Bridge": mazelayouts.makebridge,
    "Vessel": mazelayouts.makebloodvessel


}
#ENABLE_SAVING = True
#### Sims to run #Consumption_rates = [ (1, 0.01)]

#Consumption_rates = [(0.01), (0.1), (1)]


epochs = 1
#epochs = 1
#Consumption_rates = [(1,0)]
#Diffusion_Rates = [0.5]op
#create directories for each environment

path_to_save = "SimulationData_Sample"
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)



Simulation_time = int(350000)
Nx = 500
Ny = 500
maxcells = 50
Consumption_rate_pair=(1,0.0001)
diffusion_rate = 0.25
layout= "Maze"

path_to_save = f"SimulationData_Sample/{layout}/fR:{Consumption_rate_pair[0]}__sR:{Consumption_rate_pair[1]}//DiffusionRate_{diffusion_rate}/epoch_0"
print(path_to_save)


if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)



if isinstance(Consumption_rate_pair,float):
    higher_rate, lower_rate = Consumption_rate_pair, Consumption_rate_pair
else:
    higher_rate = max(Consumption_rate_pair)
    lower_rate = min(Consumption_rate_pair)

if not isinstance(Consumption_rate_pair, float):

    if (Consumption_rate_pair[0] == Consumption_rate_pair[1]):
        plot_map = {
            higher_rate: "green",
            lower_rate: "green"
        }
    else:
        plot_map = {
            higher_rate: "green",
            lower_rate: "white"
        }
else:
    plot_map = {

        higher_rate: "green",
        lower_rate: "green"
    }

start_time = time.time()

    
attractant_data = deque()
metabolite_data = deque()
cells_data = deque()

# print("Program Start")


#initialise our arguments
parser = argparse.ArgumentParser()
#General
parser.add_argument("-folder",default=path_to_save, help="Where to save the output")
parser.add_argument("-plotting",default = 50,help="How often to make a plot , 0 = never 1 = every timestep 2 = every two timesteps etc.",type=int)
parser.add_argument("-mig", type=int, default=0, choices=[0, 1],
                    help="Enable migrasomes (1=on, 0=off)")
parser.add_argument("-dmethod",
    choices=["auto", "dufort", "euler"],
    default="auto",
    help="Diffusion integrator: auto (default), dufort, or euler")

#Environmentf
parser.add_argument("-Nx",default=Nx, help="Width of the grid, default is 200",type=int)
parser.add_argument("-Ny",default=Ny, help="Height of the grid, default is 200",type=int)
parser.add_argument("-steps",default=Simulation_time, help="Duration of the simulation, default is 10000",type=int)
parser.add_argument("-cells",default=maxcells, help="Maximum number of cells to place, default is 100",type=int)
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


#Load arguments
args = parser.parse_args()
dmethod = args.dmethod.lower()
print("Args Parsed")
#initialise our datastoragex
mydatasaver = SimulationSaver.SimulationSaver(args.folder, layout)
plotinterval = args.plotting
# Environment parameters
Nx = args.Nx
Ny = args.Ny
steps = args.steps
maxcells = args.cells 

#Cell properties
cellsize = args.cellsize
mitogenfactor = args.mitogenfactor
basemitosis = args.basemitosis
basedeath = args.basedeath
enable_migrasomes = (args.mig == 1)

split = int (maxcells/2)
if type(Consumption_rate_pair) is not tuple:
    Consumption_rate_pair = (Consumption_rate_pair, "single")

if Consumption_rate_pair[1] == "single":
    consumptionRates = [Consumption_rate_pair[0] for _ in range(maxcells)]
else: 
    consumptionRates = [Consumption_rate_pair[0] for _ in range(split)] + [Consumption_rate_pair[1] for _ in range(split, maxcells)] 



#Cell movement properties
cellcollision = args.collision
celldistance = cellsize*args.celldistancefactor
persistence = args.persistence
movedistance= args.movedistance
cellfuzzing = args.cellfuzzing 


#parameters for ligands
#attractantparameters - this represents a primary attractant
attractanthalflife = 5000000 #how long-lived this ligand is - a high number means it breaks down slowly
attractantdiffusion = diffusion_rate #How fast it diffuses - a high number means it diffuses a lot
#attractantconsumption = 1 Modified to cell property
attractantweighing = float(2)#How strongly this is sensed. A negative number acts as a metabolite1

#metabolite1 - this represents a secondary attractant
metabolite1halflife = args.metabolitehalflife
metabolite1diffusion = diffusion_rate
metabolite1consumption = args.metaboliteconsumption
metabolite1weighing = float(args.attractantweighing)
metaboliteproductionratio = args.metaboliteproductionratio

# Create the grid walls

#grid_walls = mazelayouts.makebridge(Nx, Ny)
if layout in HashMapEnvs:
    
    grid_walls = HashMapEnvs[layout](Nx, Ny)
    grid_walls_bool = (grid_walls > 0).astype(bool)    
else:
    print("Environment not found")
    exit(1)

print("env instantiated")

save_interval = 1000  # Set your desired save interval
mydatasaver = SimulationSaver.SimulationSaver(path_to_save, layout)

alloccupiedsites = np.array([])
wallsites = np.array([])
# print("Adding Walls")
#Add walls to the set of occupied sites
# Add walls to the set of occupied sites using vectorized operations
wall_indices = np.argwhere(grid_walls > 0)
linear_indices = (wall_indices[:, 1] * Nx + wall_indices[:, 0]).astype(np.int64)
alloccupiedsites = np.sort(linear_indices)
wallsites = np.copy(alloccupiedsites)



print("Walls added")
#Create the chemical grids
ligands = {}
attractant_halflife = 5e10
ligands["attractant"] = environment.ligand(
        Nx, Ny, attractant_halflife, diffusion_rate,
        grid_walls_bool, "attractant", scheme=dmethod
    )

non_wall = (grid_walls < 1)

ligands["metab1"] = environment.ligand(
    Nx, Ny, metabolite1halflife, metabolite1diffusion, grid_walls,
    "metab1", scheme=dmethod
)

#   print("ligands created")
#Fill part of the grid with attractant
#for i in range(1, ligands["attractant"].grid.shape[0]-1):
#    for j in range(1, ligands["attractant"].grid.shape[1]-1):
#        if grid_walls[i,j]<1:
#            if j<1*Ny:
#                ligands["attractant"].grid_prev[i,j] = 1
    # 1) Mask non‐wall cells
non_wall = (grid_walls < 1)

# 2) Mask interior (skip the very border)
interior = np.zeros_like(non_wall, dtype=bool)
interior[1:-1, 1:-1] = True

# 3) Combine
mask = non_wall & interior

# 4) Assign in one shot
ligands["attractant"].grid_prev[mask] = 1
ligands["attractant"].grid[mask] = 1 



                
print("Environment filled")
print("Total Attractant mass at t=0: ", ligands["attractant"].grid_prev.sum())

#Do a first diffusion step on the grid, this is neccesary to do dufort diffusion later
for ligand in ligands.values():
    ligand.diffuse_euler(grid_walls_bool)

A_init = ligands["attractant"].grid_prev.sum()     # mass at t = 0
A_secr = 0.0                         # cumulative attractant secreted

A_cons = 0.0                                       # cumulative consumed
balance_records = []                               # rows for CSV


print("First Step Diffusion done")
#Place the cells
cells = list()

#pre-allocate space is we're saving their central locations only
if cellcollision ==1:
    cellxlocations = np.zeros(maxcells*10) #Make a guess for how much space we'll need
    cellylocations = np.zeros(maxcells*10)
    
for i in range(0,maxcells):
    unplaced = True
    attempt = 0
    while unplaced:
        #test if new cell would not overlap with any other cells or walls
        xlocation =random.randint(int(0.15*Nx), int(0.95*Nx))
        ylocation =random.randint(int(0.85*Ny),int(0.95*Ny))
        consumption_rate = consumptionRates[i]
        unique_rates = sorted(set(consumptionRates))       # e.g. [0.0001, 1.0]
        rate_cumulative = {r: 0.0 for r in unique_rates}   # running totals per sub-pop

        newcell = cell.cell(xlocation, ylocation, cellsize, cellcollision,
                    cellfuzzing, Nx, Ny, consumption_rate, i,
                    enable_migrasomes)

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
plotinterval = 0

for t in range(steps):
    start = time.time()
    
    #Diffuse ligands
    for ligand in ligands.values():
        ligand.diffuse(grid_walls_bool)

    #Degrade ligands
    #ligands["attractant"].decay()
    #ligands["metab1"].decay()
    
    startmove = time.time()
    #shuffle the list so we adress them in a random order
    random.shuffle(cells)
    #Have cells sense chemoattractant, set their target accordingly, consume and produce ligands, and move
    step_consumed = 0.0            # reset for this time-step

    for thiscell in cells:
        #sensing
        thiscell.sense_multiple_attractants(ligands,[attractantweighing,metabolite1weighing],persistence)
        
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
         #   ligands["metab1"].consume(thiscell.occupiedsites, metabolite1consumption,returndetail=True)
        
        #producing
        
        #ligands["metab1"].produce_singlevalue(thiscell.occupiedsites,totalconsumed*metaboliteproductionratio)
    
        #collision checking
        if thiscell.collision ==0:
            thiscell.move_simple(movedistance,grid_walls,t)
        if thiscell.collision ==1:
            cellxlocations[thiscell.id], cellylocations[thiscell.id] = thiscell.move_coarse(movedistance,Nx,Ny,grid_walls,cellxlocations,cellylocations,celldistance)
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
    #if t % 100 == 0 and t >= 8000:
        #log_migrasome_metrics(t, cells)

      # ─ Migrasome logic ─
    if enable_migrasomes:
        migrasome_locations = []
        for thiscell in cells:
            migrasome_locations.extend(thiscell.migrasomes_to_drop)
            thiscell.migrasomes_to_drop.clear()

        receive_migrasomes(ligands["attractant"], migrasome_locations, t)
        update_migrasomes(ligands["attractant"], t)
        A_secr = ligands["attractant"].secreted_cumulative


    #timekeeping
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



    if t % save_interval == 0:
        now = datetime.datetime.now()
    # Save attractant grid
       # row = {
       #     "t": t,
       #     "agrid": A_grid,
       #     "dA_step": step_consumed,
       #     "A_cons": A_cons,        
       #     "A_secr": A_secr,    # ← new line
       #     "error": balance_error,
       # }
      #  for r in unique_rates:
      #      row[f"c-{r}"] = rate_cumulative[r]   # cumulative per sub-pop
      #  balance_records.append(row)

        attractantname = "attractant"
        metabolitename = "metabolite"
        cellname = "cells"

        mydatasaver.savegrid(ligands["attractant"].grid, attractantname+ str(t))
        #mydatasaver.savegrid(ligands["metab1"].grid, metabolitename+ str(t))
        mydatasaver.savecells(cells, cellname+ str(t),t)

        #mydatasaver.save_attractant_balance(balance_records)

        # Save migrasomes if they exist
       # if enable_migrasomes and hasattr(ligands["attractant"], "active_migrasomes"):
        #    mydatasaver.save_migrasomes(ligands["attractant"].active_migrasomes, t)

    
    #Save data every this many steps using feather
    if plotinterval > 0:
        if(t % plotinterval == 0):
            startplot = time.time()
            plt.clf()
            plt.imshow(
                ligands["attractant"].grid.T + ligands["metab1"].grid.T,
                origin='lower',
                extent=[0, ligands["attractant"].grid.shape[0], 0, ligands["attractant"].grid.shape[1]],
                cmap='plasma',
                interpolation='none',
                vmin=0,
                vmax=1
            )
            plt.colorbar(label="Attractant Concentration")
            plt.title("Time = " + str(t) + " steps")

            # Plot cells as small scatter dots
            x_locs = [cell.xlocation for cell in cells]
            y_locs = [cell.ylocation for cell in cells]
            colors = [plot_map[cell.consumptionRate] for cell in cells]
            plt.scatter(
                x_locs, y_locs,
                s=5,  # Smaller dot size
                c=colors,
                alpha=0.7,
                edgecolors='none',
                marker='o'
            )

            # Plot migrasomes as red x's
# Replace:
#    for (x, y, _) in ligands["attractant"].active_migrasomes:
#        plt.plot(x, y, 'x', color='red', markersize=4, alpha=0.7, markeredgewidth=1)

# With:
            if hasattr(ligands["attractant"], "active_migrasomes"):
                for mig in ligands["attractant"].active_migrasomes:
                    x, y = mig[0], mig[1]  # Extract just x and y coordinates
                    plt.plot(x, y, 'x', color='red', markersize=4, alpha=0.7, markeredgewidth=1)

            endplot = time.time()
            print("plottime (not included)")
            print(endplot - startplot)
            plt.pause(0.000001)



