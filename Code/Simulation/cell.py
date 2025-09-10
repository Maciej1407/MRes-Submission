#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: dmv, mkm
"""
import numpy as np
import numba
import random
import math
import collisionfunctions
from collections import deque
import logging
import time

""" Fun experiment, not used in thesis
MIGRASOME_DISTANCE_THRESHOLD    = 6     # was 150
MIGRASOME_VELOCITY_THRESHOLD    = 0.002   # was 0.04
MIGRASOME_PERSISTENCE_THRESHOLD = 0.65  # was 0.95
MIGRASOME_COOLDOWN = 2000                # drop less frequently
        # Steps between drops
MIGRASOME_HISTORY_SIZE = 50            
"""

# Configure logging
logging.basicConfig(
    filename="migrasome_debug.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


@numba.jit(nopython=True)
def sense_multiple_attractants_average_fuzzy_jitted(occupiedsites,xlocation,ylocation,size,ligands,weighing,fuzzing, consumption):
    totalsquares = 0
    totalxvalues = 0
    totalyvalues = 0
    totalxlocations =0
    totalylocations = 0
    totalattractant = 0
    Nx = ligands[0].shape[0]
    for point in occupiedsites:
        i = point%Nx
        j = math.floor(point / Nx)
        gridnumber = 0
        netattraction = 0.0
        for thisligand in ligands:
            totalsquares +=1
            totalxlocations +=i
            totalylocations +=j
            #TODO  Have some sort of explicit receptor occupancy?
            #if consumption != 0:
                #perceivedattractant = V_MAX * (thisligand[i,j] / ( thisligand[i,j] + consumption) )
            #else:
                #perceivedattractant = V_MAX * thisligand[i,j]
            perceivedattractant = (thisligand[i,j])*weighing[gridnumber]
            netattraction +=perceivedattractant
            gridnumber +=1
        netattraction += random.uniform(-1*fuzzing,fuzzing)
        if netattraction <0:
            netattraction=0 #Negative attraction values lead to a lot of weird effects
        
        totalattractant += netattraction
        totalxvalues += netattraction*i
        totalyvalues += netattraction*j
        
    if totalxvalues !=0 or totalyvalues != 0:
        desiredx = totalxvalues/totalattractant
        desiredy = totalyvalues/totalattractant
        if desiredx != xlocation or desiredy != ylocation:
            xdirection = desiredx-xlocation
            ydirection = desiredy-ylocation
            targetangle = math.atan2(ydirection, xdirection)
            return targetangle 

    return random.uniform(-1*math.pi,math.pi)

#This returns the squares that are covered by this cell, used for determining where to consume and collide and such
@numba.jit(nopython=True)
def definesurfacesquares_jitted(xlocation,ylocation,size,Nx,Ny):
    occupiedsites = np.zeros(size*size*5,dtype=np.int64)
    sitecount = 0
    for i in range(max(math.floor(xlocation-size),0), min(math.ceil(xlocation+size),Nx-1)):
        for j in range(max(math.floor(ylocation-size),0),min(math.ceil(ylocation+size),Ny-1)):
            if math.sqrt(pow(xlocation-i,2)+pow(ylocation-j,2))<size:
                occupiedsites[sitecount]= j*Nx+i
                sitecount +=1
    return(occupiedsites[0:sitecount])


#Calculate a new angle between previous angles using a persistence value
@numba.jit(nopython=True)
def make_angle_jitted(targetangle,newangle,persistence):
    delta = newangle - targetangle
    delta = (delta + math.pi) % (2 * math.pi) - math.pi  # Wrap to [-π, π]
    return targetangle + (1 - persistence) * delta

#This is used for a simple move that does not need to be checked for a collision, which is much faster
@numba.jit(nopython=True)    
def move_jitted(maxdistance,xlimit,ylimit,targetangle,xlocation,ylocation,size):
    delta_x = maxdistance * math.cos(targetangle)
    delta_y = maxdistance * math.sin(targetangle)
    targetx = xlocation+delta_x
    targety = ylocation+delta_y
    if abs(delta_x) < 1e-6 and abs(delta_y) < 1e-6:
        return xlocation, ylocation  # No movement
    #If any of these are true, the move is out of bounds and should not happen
    if (targetx >=(xlimit-size)) or (targetx <=size):
        return xlocation,ylocation
    if (targety >=(ylimit-size)) or (targety <=size):
        return xlocation,ylocation
    return targetx,targety

#This is used for a move that can be checked for collision simply by distance, which is fast
@numba.jit(nopython=True)    
def move_coarse_collision_detection_jitted(myid,maxdistance,xlimit,ylimit,targetangle,xlocation,ylocation,xlocs,ylocs,maxproximity):
    delta_x = maxdistance * math.cos(targetangle)
    delta_y = maxdistance * math.sin(targetangle)
    targetx = xlocation+delta_x
    targety = ylocation+delta_y
    if not collisionfunctions.check_collision_distance(myid,targetx,targety,xlocs,ylocs,maxproximity):
        return xlocation,ylocation
        
    #If any of these are true, the move is out of bounds and should not happen
    if (targetx >=(xlimit-maxproximity)) or (targetx <maxproximity):
        return xlocation,ylocation
    if (targety >=(ylimit-maxproximity)) or (targety <maxproximity):
        return xlocation,ylocation
    return targetx,targety

#This is used for a move that does need to be checked for a collision, which is much slower
#This does allow for much more complex cell shapes
#Currently not jittable, might be done in the future (but may not be much more efficient)
def move_with_fine_collision_detection(maxdistance,xlimit,ylimit,targetangle,size,xlocation,ylocation,selfoccupiedsites,occupiedsites):
    delta_x = maxdistance * math.cos(targetangle)
    delta_y = maxdistance * math.sin(targetangle)
    targetx = xlocation+delta_x
    targety = ylocation+delta_y
    #If any of these are true, the move is out of bounds and should not happen
    if (targetx >= xlimit) or (targetx <0):
        return occupiedsites,selfoccupiedsites,xlocation,ylocation
    if (targety >= ylimit) or (targety <0):
        return occupiedsites, selfoccupiedsites,xlocation,ylocation
    
    #check what new sites we would occupy
    newoccupiedsites = definesurfacesquares_jitted(targetx,targety,size,xlimit,ylimit) 
    movepossible = False
    #use this collisionfunction to check if that overlaps with already occupied sites that aren't our own
    movepossible, occupiedsites = collisionfunctions.checkoverlapcollision(occupiedsites,selfoccupiedsites,newoccupiedsites)
    if movepossible: 
        selfoccupiedsites = newoccupiedsites #set new location
        xlocation=targetx
        ylocation=targety
    return occupiedsites, selfoccupiedsites,xlocation,ylocation

class cell:
    def __init__(self, xlocation, ylocation, size, collision, fuzzing,
             Nx, Ny, consumptionRate=1, idnumber=0,
             enable_migrasomes=True):

        self.xlocation = xlocation
        self.ylocation = ylocation
        self.Nx = Nx
        self.Ny = Ny
        self.size = size
        self.targetangle = random.uniform(-math.pi, math.pi)
        self.occupiedsites = list()
        self.collision = collision
        self.fuzzing = fuzzing
        self.consumptionRate = consumptionRate
        self.distance = 0
        self.id = idnumber
        self.alive = True
        self.lastproduced = random.randint(-1000, -10)
        
        # Migrasome state - FIXED: initialize migrasomes_to_drop
        self.migrasomes_to_drop = []  # CRITICAL FIX
        self.migrasome_history = deque(maxlen=MIGRASOME_HISTORY_SIZE)
        self.migrasome_cooldown = 0
        self.migrasome_distance = 0.0
        self.last_direction = None
        self.migrasome_start_pos = (xlocation, ylocation)
        self.migrasome_start_time = 0
                # master switch: turn all migrasome code on/off
        self.enable_migrasomes = enable_migrasomes




        
    #Return the current squares occupied by this cell
    def definesurfacesquares(self): #Wrapper to allow for JIT enhancement of underlying function
        self.occupiedsites = definesurfacesquares_jitted(self.xlocation,self.ylocation,self.size,self.Nx,self.Ny)
        return self.occupiedsites
    
    #Return the angle corresponding to the local gradient of multiple attractants combined
    def sense_multiple_attractants(self,grids,weighing,persistence):#Wrapper to allow for JIT enhancement of underlying function
        #Sense the gradient and get a new angle from that
        newangle = sense_multiple_attractants_average_fuzzy_jitted(self.occupiedsites,self.xlocation,self.ylocation,self.size,[ligand.grid for ligand in grids.values()],weighing,self.fuzzing,self.consumptionRate)
        #Set a new target angle based on our previous angle, the new angle, and how persistent we are
        self.targetangle = make_angle_jitted(self.targetangle,newangle,persistence)
        
    #Move without complex collision detection, but avoiding moving into walls
    def move_simple(self, maxdistance, walls,t):
        old_x, old_y = self.xlocation, self.ylocation

        # Compute tentative move
        targetx, targety = move_jitted(
            maxdistance, self.Nx, self.Ny,
            self.targetangle, old_x, old_y, self.size
        )
        new_sites = definesurfacesquares_jitted(
            targetx, targety, self.size, self.Nx, self.Ny
        )

        # Wall collision test
        if not collisionfunctions.wall_collision_test_jitted(new_sites, walls, self.Nx):
            # Accept the move
            self.xlocation, self.ylocation = targetx, targety
            self.occupiedsites = new_sites

            # Migrasome logic
                        # Migrasome logic (improved)
            dx = self.xlocation - old_x
            dy = self.ylocation - old_y
            step_dist = math.hypot(dx, dy)
            current_direction = (dx, dy)

            if step_dist < 1e-6:
                return  # Skip updates if no real movement

                    # ───────────────── Migrasome logic ─────────────────
            if self.enable_migrasomes:          #  <<< NEW GUARD

                self.migrasome_history.append(current_direction)

                # Time tracking
                self.migrasome_cooldown -= 1
                elapsed_time = t - self.migrasome_start_time  # You'll need to pass `t` as argument to move_simple()

                # Net displacement from last drop
                net_dx = self.xlocation - self.migrasome_start_pos[0]
                net_dy = self.ylocation - self.migrasome_start_pos[1]
                net_disp = math.hypot(net_dx, net_dy)

                # Compute cosine similarity
                should_drop = False
                if (
                    self.migrasome_cooldown <= 0 and
                    net_disp >= MIGRASOME_DISTANCE_THRESHOLD and
                    elapsed_time >= 10  # Just to avoid div-by-zero
                ):
                    avg_dx = sum(d[0] for d in self.migrasome_history) / len(self.migrasome_history)
                    avg_dy = sum(d[1] for d in self.migrasome_history) / len(self.migrasome_history)
                    avg_mag = math.hypot(avg_dx, avg_dy)
                    cur_mag = math.hypot(dx, dy)

                    if avg_mag > 0 and cur_mag > 0:
                        cos_sim = (avg_dx * dx + avg_dy * dy) / (avg_mag * cur_mag)

                        # NEW: Check velocity threshold
                        avg_velocity = net_disp / elapsed_time
                        if (
                            cos_sim >= MIGRASOME_PERSISTENCE_THRESHOLD and
                            avg_velocity >= MIGRASOME_VELOCITY_THRESHOLD  # new constant
                        ):
                            should_drop = True

                if should_drop:
                    self.migrasomes_to_drop.append((int(self.xlocation), int(self.ylocation)))
                    self.migrasome_start_pos = (self.xlocation, self.ylocation)
                    self.migrasome_start_time = t
                    self.migrasome_history.clear()
                    self.migrasome_cooldown = MIGRASOME_COOLDOWN

        
        
    #Move with a simple center-based collision detection
    def move_coarse(self,maxdistance,xlimit,ylimit,walls,xlocs,ylocs,maxproximity):
        targetx,targety = move_coarse_collision_detection_jitted(self.id,maxdistance,xlimit,ylimit,self.targetangle,self.xlocation,self.ylocation,xlocs,ylocs,maxproximity)
        if walls[math.floor(targetx),math.floor(targety)]!=1:
            self.xlocation, self.ylocation = targetx,targety
            self.definesurfacesquares()
        return targetx, targety
    
    #Move with precise shape-based collision detection
    def move_fine(self,maxdistance,xlimit,ylimit,occupiedsites): 
        occupiedsites, self.occupiedsites, self.xlocation, self.ylocation = move_with_fine_collision_detection(maxdistance,xlimit,ylimit,self.targetangle,self.size,self.xlocation, self.ylocation,self.occupiedsites,occupiedsites)
        return occupiedsites

    #Divide without collision detection (can divide into walls, but not outside of area)
    def divide_simple(self,cells,maxcells):
       dividetries = 0
       while dividetries <2:
           dividetries +=1
           #Pick a random angle
           mitosisangle = random.uniform(-1*math.pi,math.pi)
           delta_x = self.size*2 * math.cos(mitosisangle)
           delta_y = self.size*2 * math.sin(mitosisangle)
           xlocation = self.xlocation + delta_x
           ylocation = self.ylocation + delta_y
           if xlocation>self.Nx or ylocation>self.Ny:
               continue
           if xlocation<0 or ylocation<0:
               continue
           newcell = cell(xlocation,ylocation,self.size,self.collision,self.fuzzing,self.Nx,self.Ny,maxcells)
           cells.append(newcell)
           newcell.definesurfacesquares()
           maxcells = maxcells + 1
           return cells,maxcells
    
    #Divide with coarse collision detection
    def divide_coarse(self,cells,maxcells,cellxlocations,cellylocations,celldistance):
        dividetries = 0
        while dividetries <2:
            dividetries +=1
            #Pick a random angle
            mitosisangle = random.uniform(-1*math.pi,math.pi)
            delta_x = max(self.size*2,celldistance) * math.cos(mitosisangle)
            delta_y = max(self.size*2,celldistance) * math.sin(mitosisangle)
            xlocation = self.xlocation + delta_x
            ylocation = self.ylocation + delta_y
            if xlocation>self.Nx or ylocation>self.Ny:
                continue
            if xlocation<0 or ylocation<0:
                continue
            if collisionfunctions.check_collision_distance(-1,xlocation,ylocation,cellxlocations,cellylocations,celldistance):
                newcell = cell(xlocation,ylocation,self.size,self.collision,self.fuzzing,self.Nx,self.Ny,maxcells)
                cellxlocations[maxcells] = xlocation
                cellylocations[maxcells] = ylocation
                cells.append(newcell)
                newcell.definesurfacesquares()
                maxcells = maxcells + 1
                break
        return cells, cellxlocations,cellylocations,maxcells
    
    #Divide with fine collision detection
    def divide_fine(self,alloccupiedsites,cells,maxcells):
        dividetries = 0
        while dividetries <2:
            dividetries +=1
            #Pick a random angle
            mitosisangle = random.uniform(-1*math.pi,math.pi)
            delta_x = self.size*2 * math.cos(mitosisangle)
            delta_y = self.size*2 * math.sin(mitosisangle)
            xlocation = self.xlocation + delta_x
            ylocation = self.ylocation + delta_y
            if xlocation>self.Nx or ylocation>self.Ny:
                continue
            if xlocation<0 or ylocation<0:
                continue
            newcell = cell.cell(xlocation,ylocation,self.cellsize,self.cellcollision,self.cellfuzzing,self.Nx,self.Ny,maxcells)
            newoccupiedsites = newcell.definesurfacesquares()
            if sum(np.isin(alloccupiedsites,newoccupiedsites))==0:
                alloccupiedsites = np.union1d(alloccupiedsites,newoccupiedsites)
                cells.append(newcell)
                maxcells = maxcells + 1 #Increment for the next cell only if previous number is actually used
                break
        return cells,alloccupiedsites,maxcells
    
    #remove self
    def die(self):
        del self
       
    