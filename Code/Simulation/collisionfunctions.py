#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:07:30 2024

@author: dmv
"""

import numba
import numpy as np
import math

#Test if we're colliding with a wall here
#I tried to make it faster by removing the de-linearization, but that just made it slower
@numba.jit(nopython=True)
def wall_collision_test_jitted(newoccupiedsites,walls,Nx):
    for point in newoccupiedsites:
        i = point%Nx
        j = math.floor(point / Nx)
        if walls[i,j]==1:
            return True
    return False

#Check distance for the simple distance-based collision system (coarse system)
@numba.jit(nopython=True)
def check_collision_distance(cellid,targetx,targety,xlocs,ylocs,maxproximity):
    for i in range(0,len(xlocs)):
        thisxloc = xlocs[i]
        if xlocs[i] == -1:
            continue #This means the cell is dead and we have assigned its id to this location to indicate it
        thisyloc = ylocs[i]
        if thisxloc == 0 and thisyloc ==0:
            continue #this spot is uninitialised
        if i != cellid: #Do not check distance from ourself
            #First check distance using an understimating distance measure 
            #if this is still distant enough, we don't need to check exactly
            xdistance = abs(thisxloc-targetx)
            if xdistance>maxproximity:
                continue
            ydistance = abs(thisyloc-targety)
            if ydistance>maxproximity:
                continue
            #For nearby cells check exactly
            distance = math.sqrt(pow(xdistance,2)+pow(ydistance,2))
            if distance<maxproximity:
                return False
    return True
            
    
#Check if two sets of spaces overlap - if so, move is rejected. If not, new set of occupied spaces is created
#Used by the 'fine' system
#Not Jitted - could not find a more efficient way to do it with jitted functions
def checkoverlapcollision(occupiedsites,oldsites,newsites):
    overlapwithold = sum(np.isin(newsites,oldsites,assume_unique=True))
    totaloverlap = sum(np.isin(newsites,occupiedsites,assume_unique=True))
    if totaloverlap > overlapwithold:
        return False, occupiedsites
    else:
        bannedsites = np.setxor1d(occupiedsites, oldsites,assume_unique=True)
        return True, np.union1d(bannedsites,newsites)
    