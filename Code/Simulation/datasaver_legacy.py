#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used for adhoc tests not for main simulation code

@author: dmv
"""

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pandas as pd
import os
import csv
from collections import deque
#Use feather to efficiently save files to disk that can later be read in by R
class datasaver:
    def __init__(self,directoryname):
        self.directory = "./"+directoryname
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
    
    #This looks really slow, but takes less than a millisecond on my mediocre laptop
    def savegrid(self,grid,name):
        saveabledata = pa.array(np.ndarray.flatten(grid))
        schema = pa.schema([pa.field('nums', saveabledata.type)])
    
        with pa.OSFile(self.directory+"/"+name, 'wb') as sink:
            with pa.ipc.new_file(sink, schema=schema) as writer:
                batch = pa.record_batch([saveabledata], schema=schema)
                writer.write(batch)


    def savegrid2(self, grid, name):
        np.save(self.directory + "/" + name, grid)
    
    def savecells(self,cells, name, time_step):
        cellxlocs = []
        cellylocs = []
        cellids = []
        time_steps = []
        cell_consumption = []
        for cell in cells:
            cellxlocs.append(cell.xlocation)
            cellylocs.append(cell.ylocation)
            cellids.append(cell.id)
            cell_consumption.append(cell.consumptionRate)
            time_steps.append(time_step)
        saveabledata = pa.Table.from_arrays([cellxlocs, cellylocs,cellids, cell_consumption, time_steps], names=["xloc","yloc","id","consumption_rate", "time_step"])
        feather.write_feather(saveabledata, self.directory+"/"+name)


    #Not usually used
    def savecellscsv(self,cells,name,time):
        with open(self.directory+"/"+name+'.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            for cell in cells:
                xloc = cell.xlocation
                yloc = cell.ylocation
                cellid = cell.id
                writer.writerow([time,cellid,xloc,yloc])
        
 
    
        
        


