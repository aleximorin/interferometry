import obspy
from obspy.clients.filesystem import sds
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import correlate

import dask
import dask.array as da
from dask.diagnostics import ProgressBar

import os
import itertools

import matplotlib.pyplot as plt
from matplotlib import dates as mdates

import utils as ut
import interferometry as interf

from time import time

if __name__ == '__main__':

    MINF = 1
    MAXF = 40
    DOWNSAMPLE = 5
  
    import argparse
    # we define the argument parser
    parser = argparse.ArgumentParser(description='preprocessing and computing cross-correlations for a given number of jobs')
    parser.add_argument('--task-id', type=int, required=True, help='task id of the current SLURM job')
    parser.add_argument('--total-tasks', type=int, required=True, help='total number of SLURM jobs created')
    args = parser.parse_args()

    # we extract the task ID and total tasks
    task_id = args.task_id
    total_tasks = args.total_tasks
    print(f'task id {task_id + 1} out of {total_tasks}')
    
    # we create a folder associated with the outputs of the processing
    outpath = '/home/alexim/scratch/Forillon_xcorr'
    try:
        os.makedirs(outpath)
    except OSError:
        pass

    # we create an obspy client that will be used for loading the data
    data_path = '/home/alexim/projects/def-girouxb1/sharing/Forillon'
    seisclient = sds.Client(data_path)

    # we select only a few of the channels of interest
    channels = seisclient.get_all_nslc()
		    
    indices = [0, 1, 3, 4, 7, 8, 11, 12, 13, 17, 21, 25, 29]
    #indices = [0, 3]
    selected_channels = [channels[i] for i in indices]

    # we access the coordinates so that we can associate a distance between every station pair
    df = ut.get_coordinates()
    ii = [ut.parse_chn_tag(chn) for chn in selected_channels]
    coords = df[ii]
    coords.columns = ['.'.join(chn[1:]) for chn in selected_channels]

    # those are the dates we are interested in for our analysis
    t1 = UTCDateTime('2023-06-01')
    t2 = UTCDateTime('2025-09-24')

    print(f'computing cross-correlations from {t1} to {t2}')

    # we calculate the total number of days and divide them among the tasks
    total_days = int((t2 - t1) / 86400)
    days_per_task = total_days // total_tasks
    remainder_days = total_days % total_tasks
    print(f'{total_days} total days')

    # Determine the range of days for the current task
    start_day = days_per_task * (task_id) + min(task_id, remainder_days)
    end_day = start_day + days_per_task + (1 if task_id <= remainder_days else 0)
    print(start_day, end_day)
    
    # Adjust ti range
    ti = t1 + start_day * 86400
    tend = t1 + end_day * 86400
    print(f'task will compute from {ti} to {tend}, i.e. {(tend - ti) // 86400} days')

    # computation parameters for the lengths of the cross correlations. for now, dt needs to be 1 day, or we would need to modify the saving scheme (which would not be too hard to generalize). 
    dt = 1.0 * 86400
    window_length = 3600
    overlap = 0.5
    
    counter = 1

    # this part here should be parallelized over multiple tasks
    while ti < tend:
        print(f'task {task_id + 1}/{total_tasks}: processing {ti} to {ti + dt}')
        file = os.path.join(outpath, f'{ti.year}{ti.julday:03d}.nc')
        try:
            toc = time()
            corr = interf.correlate_noise(seisclient, selected_channels, ti, dt,  
                                          window_length_seconds=window_length, maxlag_seconds=2, 
                                          minf=MINF, maxf=MAXF, downsample=DOWNSAMPLE,
                                          coordinates=coords)
            if corr is not None:
                corr.to_netcdf(file, engine='h5netcdf')#, compute=False)
            tic = time()
            print(f'processing took {tic - toc:.2f}s, counter = {counter}')

        except Exception as e:
            print(f' something occured ??? ')
            print(e)
            print()

        counter += 1
        ti += dt

    print('safely finished')
