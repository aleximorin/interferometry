import numpy as np
import xarray as xr
import glob
from time import time

def load_dataarrays(outpath):
    
    files = sorted(glob.glob(outpath + '/*.nc'))
    
    da = []
    for f in files:
        print('\r', f, end='')
        tmp = xr.open_dataarray(f, chunks={'time': -1}, engine='h5netcdf')
        if tmp.indexes['pair'].duplicated().any():
            print(f" duplicate 'pair' values found in: {f}")
        else:
            da += [tmp]
    
    print('\n    concatenating...')
    correlations = xr.concat(da, dim='time')
    return correlations

if __name__ == '__main__':


    stack_length = '1h'
    outpath = '/home/alexim/scratch/Forillon_xcorr'
    
    correlations = load_dataarrays(outpath)
    
    correlations = correlations.sortby('time')
    correlations = correlations.resample(time=stack_length).mean()

    t0 = time()
    correlations.to_netcdf(f'stacked_xcorr_{stack_length}.nc', engine='h5netcdf')
    t1 = time()
    print(f'process took {t1 - t0:2f}s')
