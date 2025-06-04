import numpy as np
import xarray as xr
import glob
from time import time
from dask.diagnostics import ProgressBar

def load_datasets(outpath):
    
    files = sorted(glob.glob(outpath + '/*.nc'))
    
    da = []
    for f in files:
        print('\r', f, end='')
        tmp = xr.open_dataset(f, chunks={'time': -1}, engine='h5netcdf')

        da += [tmp['real'] + 1j * tmp['imag']]
    
    print()
    print(f'loaded {len(da)} files')
    print('concatenating...')
    with ProgressBar():
        csd = xr.concat(da, dim='time')
    return csd

if __name__ == '__main__':


    stack_length = '1h'
    outpath = '/home/alexim/scratch/Forillon_csdm_whiten'
    
    csd = load_datasets(outpath)
    
    csd = csd.sortby('time')
    csd = csd.resample(time=stack_length).mean()

    print(csd)

    csd = xr.Dataset({'real': csd.real, 'imag': csd.imag})

    print()
    print(csd)

    t0 = time()
    csd.to_netcdf(f'csdm_{stack_length}.nc', engine='h5netcdf')
    t1 = time()
    print(f'process took {t1 - t0:2f}s')
