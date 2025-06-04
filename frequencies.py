import numpy as np
import matplotlib.pyplot as plt
import obspy
import dask
from dask.diagnostics import ProgressBar

import xarray as xr
from obspy.clients.filesystem.sds import Client
from matplotlib.dates import num2date
from scipy.signal import spectrogram
from pandas import to_datetime
import utils as ut
import os

def process_data(file, fmin=1.0, fmax=50.0, 
                 decimate_factor=10, seconds_per_seg=60, overlap_factor=2, 
                 stack_freq='1h', nfbins=101, log=False):
    try:
        st = obspy.read(file)
        tr = st[0]
        tr.detrend('demean')
        tr.detrend('linear')
        tr.filter('bandpass', freqmin=fmin, freqmax=fmax, zerophase=True)

        # some stupid filters for the 20Hz harmonics?
        #tr.filter('bandstop', freqmin=19, freqmax=21, zerophase=True)
        #tr.filter('bandstop', freqmin=39, freqmax=41, zerophase=True)

        tr.decimate(decimate_factor, no_filter=True)
        tr = ut.one_bit(tr)

        fs = tr.stats.sampling_rate
        nperseg = int(seconds_per_seg * fs)
        noverlap = nperseg // overlap_factor
        nfft = nperseg * 1.0
        f, t, z = spectrogram(tr.data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, mode='magnitude')

        t = num2date(t / 3600 / 24 + tr.times('matplotlib')[0])
        datetime = to_datetime(t).values
        da = xr.DataArray(z, dims=['frequency', 'time'], coords={'frequency': f, 'time': datetime})

        if log:
            fbins = np.geomspace(fmin, fmax, nfbins)
        else:
            fbins = np.linspace(fmin, fmax, nfbins)

        labels = (fbins[:-1] + fbins[1:])/2
        da = da.groupby_bins('frequency', bins=fbins,  labels=labels).mean().rename({'frequency_bins': 'frequency'})
        da = da.resample({'time': stack_freq}).median() # mean()

        return da
    except Exception as e:
        print('problem with current file')
        print(file)
        print(e)
        return None

if __name__ == '__main__':
    
    import argparse
    # we define the argument parser
    parser = argparse.ArgumentParser(description='processing our seismic data into downsampled spectrograms')
    parser.add_argument('--task-id', type=int, required=True, help='task id of the current SLURM job')
    args = parser.parse_args()
    task_id = args.task_id

    data_path = r'/home/alexim/projects/def-girouxb1/sharing/Forillon'
    seisclient = Client(data_path)
    channels = seisclient.get_all_nslc()
    
    chn = channels[task_id]
    files = sorted(seisclient._get_filenames(*chn, starttime=obspy.UTCDateTime(2023, 1, 1), endtime=obspy.UTCDateTime(2026, 1, 1)))
    print(f'found {len(files)} files for {chn}')

    # we create a folder associated with the outputs of the processing
    outpath = '/home/alexim/scratch/spectrograms/'
    try:
        os.makedirs(outpath)
    except OSError:
        pass

    delayed_tasks = []
    kwargs = dict(fmin=1.0, fmax=90.0, decimate_factor=5, seconds_per_seg=60, overlap_factor=2, log=True, nfbins=101)
    for file in files:
        tmp = dask.delayed(process_data)(file, **kwargs)
        delayed_tasks.append(tmp)
    
    with ProgressBar():
        tmp = dask.compute(*delayed_tasks)

    tmp = [t for t in tmp if t is not None]

    da = xr.concat(tmp, dim='time')
    da.to_netcdf(os.path.join(outpath, '.'.join(chn) + '.nc'), engine='h5netcdf')
