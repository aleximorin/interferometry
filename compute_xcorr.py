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

def parse_channel(chn):
    return '.'.join(chn[1:])

@dask.delayed
def compute_correlations(x0, x1, maxlag_index, freq, window_length_seconds):
    tmp = correlate(x0, x1, mode='same', method='fft')/len(x0)
    center = int(window_length_seconds * freq) // 2
    centered = tmp[center - maxlag_index : center + maxlag_index + 1]
    return centered


def xarray_noise_correlation(client, channels,
                             t0, dt=86400, window_length_seconds=3600, 
                             overlap=0.5, maxlag_seconds=None, stack=False):

    delayed_files = []
    for chn in channels:
        files = client._get_filenames(*chn, t0, t0 + dt)
        for file in files:
            delayed_files.append(dask.delayed(obspy.read)(file, starttime=t0, endtime=t0 + dt))

    print(f'\r{t0} out of {t0 + dt}, loading {len(delayed_files)} files            ', end='')
    
    files = dask.compute(*delayed_files)
    
    stream = obspy.Stream()
    for s in files:
        if len(s) > 0:
            stream += s
    
    stream = obspy.Stream([s for s in stream if len(s) > 1])
    
    
    if len(stream) < 2:
        print(f' - not enough data to compute cross correlations')
        return

    try:
        print(f'\r{t0} out of {t0 + dt}, trimming                        ', end='\r')
        stream.merge(method=1, fill_value=0)

    except Exception as e:
        print(f'\r{t0} out of {t0 + dt}, couldn\'t merge traces, resampling   ', end='\r')
        stream.resample(stream[0].stats.sampling_rate)
        stream.merge(method=1, fill_value=0)

    print()
    print(stream)
    
    minstart = np.min([tr.stats.starttime for tr in stream])
    maxend =  np.max([tr.stats.endtime for tr in stream])
    stream.trim(minstart, maxend, pad=True, fill_value=0)    

    stations = [f'{s.stats.station}.{s.stats.location}.{s.stats.channel}' for s in stream]
    order = np.argsort(stations)
    stations = np.array(stations)[order]
    #stations = [f'{s.stats.station}.{s.stats.location}.{s.stats.channel}' for s in stream]

    indices = np.arange(len(stations))[order]
    pair_tags = [f'{t[0]} - {t[1]}' for t in itertools.combinations(stations, 2)]
    pairs = dict(zip(pair_tags, itertools.combinations(indices, 2)))
    
    freq = stream[0].stats.sampling_rate
    if maxlag_seconds is None:
        maxlag_seconds = window_length_seconds / 2 - 1 / freq
        
    maxlag_index = int(maxlag_seconds * freq)
    
    ntraces = len(stream)
    ncombinations = len(pairs)
    overall_length = len(stream[0]) / freq
    nslider = int(overall_length / window_length_seconds / overlap) 
    
    if len(stream[0]) < window_length_seconds * freq:
        print(' - minimum length is too short for computing cross correlations.')
        return [], []

    slider = stream.slide(window_length_seconds, int(overlap * window_length_seconds))

    delayed_correlations = []
    times = []

    print(f'\r{t0} out of {t0 + dt}, computing cross-correlations                    ', end='')
    
    for k, substream in enumerate(slider):
        substream.taper(max_percentage=0.05, type='cosine')   
        
        # we keep the time for ez access
        t = substream[0].times('matplotlib')
        times.append(t[0])
         
        traces = [s.data for s in substream]

        for tag, (i, j) in pairs.items():
            delayed_correlations.append(compute_correlations(traces[i], traces[j], maxlag_index, freq, window_length_seconds))

    lazy_correlations = [da.from_delayed(delayed_corr, shape=(maxlag_index * 2 + 1,), dtype=float) for delayed_corr in delayed_correlations]
    correlations = da.stack(lazy_correlations).reshape(-1, ncombinations, maxlag_index * 2 + 1)
    
    correlations = correlations.rechunk('auto')
    
    #correlations = da.from_delayed(delayed_correlations, shape=(nslider, ncombinations, maxlag_index * 2 + 1)) 
    #correlations = correlations.reshape(-1, ncombinations, maxlag_index * 2 + 1)

    correlations = xr.DataArray(correlations,
                                dims=('time', 'pair', 'lag'), 
                                coords={'time':  pd.to_datetime(mdates.num2date(times)).values, 
                                        'pair': pair_tags, 
                                        'lag': np.linspace(-maxlag_seconds, maxlag_seconds, correlations.shape[-1])})
    
    if stack:
        correlations = correlations.sum('time')

    return correlations

if __name__ == '__main__':

    outpath = '/home/alexim/scratch/Forillon_xcorr'
    
    try:
        os.makedirs(outpath)
    except OSError:
        pass
    
    seisclient = sds.Client('/home/alexim/scratch/Forillon_noise')
    channels = seisclient.get_all_nslc()
    indices = [0, 3, 4, 7, 8, 11, 12, 13, 17, 21, 24, 28]
    selected_channels = [channels[i] for i in indices]
    
    
    t1 = UTCDateTime('2023-06-01') # UTCDateTime('2023-06-01')
    t2 = UTCDateTime('2024-10-01')
    
    ti = t1
    
    # we initially stack the traces in 24 hour lengths - we then compute moving stacks of 1 week 
    dt = 1.0 * 86400 
    window_length = 3600
    overlap = 0.5
    
    delayed = []
    
    # this part here should be dask parallelized
    while ti + dt <= t2:
        print(f'\r{ti} out of {t2}', end='')
        file = os.path.join(outpath, f'{ti.year}{ti.julday:03d}.nc')
        corr = xarray_noise_correlation(seisclient, selected_channels, ti, dt, 
                                        window_length_seconds=window_length, 
                                        maxlag_seconds=30,
                                        overlap=overlap, stack=False)
        if corr is not None:
            print()
            corr.to_netcdf(file, engine='h5netcdf')#, compute=False)  
                
        ti += dt
    
    #_ = dask.compute(delayed)
