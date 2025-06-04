import obspy
from obspy.core import UTCDateTime
import numpy as np

import scipy
import xarray as xr
import dask
import itertools

from matplotlib import dates as mdates
import utils as ut

import dask.array as da
import pandas as pd

from scipy.signal import correlate
from numpy.linalg import norm

import xarray as xr
from scipy.signal import stft
from scipy.signal.windows import tukey


def _load_files(client, channels, t0, t1):
    # we get the filenames from the obspy client, adding them to a list of files to load using dask
    delayed_files = []
    for chn in channels:
        files = client._get_filenames(*chn, t0, t1)
        for file in files:
            delayed_files.append(dask.delayed(obspy.read)(file, starttime=t0, endtime=t1))

    print(f'\r{t0} out of {t1}, loading {len(delayed_files)} files                        ', end='')
    
    # we need to call compute here because dask doesn't know how to deal with obspy data
    files = dask.compute(*delayed_files)
    
    return files

def _parse_files(files):
    
    # we create a stream and only add the loaded traces with a non-zero length, sometimes it loads empty data for unknown reasons
    stream = obspy.Stream()
    for s in files:
        if len(s) > 0:
            stream += s

    # once again another safety check
    stream = obspy.Stream([s for s in stream if len(s) > 1])

    # we do a merge check. sometimes the sampling rate is not *exactly* 1kHz, sometimes the data type is not correct - resample it fixes those issues
    try:
        print(f'trimming')
        stream.merge(method=1, fill_value=0)

    except Exception as e:
        print(f'couldn\'t merge traces, resampling')
        stream.resample(stream[0].stats.sampling_rate)
        stream.merge(method=1, fill_value=0)

    # we trim the data so that we have the maximum time span possible, filling the rest with zeros
    minstart = np.min([tr.stats.starttime for tr in stream])
    maxend =  np.max([tr.stats.endtime for tr in stream])
    stream.trim(minstart, maxend, pad=True, fill_value=0)    
    return stream

def _pair_format(tags: tuple):
    return f'{tags[0]} - {tags[1]}'

def _manage_pairs(stream):
    # here we get the seismic stations' names correctly, making sure they are ordered. this is needed to have a consistent number of station pairs
    stations = [f'{s.stats.station}.{s.stats.location}.{s.stats.channel}' for s in stream]
    order = np.argsort(stations)
    stations = np.array(stations)[order]

    # we compute a simple mapping of station pairs
    indices = np.arange(len(stations))[order] # this might be really dumb? it works for now but im pretty sure indices == order
    pair_tags = [_pair_format(tags) for tags in itertools.combinations(stations, 2)]
    pairs = dict(zip(pair_tags, itertools.combinations(indices, 2)))
    
    return stations, pairs, pair_tags

@dask.delayed
def preprocess_trace(trace, minf, maxf, downsample=1):
    trace.detrend('linear')
    trace.taper(max_percentage=0.05, type='cosine')
    trace.filter('bandpass', freqmin=minf, freqmax=maxf)
    trace.decimate(downsample, no_filter=True)
    trace = ut.one_bit(trace)
    trace = ut.whiten3(trace, minf, maxf)
    return trace.data
    
@dask.delayed
def compute_correlations(x0, x1, maxlag_index, freq, window_length_seconds):
    tmp = correlate(x0, x1, mode='same', method='fft')/len(x0)
    center = int(window_length_seconds * freq) // 2
    centered = tmp[center - maxlag_index : center + maxlag_index + 1]
    return centered

def compute_snr(correlations, index):
    tmp = correlations.where(index)
    snr = np.abs(tmp.max(dim='lag')) / tmp.std(dim='lag')
    return snr
    
def correlate_noise(client, channels, t0, 
                    dt=86400, window_length_seconds=60, 
                    minf=1.0, maxf=90.0, downsample=5,
                    overlap=0.5, maxlag_seconds=None, coordinates=None):

    # we load files in parallel using dask
    files = _load_files(client, channels, t0, t0 + dt)
    
    # we do some processing on the stream, merging traces and padding holes with zeros
    stream = _parse_files(files)

    # if we only have one stream loaded, there is nothing to cross-correlate the trace with and gg go next
    if len(stream) < 2:
        print(f'not enough data to compute cross correlations')
        print()
        return
        
    print()
    print(stream)

    # we compute every combination of station pairs and keep track of them through a simple mapping
    stations, pairs, pair_tags = _manage_pairs(stream)
    
    # locations is an interable of coordinates associated for every channel
    if coordinates is not None:
        distances = {tag: norm(coordinates.iloc[:, i] - coordinates.iloc[:, j]) for tag, (i, j) in pairs.items()}
    else:
        distances = {tag: 0.0 for tag, (i, j) in pairs.items()}
    
    # we compute a few parameters needed for a consistent data processing scheme
    freq = stream[0].stats.sampling_rate / downsample
    if maxlag_seconds is None:
        maxlag_seconds = window_length_seconds / 2 - 1 / freq

    # this tells us how many samples we keep in every direction (causal or acausal lags)
    maxlag_index = int(maxlag_seconds * freq)

    # we compute the shape of the final array
    ntraces = len(stream)
    ncombinations = len(pairs)
    overall_length = len(stream[0]) / freq / downsample
    nslider = int(overall_length / window_length_seconds / overlap) 
    
    # simple safety check
    if len(stream[0]) < window_length_seconds * freq:
        print(' - minimum length is too short for computing cross correlations.')
        return 

    # we will use dask to compute the cross-correlations in parallel 
    slider = stream.slide(window_length_seconds, int(overlap * window_length_seconds))

    delayed_correlations = []
    times = []
    print(f'computing cross-correlations')
    for k, substream in enumerate(slider):
        
        # we keep the time for ez access
        t = substream[0].times('matplotlib')
        times.append(t[0])

        delayed_traces = [preprocess_trace(s, minf, maxf, downsample) for s in substream]
        delayed_args = dict(shape=(substream[0].stats.npts // downsample,), dtype=float)
        traces = da.stack([da.from_delayed(s, **delayed_args) for s in delayed_traces])
        
        # this compute every possible pair with the corresponding indices
        for tag, (i, j) in pairs.items():
            ccf = compute_correlations(traces[i], traces[j], maxlag_index, freq, window_length_seconds)
            delayed_correlations.append(ccf)


    print('stacking and dasking')
    # we finally stack the dask array data properly   
    lazy_correlations = [da.from_delayed(delayed_corr, shape=(maxlag_index * 2 + 1,), dtype=float) for delayed_corr in delayed_correlations]
    correlations = da.stack(lazy_correlations).reshape(-1, ncombinations, maxlag_index * 2 + 1)
    correlations = correlations.rechunk('auto')

    correlations = xr.DataArray(correlations,
                                dims=('time', 'pair', 'lag'), 
                                coords={'time': pd.to_datetime(mdates.num2date(times)).values, 
                                        'pair': pair_tags, 
                                        'lag': np.linspace(-maxlag_seconds, maxlag_seconds, maxlag_index * 2 + 1),
                                        'dist': ('pair', [distances[p] for p in pair_tags])})

    return correlations

def cross_spectral_density_matrix(client, channels, t0, 
                                  dt=86400, window_length_seconds=60, 
                                  minf=1.0, maxf=50.0, downsample=10,
                                  overlap=0.5, nfbins=100, coordinates=None, 
                                  onebit=True, whiten=True):

    # we load files in parallel using dask
    files = _load_files(client, channels, t0, t0 + dt)
    
    # we do some processing on the stream, merging traces and padding holes with zeros
    stream = _parse_files(files)

    # if we only have one stream loaded, there is nothing to cross-correlate the trace with and gg go next
    if len(stream) < 2:
        print(f'not enough data to compute cross correlations')
        print()
        return
        
    print()
    print(stream)

    # we compute every combination of station pairs and keep track of them through a simple mapping
    stations, pairs, pair_tags = _manage_pairs(stream)
    
    stream.filter('bandpass', freqmin=minf, freqmax=maxf, zerophase=True)
    stream.decimate(downsample, no_filter=True)

    # we compute a few parameters needed for a consistent data processing scheme
    freq = stream[0].stats.sampling_rate
    data = [tr.data for tr in stream]

    if onebit:
        # sign the data to one bit, this is useful for noise correlation
        data = np.sign(data)

    # we use stft to do sliding window operations
    nperseg = int(window_length_seconds * freq)
    noverlap = int(overlap * nperseg)
    freqs, t, z = stft(data, fs=freq, nperseg=nperseg, noverlap=noverlap, axis=-1)

    if whiten:
        # we can normalize the amplitude spectra within a tukey window
        alpha = 0.1
        eps = 1e-10

        bandwidth = maxf - minf
        band = (minf - alpha * bandwidth / 2 <= freqs) & (freqs <= maxf + alpha * bandwidth / 2)
        nband = np.sum(band)

        window = np.zeros_like(freqs)
        window[band] = tukey(nband, alpha=alpha, sym=False)
        amp = np.abs(z)

        z = z / (amp + eps) * window[None, :, None]

    # we compute the cross-spectral density matrix using einsum for every pair of station (n, m) and every frequency and time (f, t)
    csd = np.einsum('nft, mft-> nmft', z, z.conj())

    # we convert the data to an xarray DataArray for easy manipulation
    start = UTCDateTime(stream[0].stats.starttime)
    datetime = np.array([start + ti for ti in t], dtype='datetime64[ns]')
    csd = xr.DataArray(csd, dims=['sta1', 'sta2', 'freqs', 'time'], coords={'sta1': stations, 'sta2': stations, 'freqs': freqs, 'time': datetime})

    # we downsample the frequencies to a fixed number of bins for memory usage, 
    # this could be changed if we wanted to use this product for cross-correlation functions
    fbins = np.linspace(freqs.min(), freqs.max(), nfbins + 1)
    csd = csd.groupby_bins('freqs', fbins).mean()
    csd = csd.rename({'freqs_bins': 'freqs'})
    csd = csd.assign_coords({'freqs': (['freqs'], 0.5 * (fbins[1:] + fbins[:-1]))})    

    # we need to store the data to a dataset for saving in netcdf format
    ds = xr.Dataset({'real': csd.real, 'imag': csd.imag})

    return ds


# this adds a functionality to xarray datasets, making it so that we're able to easily access station pairs
@xr.register_dataarray_accessor('cc')
class CCPairs:
    def __init__(self, data_array):
        self._obj = data_array
        self.stations = self._get_base_stations()

    def _get_base_stations(self):
        if 'pair' not in self._obj.dims:
            raise ValueError('The DataArray must have a "pair" dimension.')

        stations = self._obj.pair.str.split(' - ')[:, [0, -1]]
        return np.unique(stations)

    def select_pairs(self, base_station, other=None):
        """
        Select cross-correlation pairs based on a base station and other stations.
        Handles reversed pairs by flipping the lag dimension.

        Parameters:
        - base_station: str
            The station from which cross-correlation is computed.
        - other: str or list of str
            The other station(s) to select pairs with. Can be a single station or a list of stations.

        Returns:
        - A new DataArray containing only the selected pairs.
        """

        if other is None:
            other = set(self.stations) - {base_station}

        if isinstance(other, str):
            other = [other]  # Convert to list for consistent handling

        # initialize lists for selected data
        pairs = []
        pair_tags = []

        for station in other:
            # construct pair names
            direct_pair = _pair_format((base_station, station))
            reversed_pair = _pair_format((station, base_station))

            if direct_pair in self._obj.coords['pair'].values:
                # select direct pair
                data = self._obj.sel(pair=direct_pair)
                
            elif reversed_pair in self._obj.coords['pair'].values:
                # select reversed pair and flip along lags
                data = self._obj.sel(pair=reversed_pair)

                # we flip the data
                data = data.sel(lag=data.lag[::-1])

                # we also need to flip the lags and store the associated name
                data = data.assign_coords(lag=data.lag[::-1])

            else:
                raise ValueError(f'No matching pair found for "{base_station}" and "{station}".')

            pairs.append(data)
            pair_tags.append(direct_pair)
        
        # concatenate selected data along the 'pairs' dimension and reorder so that time is in the first dimension
        out = xr.concat(pairs, dim='pair')
        out = out.transpose('time', 'pair', 'lag')
        out = out.assign_coords(pair=pair_tags)
        out = out.sortby('pair')
        
        return out

if __name__ == '__main__':

    from obspy.clients.filesystem import sds
    from obspy.core import UTCDateTime
    
    # we create an obspy client that will be used for loading the data
    data_path = r'/home/alexim/projects/def-girouxb1/sharing/Forillon'
    seisclient = sds.Client(data_path)

    # we select only a few of the channels of interest
    channels = seisclient.get_all_nslc()
    indices = [0, 3, 7] # [0, 1, 3, 4, 7, 8, 11, 12, 13, 17, 21]
    selected_channels = [channels[i] for i in indices]

    # we access the coordinates so that we can associate a distance between every station pair
    df = ut.get_coordinates()
    ii = [ut.parse_chn_tag(chn) for chn in selected_channels]
    coords = df[ii]
    coords.columns = ['.'.join(chn[1:]) for chn in selected_channels]
    
    t1 = UTCDateTime('2024-11-01')
    t2 = UTCDateTime('2024-11-03')
    
    print(f'computing cross-correlations from {t1} to {t2}')

    # computation parameters for the lengths of the cross correlations. 
    dt = 1.0 * 86400
    window_length = 3600 * 2
    overlap = 0.5

    MINF = 1
    MAXF = 90
    DOWNSAMPLE = 5
    
    ti = t1
    # this part here should be parallelized over multiple tasks
    while ti + dt <= t2:
        print(f'processing {ti} to {ti + dt}')
        corr = correlate_noise(seisclient, selected_channels, ti, dt, 
                               window_length_seconds=window_length, 
                               maxlag_seconds=10, minf=MINF, maxf=MAXF, downsample=DOWNSAMPLE,
                               coordinates=coords)
        ti += dt


    print('hah')
    print(corr)
    print('hah')
    tmp = corr.cc.select_pairs(corr.cc.stations[0])
    print(tmp)
    print('hah')
    print(tmp.pair)

