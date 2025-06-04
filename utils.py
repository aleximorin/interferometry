import glob
import numpy as np
import obspy
import pandas as pd
import xarray as xr
from scipy.signal import butter, filtfilt
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, CenteredNorm
from matplotlib import dates as mdates
fmt = mdates.DateFormatter('%y-%m-%d')

def whiten(tr, freqmin, freqmax):
    # taken from https://seismo-live.github.io/html/Ambient%20Seismic%20Noise/NoiseCorrelation_wrapper.html
    # By Celine Hadziioannou
    
    nsamp = tr.stats.sampling_rate
    
    n = len(tr.data)
    if n == 1:
        return tr
    else: 
        frange = float(freqmax) - float(freqmin)
        nsmo = int(np.fix(min(0.01, 0.5 * (frange)) * float(n) / nsamp))
        f = np.arange(n) * nsamp / (n - 1.)
        JJ = ((f > float(freqmin)) & (f<float(freqmax))).nonzero()[0]
            
        # signal FFT
        FFTs = np.fft.fft(tr.data)
        FFTsW = np.zeros(n) + 1j * np.zeros(n)

        # Apodization to the left with cos^2 (to smooth the discontinuities)
        smo1 = (np.cos(np.linspace(np.pi / 2, np.pi, nsmo+1))**2)
        FFTsW[JJ[0]:JJ[0]+nsmo+1] = smo1 * np.exp(1j * np.angle(FFTs[JJ[0]:JJ[0]+nsmo+1]))

        # boxcar
        FFTsW[JJ[0]+nsmo+1:JJ[-1]-nsmo] = np.ones(len(JJ) - 2 * (nsmo+1))\
        * np.exp(1j * np.angle(FFTs[JJ[0]+nsmo+1:JJ[-1]-nsmo]))

        # Apodization to the right with cos^2 (to smooth the discontinuities)
        smo2 = (np.cos(np.linspace(0., np.pi/2., nsmo+1))**2.)
        espo = np.exp(1j * np.angle(FFTs[JJ[-1]-nsmo:JJ[-1]+1]))
        FFTsW[JJ[-1]-nsmo:JJ[-1]+1] = smo2 * espo

        whitedata = 2. * np.fft.ifft(FFTsW).real
        
        tr.data = np.require(whitedata, dtype="float32")

        return tr

def whiten2(tr, eps=1e-10):
    data = tr.data
    data_fft = np.fft.rfft(data)
    data_fft = data / (abs(data) + eps)
    tr.data = np.fft.irfft(data_fft)
    return tr


def whiten3(tr, freqmin, freqmax, alpha=0.1, eps=1e-10):
    freqs = np.fft.rfftfreq(tr.stats.npts, tr.stats.delta)
    bandwidth = freqmax - freqmin
    band = (freqmin - alpha * bandwidth / 2 <= freqs) & (freqs <= freqmax + alpha * bandwidth / 2)
    nband = np.sum(band)

    window = np.zeros_like(freqs)
    window[band] = tukey(nband, alpha=alpha, sym=False)

    data = tr.data
    data_fft = np.fft.rfft(data)
    amp = np.abs(data_fft)

    data_fft = data_fft / (amp + eps) * window
    tr.data = np.fft.irfft(data_fft, n=tr.stats.npts)

    return tr

def running_average(tr, window_size):
    data = tr.data
    data = data - data.mean()
    rm = np.ones(window_size)/window_size
    tr.data = data/np.convolve(np.abs(data), rm, mode='same')
    return tr

def one_bit(tr):
    tr.data = np.sign(tr.data)
    return tr

def parse_chn_tag(chn):
    _, sta, ncomp, _ = chn
    sta = sta.replace('000', '')
    ncomp = 'tri' if ncomp == '3' else 'z'
    return sta + ncomp

def parse_chn_name(chn):
    sta, ncomp, _ = chn.split('.')
    sta = sta.replace('000', '')
    ncomp = 'tri' if ncomp == '3' else 'z'
    return sta + ncomp

def get_coordinates():
    coordinates = pd.read_csv('../geophone_coordinates_final.csv', index_col=1).iloc[:, 1:]
    return coordinates.T
    
def _apply_filter(data, fs, cutoff_freq, btype, degree=5):
    b, a = butter(5, cutoff_freq, btype=btype, fs=fs)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def xr_filtfilt(data, cutoff_freq, filter_type, degree=5):
    fs = int(1/(data.lag[1] - data.lag[0]))
    filtered_data = xr.apply_ufunc(_apply_filter, 
                                   data,
                                   input_core_dims=[['lag']],
                                   kwargs={'cutoff_freq': cutoff_freq, 'fs':fs, 'btype': filter_type, 'degree': degree},
                                   output_core_dims=[['lag']],
                                   vectorize=True,
                                   dask='allowed')
    return filtered_data
    
def weighted_lstsq(x, y, w, intercept=True, axis=-1, verbose=False, nugget=1e-16):

    # mostly following along https://en.wikipedia.org/wiki/Weighted_least_squares
    
    # we first reorder the axis of interest at the end
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    w = np.moveaxis(w, axis, -1)

    w = np.sqrt(w)
    
    # we add one more axis at the end and check if we want to fit an intercept
    x = x.reshape(*x.shape, 1)
    if intercept:
        ones = np.ones(x.shape[-2]).reshape(x.shape)
        x = np.concatenate([x, ones], axis=-1)

    if verbose:
        print()
        print(f'{x.shape=}')
        print(f'{y.shape=}')
        print(f'{w.shape=}')

    # nugget for stabilizing the inverse? does this help?
    nug = np.eye(x.shape[-1]) * nugget
    
    # here we define the terms in the weighted linear regressions using numpy's einsum confusing notation
    # we have that (X^T W X) B = X^T W y

    # (X^T W X)
    left = np.einsum('...ij, ...k, ...il -> ...jl', x, w, x)

    # X^T W y
    right = np.einsum('...ij, ...k, ...i -> ...j', x, w, y)

    # np.linalg.inv inverts along the two last axis of our array
    invleft = np.linalg.inv(left + nug) 

    # we can then solve for beta by computing inv(X^T W X) X^T W y
    beta = np.linalg.solve(left, right)

    # we compute our estimates of y using our newfound model as well as residuals
    yhat = np.einsum('...ij, ...j -> ...i', x, beta)
    r = y - yhat

    # this lets us easily compute the covariance matrix of our parameters
    sigma = np.einsum('...i, ...i, ...i -> ...', r, w, r) / (x.shape[-2] - x.shape[-1])
    cov = sigma[:, :, None, None] * invleft
    
    if verbose:
        print()
        print(f'{left.shape=}')
        print(f'{right.shape=}')
        print(f'{invleft.shape=}')
        print(f'{beta.shape=}')
        print(f'{r.shape=}')
        print(f'{sigma.shape=}')
        print()
        
    return beta, cov


def correlogram(xcorr, cutoff_freq=None, maxlag=None, vmax=1.0, logscale=True, filter_type='bandpass'):

    xcorr = xcorr/np.abs(xcorr).max('lag')

    if maxlag is None:
        maxlag = xcorr.lag.max()
        
    freq_title = ''
    if cutoff_freq is not None:
        xcorr = xr_filtfilt(xcorr, cutoff_freq=cutoff_freq, filter_type=filter_type)
        freq_title = f' filtered between {cutoff_freq[0]}-{cutoff_freq[-1]} Hz'
    
    index =  np.abs(xcorr.lag) <= maxlag
    xcorr = xcorr[:, index]

    vmax = np.nanquantile(np.abs(xcorr), vmax)
    if logscale:
        norm = SymLogNorm(linthresh=vmax, linscale=0.01)
    else:
        norm = CenteredNorm(vcenter=0, halfrange=vmax)

    fig, axs = plt.subplots(1, 2, width_ratios=[5, 1], sharey='all', figsize=(10, 2))
    axs[0].pcolormesh(mdates.date2num(xcorr.time), xcorr.lag, xcorr.values.T, cmap='RdBu', norm=norm)
    axs[1].plot(np.nanmean(xcorr, axis=0), xcorr.lag, c='k', lw=1)
    fig.subplots_adjust(wspace=0)

    axs[0].xaxis.set_major_formatter(fmt)
    axs[0].set_ylabel('lag (s)')
    axs[0].set_title(str(xcorr.pair.values) + freq_title)
    
    axs[1].set_xticks([])

    return fig, axs

def load_meteo():

    meteo_path = '../forillon/meteo/'
    meteo_files = glob.glob(meteo_path + '*.csv')

    meteo = load_weatherstation_data()

    rain = meteo.iloc[:, 7]
    rain[meteo.iloc[:, 2] <= 0] = 0

    temp = load_temperature_data()
    wlvl = load_wlvl_data()

    return rain, temp, wlvl

def load_weatherstation_data():

    meteo_path = '../forillon/meteo/'
    meteo_files = glob.glob(meteo_path + '*.csv')
    # processing the weather station data
    dfs = []
    for f in meteo_files:
        if 'Xc_MET' not in f:
            continue
        print(f)
        
        df = pd.read_csv(f, skiprows=1, index_col=1)
        print(df.index[0])

        df.index = pd.to_datetime(df.index, format=r'%m/%d/%y %I:%M:%S %p')
        dfs.append(df)
        
    df = pd.concat(dfs)
    df = df.sort_index()
    df = df.resample('15min').last()
    return df

def load_temperature_data():

    meteo_path = '../forillon/meteo/'
    meteo_files = glob.glob(meteo_path + '*.csv')
    # processing the temperature probe data
    dfs = []
    for f in meteo_files:
        if 'Rs_GP3' not in f:
            continue
        print(f)
        
        df = pd.read_csv(f, skiprows=9, index_col=1)
        df = df.dropna()
        df.index = pd.to_datetime(df.index, dayfirst=True)
        df = df.iloc[:, 1:-1]
        dfs.append(df)
        print(df.index.max())
        
    temp = pd.concat(dfs)
    temp = temp.sort_index()
    return temp

def load_wlvl_data():
    wlvl = pd.read_excel('../forillon/puits/Puitsv3.xlsx', sheet_name='NiveauNappe', index_col=0).iloc[:, :6]
    wlvl.columns = [col[-1] for col in wlvl.columns.str.split('_')]
    wlvl = wlvl.sort_index(axis=1)
    return wlvl

def load_tidal_data():
    path = r'C:\Users\alexi\Desktop\2scool4cool\passive seismic\meteo'
    files = glob.glob(path + "/AWAC*.txt")
    df = pd.DataFrame()
    for f in files:
        tmpdf = pd.read_csv(f, skiprows=12).iloc[1:]
        tmpdf = tmpdf.set_index(pd.to_datetime(tmpdf.iloc[:, 0])).iloc[:, 1:].astype(float)
        df = pd.concat([df, tmpdf], axis=0)

    df = df.sort_index()
    return df
 
def static_corr(xcorr, dtmax:int=None, maxlag:float=None):
    from scipy.signal import correlate


    if maxlag is None:
        maxlag  = xcorr.lag.max()

    subxcorr = xcorr.where(np.abs(xcorr.lag) <= maxlag, drop=True)
    subxcorr = subxcorr / np.abs(subxcorr).max('lag')
    gf = subxcorr.mean('time')

    corr = np.array([correlate(gf, t) for t in subxcorr.values])
    dt = corr.argmax(axis=1) - subxcorr.shape[-1]

    if dtmax is None:
        dtmax = xcorr.lag.max().values
        
    dtmax = dtmax // (xcorr.lag[1] - xcorr.lag[0]).values
    
    dt[dt > dtmax] = dtmax
    dt[dt < -dtmax] = -dtmax

    shift = np.array([np.roll(xcorr[i], dt[i]) for i in range(xcorr.shape[0])])
    shift = xr.DataArray(shift, dims=xcorr.dims, coords=xcorr.coords)

    return shift, dt

def load_dem():
    from scipy.interpolate import griddata, RegularGridInterpolator
    dem_path = r"C:\Users\alexi\Desktop\2scool4cool\passive seismic\terrain_a2022 - Copie\crop_dem.tif"
    dem = xr.load_dataarray(dem_path)[:, ::1, ::1]
    dem = dem.rio.reproject("EPSG:2142", nodata=np.nan)#, shape=(1000, 1000))

    # we will fill in the NaN values in the DEM with the nearest value
    x, y = np.meshgrid(dem["x"], dem["y"])
    valid_mask = ~np.isnan(dem.values[0])

    dem_interp_values = griddata(
        (x[valid_mask], y[valid_mask]), 
        dem.values[0, valid_mask], 
        (x, y), 
        method="nearest"
    )

    # we convert it back to xarray for... reasons
    dem_interp = xr.DataArray(dem_interp_values[None], coords=dem.coords, dims=dem.dims)
    zf = RegularGridInterpolator((dem_interp.y, dem_interp.x), dem_interp.values[0], bounds_error=False, method='linear', fill_value=None)

    return dem_interp, zf
