import utils as ut
import numpy as np
import obspy
import dask
import argparse
from obspy.clients.filesystem import sds
import os
import glob

def parse_paths(stats):
    folder = f"{output_folder}/{stats.starttime.year}/{stats.network}/{stats.station}/{stats.channel}.D/"
    file = f"{stats.network}.{stats.station}.{stats.location}.{stats.channel}.D.{stats.starttime.year}.{stats.starttime.julday:03d}"
    return folder, file
    
def preprocess_stream(stream, minf, maxf, downsample=1):
    stream.detrend('linear')
    stream.taper(max_percentage=0.05, type='cosine')
    stream.filter('bandpass', freqmin=minf, freqmax=maxf)
    stream.decimate(downsample, no_filter=True)
    for i, s in enumerate(stream):
        stream[i] = ut.one_bit(stream[i])
        stream[i] = ut.whiten(stream[i], minf, maxf)
    return stream

@dask.delayed
def process_file(input_path, output_folder, minf, maxf, downsample, fmt='MSEED'):
    try:
        stream = obspy.read(input_path, fmt=fmt)
    except:
        return

    stats = stream[0].stats
    folder, file = parse_paths(stats)
    
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, file)
    
    print(f'Processing {output_path}')
    try:
        stream = preprocess_stream(stream, minf, maxf, downsample)
        stream.write(output_path, format=fmt)
    except Exception as e:
        print(f"Could not save {output_path}:")
        print(e)
        

if __name__ == '__main__':

    
    #we parse the argument to know which channel we're interested in
    parser = argparse.ArgumentParser()
    parser.add_argument('job_index')
    args = parser.parse_args()    
    seisclient = sds.Client('../../sharing/Forillon')

    # this part takes care of parallelizing the multiple seismic channels 
    streams = seisclient.get_all_nslc()
    component_index = int(args.job_index)
    stream = streams[component_index]
    network, station, location, channel = stream
    print(stream)

    output_folder = '/home/alexim/scratch/Forillon_noise'

    MINF = 1
    MAXF = 40
    DOWNSAMPLE = 10

    files = sorted(glob.glob(seisclient.sds_root + f'/20*/{network}/{station}/{channel}.D/*'))

    delayed = []
    for file in files:
        tmp = process_file(file, output_folder, MINF, MAXF, DOWNSAMPLE)
        delayed.append(tmp)
        
    dask.compute(*delayed)
    
