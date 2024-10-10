import numpy as np
import sys, os
import matplotlib.pyplot as plt
import scipy.ndimage
#import astropy.stats.sigma_clipping
import scipy.signal
import time
import os
import configuration as conf
from scipy.signal import find_peaks
from scipy.sparse import spdiags, linalg, diags
from scipy.linalg import norm


def openStored(path):
    files = sorted(os.listdir(path))
    I_list = [os.path.join(path, filename) for filename in files if filename.startswith('I')]
    Q_list = [os.path.join(path, filename) for filename in files if filename.startswith('Q')]
    chan_I = np.array([np.load(filename) for filename in I_list])
    chan_Q = np.array([np.load(filename) for filename in Q_list])
    return chan_I, chan_Q

def normalize_and_stack(path, bb_freqs, lo_freqs):
    chan_I, chan_Q = openStored(path)
    channels = np.arange(np.shape(chan_I)[1])
    print("VNA with {:d} channels".format(len(channels)))
    mag = np.zeros((len(channels),len(lo_freqs)))
    chan_freq = np.zeros((len(channels),len(lo_freqs)))

    for chan in channels:        
        mag[chan] = (np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2)) 
        chan_freq[chan] = (lo_freqs/conf.mixer_const+ bb_freqs[chan])/1.0e6 #era lo_freqs/2
    
    # normalization and conversion in dB   
    for chan in channels:
        mag[chan] /= (2**31-1)
        mag[chan] /= ((2**21-1)/512)
        mag[chan] = 20*np.log10(mag[chan])
    
    for chan in channels:
        delta = mag[chan-1][-1]-mag[chan][0]
        mag[chan] += delta
    
    mags = np.hstack(mag)        
    chan_freqs = np.hstack(chan_freq)

    return chan_freqs, mags

def adaptive_iteratively_reweighted_penalized_least_squares_smoothing(data, lam=1.0e6, N_iter=5):
    '''
    lam: adjusting parameter
    N_iter: number of iteration
    '''
    L = len(data)
    D = diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(N_iter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = linalg.spsolve(Z, w*data)
        d_mod = norm((z-data)[z>data])
        if d_mod < 0.001 * norm(data):
            return z
        p = np.exp(i*(data-z)/d_mod)
        w = 0.0*(data < z) + p*(data >= z)
    return z

def main(path, savefile=False):
    print("Searching for KIDs")
    
    bb_freqs = np.load(path + "/bb_freqs.npy")
    lo_freqs = np.load(path + "/sweep_freqs.npy")
    chan_freqs, mags = normalize_and_stack(path, bb_freqs, lo_freqs)
   
    sweep_step = 1.25 # kHz
    smoothing_scale = 2500.0 # kHz
    
    #filtered = lowpass_cosine( mags, sweep_step, 1./smoothing_scale, 0.1 * (1.0/smoothing_scale))
    filtered = adaptive_iteratively_reweighted_penalized_least_squares_smoothing(mags)

    # parametri buoni per MISTRAL 415 v4
    peak_width = (1, 150.0)
    peak_height = (4.5, 30)#5 # era 1.3 sul GP5v2 montato in MISTRAL
    peak_prominence = (2, 30)
    peaks, _ = find_peaks(-(mags-filtered), width=peak_width, prominence=peak_prominence, height=peak_height)
                                       
    target_freqs = chan_freqs[peaks]

    print("Found {:d} KIDs".format(len(peaks)))
    
    np.save(path + '/target_freqs.npy', target_freqs)
    np.savetxt(path + '/target_freqs.dat', target_freqs)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(chan_freqs, mags-filtered, label="VNA sweep, low-passed")
    ax.set_title("VNA sweep and automatic KIDs search")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.plot(chan_freqs[peaks], mags[peaks]-filtered[peaks],"x", label="KIDs")
    ax.legend()
    plt.ion()
    plt.show()
    return target_freqs