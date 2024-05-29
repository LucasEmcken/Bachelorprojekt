
import numpy as np

from helpers.generators import *
from Hardmodel_single_peaks import Single_Model
import scipy
from scipy.signal import find_peaks
import itertools


def single_fit(X):
        #find peaks in the sample
        #peaks = find_peaks_cwt(X, widths=[100, 300])
        peaks = find_peaks(X, height=max(X)/20)[0]
        
        sigmas = scipy.signal.peak_widths(X, peaks, wlen=1000)[0]/2 #.355*1.5
        select = [prom>10 for prom in sigmas]
        select = [i for i, x in enumerate(select) if x == False]
        sigmas = np.delete(sigmas, select)
        peaks = np.delete(np.array(peaks), select)
        print("Found peaks:"+str(peaks))
        model = Single_Model(X, peaks, sigmas, lr=5, alpha = 1e-7, factor=1, patience=1, min_imp=0.001) # min_imp=1e-3)
        W, C = model.fit(verbose=True)

        mean = model.means.detach().numpy()
        sigmas = model.sigma.detach().numpy()
        n = model.N.detach().numpy()

        return mean, sigmas, n

def calc_sigma_matrix(nr_runs=10):
    mean, sigmas = single_fit(X)
    sigma_matrix = np.array([sigmas])
    for i in range(nr_runs-1):
        means, sigmas = single_fit(X)
        sigma_matrix = np.append(sigma_matrix, [sigmas], axis=0)
    return sigma_matrix
    
def calc_difference_matrix(sigmas):
    diff_matrix = np.zeros((len(sigmas),len(sigmas)))
    for i in range(len(sigmas)):
        for j in range(len(sigmas)):
            diff_matrix[i,j] = abs(sigmas[i]-sigmas[j])/sigmas[i]
    return diff_matrix

def peak_hypothesis(value_matrix, cutoff= 5/100):
    H = set()
    for i,peaks in enumerate(value_matrix):
        peaks = peaks.tolist()
        valid_peaks = set()
        for peak_index, peak in enumerate(peaks):
            if peak < cutoff:
                valid_peaks.add(peak_index)
        for combination_length in range(1,len(valid_peaks)+1):
            for h in itertools.combinations(valid_peaks, combination_length):
                H.add(tuple(sorted(h)))
    return H