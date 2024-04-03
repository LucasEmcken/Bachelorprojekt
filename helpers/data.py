import scipy.io
import numpy as np
# import matplotlib.pyplot as plt

#load data from .MAT file
mat = scipy.io.loadmat('helpers/data/nmrdata.mat')

#Get X and Labels. Probably different for the other dataset, but i didn't check :)
mat = mat.get('nmrdata')
X = mat[0][0][0]
labels = mat[0][0][1]

#Store mean and std for inversing the normalization
mu_Y = np.mean(X)
std_U = np.std(X)

#functions for normalizing, and inversing the normalization of data
def normalize_data(target):
    # return (target - np.mean(target))/np.std(target)
    # don't subtract mean, resulting values would be negative
    # and not reproducible by a positive matrix
    return target/np.std(target)

def inv_normalize_data(target, std):
    # return target * std + mean
    #same as above
    return target * std


#Clean data

mat = scipy.io.loadmat('helpers/data/nmrdata_Oil_group3.mat')

#Get X and Labels. Probably different for the other dataset, but i didn't check :)
mat = mat.get('nmrdata_Oil_group3')
X_clean = mat[0][0][0]
labels = mat[0][0][1]

#Store mean and std for inversing the normalization
mu_Y = np.mean(X)
std_U = np.std(X)

#functions for normalizing, and inversing the normalization of data
def normalize_data(target):
    # return (target - np.mean(target))/np.std(target)
    # don't subtract mean, resulting values would be negative
    # and not reproducible by a positive matrix
    return target/np.std(target)

def inv_normalize_data(target, std):
    # return target * std + mean
    #same as above
    return target * std
