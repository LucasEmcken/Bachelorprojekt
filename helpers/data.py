import scipy.io
import numpy as np



# import matplotlib.pyplot as plt
#load data from .MAT file
mat = scipy.io.loadmat('helpers/data/nmrdata.mat')
#Get X and Labels. Probably different for the other dataset, but i didn't check :)
mat = mat.get('nmrdata')
X_URINE = mat[0][0][0]
labels_URINE = mat[0][0][1]


#OIL DATA
mat = scipy.io.loadmat('helpers/data/nmrdata_Oil_group3.mat')
#Get X and Labels. Probably different for the other dataset, but i didn't check :)
mat = mat.get('nmrdata_Oil_group3')
X_OIL = mat[0][0][0]
OIL_labels = mat[0][0][1]

#WINE DATA
mat = scipy.io.loadmat('helpers/data/NMR_40wines.mat')
#40 wines times 8712 length spectrum
X_WINE = mat.get('X')
WINE_PARAMETERS = mat.get('Y')
#ppm is the scale of the x-axis.
# ppm = mat.get('ppm')

labels = mat.get('Label')
# #try to uncover mixings
WINE_labels = [x[0] for x in labels[0]]

# load data from .MAT file
mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')
# Get X and Labels. Probably different for the other dataset, but i didn't check :)
X_ALKO = mat.get('xData')
Y_ALKO = mat.get('yData')
ALKO_labels = mat.get('yLabels')
axis = mat.get("Axis")



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
