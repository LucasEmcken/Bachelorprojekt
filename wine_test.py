import scipy
import matplotlib.pyplot as plt
import numpy as np
mat = scipy.io.loadmat('helpers/data/NMR_40wines.mat')
#Get X and Labels. Probably different for the other dataset, but i didn't check :)
X = mat.get('X')
Y = mat.get('Y')
#ppm is the scale of the x-axis.
ppm = mat.get('Label')

labels = mat.get('Label')
#40 wines times 8712 length spectrum
N, M = X.shape
#try to uncover mixings
label = [x[0] for x in labels[0]]

plt.figure(figsize=(15,8))
plt.plot(X[5:15].T)
plt.show()



