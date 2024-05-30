import numpy as np
import matplotlib.pyplot as plt
data_name = "alko"
model_name = "OPT_NMF"
def plot_error(model_name = "OPT_NMF",data_name = "wine", lr = "0.1"):
	data = np.load("./losses/"+str(data_name)+"_"+str(model_name)+"_"+str(lr)+"_1_"+"lr_test.npy")
	data = np.expand_dims(data,0)

	for nr_components in range(2,11):
		data = np.append(data,np.expand_dims(np.load("./losses/"+str(data_name)+"_"+str(model_name)+"_"+str(lr)+"_"+str(nr_components)+"_"+"lr_test.npy"),0), axis=0)
	x = np.arange(1,len(data)+1)
	y = np.mean(data, axis=1)
	yerr = np.std(data, axis=1)


	plt.errorbar(x, y, yerr=yerr, fmt='o', label="model: "+ model_name + " lr: "+lr)
	plt.legend()
	plt.title('Error Bar Plot. DATA:'+data_name)
	plt.xlabel('Nr of components')
	plt.ylabel('Error')


plot_error(model_name="OPT_NMF", data_name="art", lr="0.1")
plot_error(model_name="DISC_NMF", data_name="art", lr="0.1")

#plot lower error bound from noise
from helpers.data import NOISE_ART, X_ART
num = np.linalg.norm(NOISE_ART, ord='fro')**2
dem = np.linalg.norm(X_ART)

plt.axhline(y=num/dem, color='r', linestyle='-')

plt.show()