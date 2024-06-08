import numpy as np
import matplotlib.pyplot as plt

def plot_error(model_name = "OPT_NMF",data_name = "art", lr = "0.1"):
	start_index = 1
	data = np.load("./losses/"+str(data_name)+"_"+str(model_name)+"_"+str(lr)+"_"+str(start_index)+"_"+"lr_test.npy")
	data = np.expand_dims(data,0)
	
	for nr_components in range(start_index+1,4):
		data = np.append(data,np.expand_dims(np.load("./losses/"+str(data_name)+"_"+str(model_name)+"_"+str(lr)+"_"+str(nr_components)+"_"+"lr_test.npy"),0), axis=0)
	x = np.arange(1,len(data)+1)
	y = np.mean(data, axis=1)
	yerr = np.std(data, axis=1)


	plt.errorbar(x, y, yerr=yerr, fmt='o', label="model: "+ model_name + " lr: "+lr)
	plt.xticks(x)
	plt.legend()
	plt.title('Error Bar Plot. DATA:'+data_name)
	plt.xlabel('Nr of components')
	plt.ylabel('Error')
plt.figure(figsize=(15,8))
# plot_error(model_name="OPT_NMF", data_name="wine", lr="0.01")
plot_error(model_name="GRAD_NMF", data_name="art", lr="0.1")
plot_error(model_name="GRAD_NMF", data_name="art", lr="0.01")
plot_error(model_name="GRAD_NMF", data_name="art", lr="1")
# plot_error(model_name="OPT_NMF", data_name="wine", lr="1")

# plot_error(model_name="DISC_NMF", data_name="art", lr="0.01")
#plot_error(model_name="DISC_NMF", data_name="art", lr="0.1")

# #plot lower error bound from noise
# from helpers.data import NOISE_ART, X_ART
# num = np.linalg.norm(NOISE_ART, ord='fro')**2
# dem = np.linalg.norm(X_ART)

# plt.axhline(y=num/dem, color='r', linestyle='-', label="noise baseline")
# plt.legend()
plt.show()