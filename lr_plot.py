import numpy as np
import matplotlib.pyplot as plt
data_name = "alko"
model_name = "OPT_NMF"
def plot_error(model_name = "OPT_NMF",data_name = "wine", lr_index = 1):
	x = np.load("./losses/"+str(data_name)+"_"+str(model_name)+"_1_"+"lr_test.npy")
	x = np.expand_dims(x,0)

	for nr_components in range(2,11):
		x = np.append(x,np.expand_dims(np.load("./losses/"+str(data_name)+"_"+str(model_name)+"_"+str(nr_components)+"_"+"lr_test.npy"),0), axis=0)

	x = x[:,lr_index,:]
	print(x.shape)
	data = x
	x = np.arange(1,len(data)+1)
	y = np.mean(data, axis=1)
	yerr = np.std(data, axis=1)

	plt.errorbar(x, y, yerr=yerr, fmt='o', label=model_name)
	plt.legend()
	plt.title('Error Bar Plot. DATA:'+data_name)
	plt.xlabel('Nr of components')
	plt.ylabel('Error')


plot_error(lr_index = 0, data_name="wine")
plot_error(model_name="DISC_NMF", data_name="wine", lr_index = 0)
plt.show()