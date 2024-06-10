










import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))

data_name = "wine"
model_name = "OPT_NMF"
nr_components= 3
run = 0
lr = 0.1
def plot_loss_path(data_name, model_name, nr_components, run, lr):
    data = np.load("./loss_path/"+str(data_name)+"_"+str(model_name)+"_"+str(lr)+"_"+str(nr_components)+"_"+"lr_test"+"_"+str(run)+".npy")
    plt.plot(data[:2000], label="lr:"+str(lr))

plot_loss_path(data_name, model_name, nr_components, run, lr)
plot_loss_path(data_name, model_name, nr_components, run, 0.01)
plot_loss_path(data_name, model_name, nr_components, run, 1)
plt.legend()
plt.title('Model: '+model_name+' Data:'+data_name)
plt.xlabel('Epochs')
plt.ylabel('Error')

plt.show()