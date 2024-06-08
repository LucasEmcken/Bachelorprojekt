import torch
import scipy
import scipy.io
import sys
import numpy as np

nr_components = int(sys.argv[1])
model_name = sys.argv[2]
data_name = sys.argv[3]


if data_name == "alko":
    from helpers.data import X_ALKO
    X = X_ALKO
if data_name == "art":
    from helpers.data import X_ART
    X = X_ART
if data_name == "wine":
    from helpers.data import X_WINE
    X = X_WINE


print("starting")
print(model_name)
print(data_name)

lrs = [0.1, 0.01, 1]

nr_tests = 10

alpha = 1e-5
min_imp = 0.0001

for i, lr in enumerate(lrs):
    print("learning rate:" + str(lr))
    losses = np.zeros((nr_tests))
    for it in range(nr_tests):
        print("iteration: "+str(it)+" out of "+str(nr_tests))
        if model_name == "DISC_NMF":
            from shiftNMFDiscTau import ShiftNMF
            model = ShiftNMF(X, nr_components, lr=lr, alpha = alpha, factor=1, patience=30, min_imp=min_imp)
        if model_name == "OPT_NMF":
            from shiftNMF_frozen import ShiftNMF
            model = ShiftNMF(X, nr_components, lr=lr, alpha = alpha, factor=1, patience=30, min_imp=min_imp)
        if model_name == "GRAD_NMF":
            from ShiftNMF_half_frequencies import ShiftNMF
            model = ShiftNMF(X, nr_components, lr=lr, alpha = alpha, factor=1, patience=30, min_imp=min_imp)
        returns = model.fit(verbose=True, return_loss=True, max_iter=5000)
        loss = returns[-1]
        losses[it] = loss[-1]
        np.save("./loss_path/"+str(data_name)+"_"+str(model_name)+"_"+str(lr)+"_"+str(nr_components)+"_"+"lr_test"+"_"+str(it),loss)
    np.save("./losses/"+str(data_name)+"_"+str(model_name)+"_"+str(lr)+"_"+str(nr_components)+"_"+"lr_test",losses)


print("DONE")
    # plt.ylabel("average loss")
    # plt.xlabel("Learning rate")
    # plt.plot([str(lr) for lr in lrs], np.mean(losses,axis=1).flatten())
    # plt.suptitle('Categorical Plotting')
    # plt.savefig("lr_test_"+str(model_name)+"_"+str(data_name)+"_"+str(comp_nr))




