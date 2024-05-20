import scipy.signal
import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper, ImprovementStopper
from helpers.losses import frobeniusLoss, VolLoss
import scipy
import itertools
from scipy.signal import find_peaks, find_peaks_cwt, ricker, cwt


torch.manual_seed(4)

torch.autograd.set_detect_anomaly(True)

class Single_Model(torch.nn.Module):
    def __init__(self, X, init_means, alpha=1e-6, lr=0.1, patience=5, factor=1, min_imp=1e-6):
        super().__init__()

        rank = len(init_means)
        print(scipy.signal.peak_prominences(X,init_means, wlen=2000)[0])
        sigmas = scipy.signal.peak_widths(X, init_means, wlen=1000)[0]/2 #.355*1.5    
        print("initial sigmas:")
        print(sigmas)

        self.X = torch.tensor(X)

        if len(X.shape) == 1:
            self.X = torch.unsqueeze(self.X,dim=0)
        
        n_row, n_col = self.X.shape
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax()
        self.n_row = n_row # nr of samples
        self.n_col = n_col
        self.rank = rank
        

        self.std = torch.std(self.X)
        self.X = self.X/self.std
        
        
        #self.lossfn = frobeniusLoss(torch.tensor(self.X))
        self.lossfn = frobeniusLoss(self.X.clone().detach())
        
        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        #Weights of each component
        # true_W  = torch.tensor(pd.read_csv("W.csv").to_numpy()/self.std, dtype=torch.float)
        # self.W = torch.nn.Parameter(true_W, requires_grad=True)
        self.W = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        # print(torch.mean(self.X, dim=0).shape)
       
        #self.means = torch.nn.Parameter(torch.tensor([(i+1/2)*n_col/rank for i in range(rank+1)], requires_grad=True,dtype=torch.double))
        # self.sigma = torch.nn.Parameter(torch.rand(rank, 1, requires_grad=True,dtype=torch.float32)*200)
        # self.spacing = torch.nn.Parameter((torch.rand(rank, 1, requires_grad=True,dtype=torch.float32)+1)*1000)
        
        self.sigma = torch.nn.Parameter(torch.tensor(sigmas, requires_grad=True,dtype=torch.float32))
        self.N = torch.nn.Parameter(torch.zeros(rank, requires_grad=True,dtype=torch.float32))
        # self.sigma = torch.nn.Parameter(torch.tensor([100, 100, 300, 300,50,50], requires_grad=True,dtype=torch.float32))
        #self.spacing = torch.nn.Parameter(torch.tensor([1000,1000,1000], requires_grad=True,dtype=torch.float32))

        #self.means = torch.nn.Parameter(torch.tensor([1000, 4000, 8000.], requires_grad=True,dtype = torch.float32))
        self.means = torch.tensor(init_means,dtype = torch.float32)
        # self.sigma = torch.nn.Parameter(torch.tensor([100,50,150], requires_grad=True,dtype=torch.double))
        # self.spacing = torch.nn.Parameter(torch.tensor([100,100,100], requires_grad=True,dtype=torch.double))
        # #

        

        self.stopper = ChangeStopper(alpha=alpha, patience=patience)
        self.improvement_stopper = ImprovementStopper(min_improvement=min_imp, patience=patience)
        
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.w_optimizer = Adam([self.W], lr=lr)
        
        self.peak_position_optimizer  = Adam([self.means, self.N], lr=lr)
        self.all_peak_optimizer = Adam([self.means, self.sigma, self.N], lr=lr)
        
        if factor < 1:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience-5)
        else:
            self.scheduler = None



    def lorentzian(self, x, mean, variance):
        #print(variance) nan problem was the variance
        return 1 / (torch.pi * variance * (1 + ((x - mean) / variance) ** 2))
    def gauss(self, x, mean, variance):
        return 1/(variance*(2*torch.pi)**(1/2))*torch.exp(-1/2*((x-mean)/variance)**2)
    def voigt(self, x, mean, variance, n):
        return n*self.lorentzian(x,mean,variance)+(1-n)*self.gauss(x,mean,variance)


    def forward(self):
        time = torch.linspace(0,self.n_col,self.n_col)
        self.C = torch.zeros((self.rank, self.n_col))
        
        for i in range(self.rank):
            self.C[i] += self.voigt(time,
                                    self.means[i],
                                    torch.clamp(self.sigma[i],1), torch.sigmoid(self.N[i]))
            # self.C[i] += self.lorentzian(time,
            #                         self.means[i],
            #                         torch.clamp(self.sigma[i],1))
        
        WC = torch.matmul(self.softplus(self.W), self.C) #self.softplus(self.C))
        return WC
    
    def fit_grad(self, grad, alpha=1e-3, min_improvement=1e-3):
        stopper = ChangeStopper(alpha=1e-3, patience=3)
        improvement_stopper = ImprovementStopper(min_improvement=1e-3, patience=3)

        while not stopper.trigger() and not improvement_stopper.trigger():
            grad.zero_grad()
            output = self.forward()
            loss = self.lossfn.forward(output)
            print(f"Loss: {loss.item()}", end='\r')
            loss.backward()
            grad.step()
            stopper.track_loss(loss)
            improvement_stopper.track_loss(loss)


    def fit(self, verbose=False, return_loss=False):
        running_loss = []
        
        while not self.stopper.trigger() and not self.improvement_stopper.trigger():
            if (self.improvement_stopper.trigger()):
                print(self.improvement_stopper.trigger())
            # # zero optimizer gradient
            #self.optimizer.zero_grad()

            # self.fit_grad(self.peak_position_optimizer)
            self.fit_grad(self.w_optimizer, alpha=1e-5, min_improvement=1e-5)
            self.fit_grad(self.all_peak_optimizer, alpha=1e-5, min_improvement=1e-5)
            self.fit_grad(self.optimizer)
            # # forward
            output = self.forward()

            #loss calc
            loss = self.lossfn.forward(output)
            loss.backward()
            # # Update
            #self.optimizer.step()
    
            if self.scheduler != None:
                self.scheduler.step(loss)

            running_loss.append(loss.item())
            self.stopper.track_loss(loss)
            self.improvement_stopper.track_loss(loss)

            # print loss
            if verbose:
                #print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        W = self.softplus(self.W).detach().numpy()
        C = self.C.detach().numpy()
        print("Lorentzian %:")
        print(torch.sigmoid(self.N))
        print("Sigma:")
        print(self.sigma)

        if return_loss:
            return W, C, running_loss
        else:
            return W, C



if __name__ == "__main__":
    from helpers.callbacks import explained_variance
    #from helpers.data import X_clean
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')
    
    # X = mat.get('xData')
    # targets = mat.get('yData')
    # target_labels = mat.get('yLabels')
    # axis = mat.get("Axis")
    from helpers.data import X_WINE

    X  = pd.read_csv("X_duplet.csv").to_numpy()
    X = X[5]
    
    def single_fit(X):
        

        alpha = 1e-7
        #find peaks in the sample
        peaks, _ = find_peaks(X)
        print("peaks:"+str(peaks))
        model = Single_Model(X, peaks, lr=5, alpha = alpha, factor=1, patience=1, min_imp=0.001) # min_imp=1e-3)
        W, C = model.fit(verbose=True)

        mean = model.means.detach().numpy()
        sigmas = model.sigma.detach().numpy()

        return mean, sigmas

    from scipy import stats

    # Choose two numbers from your list
   
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

    def calculate_t_test(sigma_matrix):
        nr_samples, nr_peaks = sigma_matrix.shape
        # Conduct a t-test
        t_matrix = np.zeros((nr_peaks,nr_peaks))
        p_matrix = np.zeros((nr_peaks,nr_peaks))
        for i in range(nr_peaks):
            for j in range(nr_peaks):
                t_statistic, p_value = stats.ttest_ind(sigma_matrix[:,i], sigma_matrix[:,j])
                t_matrix[i,j] = t_statistic
                t_matrix[j,i] = t_statistic
                p_matrix[i,j] = p_value
                p_matrix[j,i] = p_value
        return t_matrix, p_matrix
    # sigma_matrix = calc_sigma_matrix()
    # t_matrix, p_matrix = calculate_t_test(sigma_matrix)
    # print(f"t-statistic: {t_matrix}")
    # print(f"p-value: {p_matrix}")

    means, sigmas = single_fit(X)
    diff_matrix = calc_difference_matrix(sigmas)

    
    #must be a nr_peaks x nr_peaks matrix with the a measurement of how close the peaks are related.
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

    hypothesis = peak_hypothesis(diff_matrix)
    print(hypothesis)
    from Hardmodel import Hard_Model

    model = Hard_Model(X, hypothesis, means, sigmas, lr=10, alpha = 1e-3, factor=1, patience=1, min_imp=0.01) # min_imp=1e-3)
    W, C = model.fit(verbose=True)
    plt.plot(model.X.detach().numpy())
    print(W)
    for i, vec in enumerate(C):
        plt.plot(vec*W[:,i])
    plt.title("C")
    plt.show()

    plt.plot(model.X.detach().numpy())
    plt.plot(np.matmul(W,C).T)
    plt.show()

