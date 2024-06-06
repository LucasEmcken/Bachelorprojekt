import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
# from TimeCor import estT
from estTimeAutCor import estT
from helpers.callbacks import ChangeStopper, ImprovementStopper
from helpers.losses import frobeniusLoss
from helpers.initializers import PCA_init
# import matplotlib.pyplot as plt

def generateTauWMatrix(TauW, N2):
    TauWMatrix = np.zeros((TauW.shape[0], N2))

    for d in range(TauW.shape[0]):
        TauWMatrix[d, 0:int(TauW[d, 1])] = 1
        TauWMatrix[d, -1:(-int(TauW[d, 0]) - 1):-1] = 1

    return TauWMatrix
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ShiftNMF(torch.nn.Module):
    def __init__(self, X, rank, lr=0.2, alpha=1e-8, patience=10, factor=0.5, min_imp=1e-6, Lambda=0):
        super().__init__()

        self.rank = rank
        self.X = torch.tensor(X)
        self.std = torch.std(self.X)
        self.X = self.X / self.std
        
        self.N, self.M = X.shape

        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=1)
        self.lossfn = frobeniusLoss(self.X)
        self.Lambda = Lambda
        
        # Initialization of Tensors/Matrices NxR and RxM
        self.W = torch.nn.Parameter(torch.rand(self.N, rank, requires_grad=True, dtype=torch.double)*torch.max(self.X))
        self.H = torch.nn.Parameter(torch.randn(rank, self.M, requires_grad=True, dtype=torch.double)*0.1)
        self.tau = torch.zeros(self.N, self.rank, dtype=torch.double)

        self.optimizer = Adam([self.W, self.H], lr=lr)
        
        if factor < 1:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience-2)
        else:
            self.scheduler = None

    def forward(self):
        WH = torch.zeros_like(self.X)

        for i in range(self.N):
            H_rolled = torch.zeros_like(self.H)
            for j in range(self.rank):
                H_rolled[j] = torch.roll(self.H[j], shifts=int(self.tau[i, j]))
            WH[i] = torch.matmul(self.softplus(self.W[i]),
                                 self.softmax(H_rolled))
        
        return WH
    
    def fit_tau(self):
        X = np.array(self.X.detach().numpy(), dtype=np.double)
        H = np.array(self.softmax(self.H).detach().numpy(), dtype=np.double)
        tau = np.array(self.tau.detach().numpy(), dtype=np.double)
        W = np.array(self.W.detach().numpy(), dtype=np.double)

        T, A = estT(X, W, H, tau)

        self.tau = torch.tensor(T, dtype=torch.double)
        # self.W.data = torch.tensor(A, dtype=torch.double)

    def fit(self, verbose=False, return_loss=False, max_iter=15000, tau_iter=100):
        running_loss = []
        self.iters = 0
        self.tau_iter = tau_iter

        while self.iters < max_iter:
            self.iters += 1

            self.optimizer.zero_grad()
            output = self.forward()
            loss = self.lossfn(output) + self.Lambda * torch.sum(self.softplus(self.W))
            loss.backward()
            self.optimizer.step()

            if (self.iters % tau_iter) == 0:
                self.fit_tau()

            if self.scheduler is not None:
                self.scheduler.step(loss)
            
            running_loss.append(loss.item())

            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}, Tau: {torch.norm(self.tau)}", end='\r')
        
        if verbose:
            print(f"epoch: {len(running_loss)}, Loss: {loss.item()}, Tau: {torch.norm(self.tau)}")

        W = self.softplus(self.W).detach().numpy()
        H = (self.softmax(self.H) * self.std).detach().numpy()
        tau = self.tau.detach().numpy()

        output = self.forward()
        self.recon = torch.fft.ifft(output) * self.std

        if return_loss:
            return W, H, tau, running_loss
        else:
            return W, H, tau

# You may need to define the frobeniusLoss and estT functions as per your requirement


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    # from TimeCor import *
    
    def shift_dataset(W, H, tau):
        # Get half the frequencies
        Nf = H.shape[1] // 2 + 1
        # Fourier transform of S along the second dimension
        Hf = np.fft.fft(H, axis=1)
        # Keep only the first Nf[1] elements of the Fourier transform of S
        Hf = Hf[:, :Nf]
        # Construct the shifted Fourier transform of S
        Hf_reverse = np.fliplr(Hf[:, 1:Nf - 1])
        # Concatenate the original columns with the reversed columns along the second dimension
        Hft = np.concatenate((Hf, np.conj(Hf_reverse)), axis=1)
        f = np.arange(0, M) / M
        omega = np.exp(-1j * 2 * np.pi * np.einsum('Nd,M->NdM', tau, f))
        Wf = np.einsum('Nd,NdM->NdM', W, omega)
        # Broadcast Wf and H together
        Vf = np.einsum('NdM,dM->NM', Wf, Hft)
        V = np.fft.ifft(Vf)
        return V.real

    N, M, d = 5, 10000, 3
    Fs = 1000  # The sampling frequency we use for the simulation
    t0 = 10    # The half-time interval we look at
    t = np.arange(-t0, t0, 1/Fs)  # the time samples
    f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples

    def gauss(mu, s, time):
        return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)

    W = np.random.dirichlet(np.ones(d), N)

    shift = 200
    tau = np.random.randint(-shift, shift, size=(N, d))
    tau = np.array(tau, dtype=np.int32)

    mean = [1500, 5000, 8500]
    std = [30, 40, 50]
    t = np.arange(0, 10000, 1)

    H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])

    X = shift_dataset(W, H, tau)
        
    alpha = 1e-5
    noc = 3
    nmf = ShiftNMF(X, 3, lr=1, alpha=1e-6, patience=1000, min_imp=0, Lambda = 0)
    W_est, H_est, tau_est = nmf.fit(verbose=1, max_iter=750, tau_iter=25)
    
    fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(tau)
    # ax[1].imshow(tau_est.real)
    
    ax[0].imshow(W)
    ax[1].imshow(W_est)
    
    plt.show()
    
    fig, ax = plt.subplots(2, 1)
    
    ax[0].plot(np.matmul(W, H).T)
    ax[1].plot(np.matmul(W_est, H_est).T)
    
    plt.show()