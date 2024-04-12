import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from TimeCor import estT
from helpers.callbacks import ChangeStopper, ImprovementStopper
from helpers.losses import frobeniusLoss
# import matplotlib.pyplot as plt

def generateTauWMatrix(TauW, N2):
    TauWMatrix = np.zeros((TauW.shape[0], N2))

    for d in range(TauW.shape[0]):
        TauWMatrix[d, 0:int(TauW[d, 1])] = 1
        TauWMatrix[d, -1:(-int(TauW[d, 0]) - 1):-1] = 1

    return TauWMatrix

class ShiftNMF(torch.nn.Module):
    def __init__(self, X, rank, lr=0.2, alpha=1e-8, patience=10, factor=0.9, min_imp=1e-6):
        super().__init__()

        self.rank = rank
        self.X = torch.tensor(X)
        self.std = torch.std(self.X)
        self.X = self.X / self.std
        
        self.N, self.M = X.shape

        self.softplus = torch.nn.Softplus()
        self.lossfn = frobeniusLoss(torch.fft.fft(self.X))
        
        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.randn(self.N, rank, requires_grad=True, dtype=torch.double)*0)
        self.H = torch.nn.Parameter(torch.randn(rank, self.M, requires_grad=True, dtype=torch.double))
        self.tau = torch.zeros(self.N, self.rank,dtype=torch.double)
        # self.tau_tilde = torch.nn.Parameter(torch.zeros(self.N, self.rank, requires_grad=False))
        # self.tau = lambda: self.tau_tilde

        self.stopper = ChangeStopper(alpha=alpha, patience=patience + 5)
        
        self.optimizer = Adam([self.W, self.H], lr=lr)
        self.optimizer = Adam([self.W, self.H], lr=lr)
        self.improvement_stopper = ImprovementStopper(min_improvement=min_imp)
        
        if factor < 1:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience-2)
        else:
            self.scheduler = None

    def forward(self):
        # Get half of the frequencies
        Nf = self.M // 2 + 1
        # Fourier transform of H along the second dimension
        Hf = torch.fft.fft(self.softplus(self.H), dim=1)[:, :Nf]
        # Keep only the first Nf[1] elements of the Fourier transform of H
        # Construct the shifted Fourier transform of H
        Hf_reverse = torch.flip(Hf[:, 1:Nf-1], dims=[1])
        # Concatenate the original columns with the reversed columns along the second dimension
        Hft = torch.cat((Hf, torch.conj(Hf_reverse)), dim=1)
        f = torch.arange(0, self.M) / self.M
        omega = torch.exp(-1j * 2 * torch.pi * torch.einsum('Nd,M->NdM', self.tau, f))
        Wf = torch.einsum('Nd,NdM->NdM', self.softplus(self.W), omega)
        # Broadcast Wf and H together
        V = torch.einsum('NdM,dM->NM', Wf, Hft)
        return V
    def fit_tau(self):
        X = np.array(self.X.detach().numpy(), dtype=np.complex128)
        W = np.array(self.W.detach().numpy(), dtype=np.complex128)
        H = np.array(self.H.detach().numpy(), dtype=np.complex128)
        T = estT(X,W,H)
        self.tau = torch.tensor(T, dtype=torch.cdouble)

    def fit(self, verbose=False, return_loss=False, max_iter = 15000, tau_iter=0):
        running_loss = []
        self.iters = 0
        self.tau_iter = tau_iter
        while self.iters < max_iter:#not self.stopper.trigger() and self.iters < max_iter and not self.improvement_stopper.trigger():
            self.iters += 1
            # zero optimizer gradient
            self.optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn(output)
            
            loss.backward()

            # Update W and H
            self.optimizer.step()

            if (self.iters%20) == 0:
                self.fit_tau()

            if self.scheduler != None:
                self.scheduler.step(loss)
            
            running_loss.append(loss.item())
            self.stopper.track_loss(loss)
            self.improvement_stopper.track_loss(loss)
            
            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}, Tau: {torch.norm(self.tau)}", end='\r')

        W = self.softplus(self.W).detach().numpy()
        H = (self.softplus(self.H)*self.std).detach().numpy()
        tau = self.tau.detach().numpy()

        output = self.forward()
        self.recon = torch.fft.ifft(output)*self.std

        if return_loss:
            return W, H, tau, running_loss
        else:
            return W, H, tau

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from TimeCor import *
    
    X  = pd.read_csv("X.csv").to_numpy()


    
    alpha = 1e-5
    noc = 3
    nmf = ShiftNMF(X, 3, lr=0.1, alpha = alpha, factor=1, patience=10000)
    W, H, tau = nmf.fit(verbose=1, max_iter=1000)
    plt.plot(H.T)
    plt.show()