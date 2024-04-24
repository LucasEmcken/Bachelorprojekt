import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
# from TimeCor import estT
from estTimeAutCor import estT
from helpers.callbacks import ChangeStopper, ImprovementStopper
from helpers.losses import frobeniusLoss
from torchrl.modules.utils import inv_softplus
from helpers.initializers import PCA_init
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
        # self.inv_softplus = inv_softplus(bias=1)
        self.lossfn = frobeniusLoss(torch.fft.fft(self.X))
        
        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.randn(self.N, rank, requires_grad=True, dtype=torch.double)*0)
        #self.H = torch.nn.Parameter(torch.randn(rank, self.M, requires_grad=True, dtype=torch.double)*np.std(X))
        self.H = torch.nn.Parameter(torch.tensor(PCA_init(X.T, rank).T, dtype=torch.double))
        self.H = torch.nn.Parameter(inv_softplus(self.H))
#        plt.plot(self.H.detach().numpy().T)
#        plt.show()

    #    print(self.H.shape)
        #exit()
        self.tau = torch.zeros(self.N, self.rank,dtype=torch.double)
        # self.tau_tilde = torch.nn.Parameter(torch.zeros(self.N, self.rank, requires_grad=False))
        # self.tau = lambda: self.tau_tilde

        self.fit_tau()
        
        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(torch.fft.ifft(self.forward()).detach().numpy().T)
        # ax[1].plot(X.T)
        # plt.show()
        # exit()
        
        
        self.stopper = ChangeStopper(alpha=alpha, patience=patience + 5)
        
        self.optimizer = Adam([self.H], lr=lr)
        #self.optimizer = Adam([self.W], lr=lr)
        # self.optimizer = Adam([self.W, self.H], lr=lr)
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
        # Wf = torch.einsum('Nd,NdM->NdM', self.W, omega)
        # Broadcast Wf and H together
        V = torch.einsum('NdM,dM->NM', Wf, Hft)
        return V
    
    def fit_tau(self):
        X = np.array(self.X.detach().numpy(), dtype=np.complex128)
        
        W = np.array(self.softplus(self.W).detach().numpy(), dtype=np.complex128)
        H = np.array(self.softplus(self.H).detach().numpy(), dtype=np.complex128)
        
        T = estT(X,W,H)
        W = inv_softplus(W.real)
        
        self.tau = torch.tensor(T, dtype=torch.cdouble)
        self.W = torch.nn.Parameter(W)
        # self.W = torch.nn.Parameter(torch.tensor(W,  dtype=torch.double))

    def fit(self, verbose=False, return_loss=False, max_iter = 15000, tau_iter=100):
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

            if (self.iters%20) == 0 and self.iters > tau_iter:
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
    # from TimeCor import *
    
    X  = pd.read_csv("X_duplet.csv").to_numpy()

    X = np.pad(X, ((0, 0), (1000, 1000)), 'edge')
    
    # plt.plot(X.T)
    # plt.show()
    # exit()
    
    alpha = 1e-5
    noc = 3
    nmf = ShiftNMF(X, 3, lr=0.05, alpha = alpha, factor=1, patience=10000)
    W, H, tau = nmf.fit(verbose=1, max_iter=250, tau_iter=0)
    print("")
    
    
    plt.plot(H.T)
    plt.show()
    
    #fig, ax = plt.subplots(1, 2)
    #ax[0].plot(H.T)
    #
    #ax[1].plot(inv_softplus(H).T)
    #
    #plt.show()
    
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(X.T)
    # ax[1].plot(nmf.recon.detach().numpy().T)
    # plt.show()
    
    # plt.imshow(tau.real)
    # plt.imshow(W)
    # plt.show()