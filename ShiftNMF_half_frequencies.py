import torch
import scipy
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper, ImprovementStopper
from helpers.losses import frobeniusLoss, ShiftNMFLoss
import matplotlib.pyplot as plt


class ShiftNMF(torch.nn.Module):
    def __init__(self, X, rank, lr=0.2, alpha=1e-8, patience=10, factor=0.9, min_imp=1e-6):
        super().__init__()

        self.rank = rank
        self.X = torch.tensor(X)
        self.std = torch.std(self.X)
        self.X = self.X / self.std
        self.N, self.M = X.shape

        self.softplus = torch.nn.Softplus()
        self.lossfn = ShiftNMFLoss(torch.fft.fft(self.X))
        
        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.randn(self.N, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.randn(rank, self.M, requires_grad=True))
        self.tau = torch.nn.Parameter(torch.zeros(self.N, self.rank), requires_grad=True)
        # Prøv også med SGD
        self.stopper = ChangeStopper(alpha=alpha, patience=patience)
        self.improvement_stopper = ImprovementStopper(min_improvement=min_imp, patience=patience)
        self.optimizer = Adam(self.parameters(), lr=lr)

        if factor < 1:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience)
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

    def fit(self, verbose=False, return_loss=False, max_iter = 15000):
        running_loss = []
        self.iters = 0
        while not self.stopper.trigger() and self.iters < max_iter and not self.improvement_stopper.trigger():
            self.iters += 1
            # zero optimizer gradient
            self.optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn(output)
            loss.backward()

            # Update W, H and tau
            self.optimizer.step()
            if self.scheduler != None:
                self.scheduler.step(loss)
            running_loss.append(loss.item())
            self.stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        W = self.softplus(self.W).detach().numpy()
        H = self.softplus(self.H).detach().numpy()
        tau = self.tau.detach().numpy()

        output = self.forward()
        self.recon = torch.fft.ifft(output)

        if return_loss:
            return W, H, tau, running_loss
        else:
            return W, H, tau
