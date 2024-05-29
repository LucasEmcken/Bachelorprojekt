import torch
from torch.optim import Adam, lr_scheduler, SGD
from helpers.callbacks import ChangeStopper, ImprovementStopper
from helpers.losses import frobeniusLoss
from helpers.losses import ShiftNMFLoss


class torchShiftAADisc(torch.nn.Module):
    def __init__(self, X, rank, alpha=1e-9, lr = 10, factor = 0.9, patience = 5, min_imp = 1e-6):
        super(torchShiftAADisc, self).__init__()

        # Shape of Matrix for reproduction
        N, M = X.shape
        self.N, self.M = N, M
        self.X = torch.tensor(X)

        # softmax layer
        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus()


        # self.lossfn = frobeniusLoss(torch.fft.fft(self.X))
        # Should be the same as NMF?
        self.lossfn = frobeniusLoss(torch.fft.fft(self.X))

        # Initialization of Tensors/Matrices S and C with size Col x Rank and Rank x Col
        # DxN (C) * NxM (X) =  DxM (A)
        # NxD (S) *  DxM (A) = NxM (SA)
        self.C_tilde = torch.nn.Parameter(torch.randn(rank, N, requires_grad=True,dtype=torch.double))
        self.S_tilde = torch.nn.Parameter(torch.randn(N, rank, requires_grad=True, dtype=torch.double))
        
        self.tau_tilde = torch.nn.Parameter(torch.zeros(N, rank, requires_grad=False, dtype=torch.double))

        #Parameter for the archetypical analysis
        self.C = lambda:self.softmax(self.C_tilde).type(torch.cdouble)
        self.S = lambda:self.softmax(self.S_tilde).type(torch.cdouble)
        
        #Parameter for the shift
        self.tau = lambda: self.tau_tilde
        
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.stopper = ChangeStopper(alpha=alpha, patience=patience)
        self.improvement_stopper = ImprovementStopper(min_improvement=min_imp)
        
        if factor < 1:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience-2)
        else:
            self.scheduler = None

    def forward(self):
        # Implementation of shift AA.
        f = torch.arange(0, self.M) / self.M
        # first matrix Multiplication
        omega = torch.exp(-1j * 2 * torch.pi*torch.einsum('Nd,M->NdM', self.tau(), f))
        # omega_neg = torch.exp(-1j*2 * torch.pi*torch.einsum('Nd,M->NdM', self.tau()*(-1), f))
        omega_neg = torch.conj(omega)

        #data to frequency domain
        X_F = torch.fft.fft(self.X)

        #Aligned data (per component)
        X_F_align = torch.einsum('NM,NdM->NdM',X_F, omega_neg)
        #X_align = torch.fft.ifft(X_F_align)
        #The A matrix, (d,M) A, in frequency domain
        #self.A = torch.einsum('dN,NdM->dM', self.C(), X_align)
        #A_F = torch.fft.fft(self.A)
        self.A_F = torch.einsum('dN,NdM->dM',self.C(), X_F_align)
        #S_F = torch.einsum('Nd,NdM->NdM', self.S().double(), omega)

        # archetypes back shifted
        #A_shift = torch.einsum('dM,NdM->NdM', self.A_F.double(), omega.double())
        self.S_shift = torch.einsum('Nd,NdM->NdM', self.S(), omega) 

        # Reconstruction
        x = torch.einsum('NdM,dM->NM', self.S_shift, self.A_F)
        
        return x

    def fit(self, verbose=False, return_loss=False, max_iter=15000, tau_iter=0, tau_thres = 1e-5):
        self.stopper.reset()
        # Convergence criteria
        running_loss = []
        self.iters = 0
        self.tau_iter = tau_iter
        while not self.stopper.trigger() and self.iters < max_iter and not self.improvement_stopper.trigger():
            self.iters += 1
            # zero optimizer gradient
            self.optimizer.zero_grad()

            # forward
            output = self.forward()
            
            # backward
            loss = self.lossfn.forward(output)
            loss.backward()

            #update tau
            change = torch.sign(self.tau_tilde.grad)
            grad = self.tau_tilde.grad
            #set gradient 0, such that the tau is not updated by the optimizer
            self.tau_tilde.grad = self.tau_tilde.grad * 0
            if self.iters > tau_iter:
                #update change such that only the gradients with a magnitude larger than tau_thres are updated
                change = (torch.abs(grad) > tau_thres) * change
                #update tau
                self.tau_tilde = torch.nn.Parameter(self.tau_tilde + change)

            
            # Update parameters
            self.optimizer.step()
            if self.scheduler != None:
                self.scheduler.step(loss)
            
            # append loss for graphing
            running_loss.append(loss.item())

            # count with early stopping
            self.stopper.track_loss(loss)
            self.improvement_stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}, Tau: {torch.norm(self.tau_tilde)}", end="\r")
        
        C = self.softmax(self.C_tilde)
        S = self.softmax(self.S_tilde)
        tau = self.tau().detach().numpy() 

        C = C.detach().numpy()
        S = S.detach().numpy()
        
        output = self.forward()
        self.recon = torch.fft.ifft(output)
        if return_loss:
            return C, S, tau, running_loss
        else:
            return C, S, tau


if __name__ == "__main__":
    exit()
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io
    mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

    #Get X and Labels. Probably different for the other dataset, but i didn't check :)
    X = mat.get('xData')
    X = X[:10]
    N, M = X.shape
    rank = 3
    D = rank
    AA = torchShiftAADisc(X, rank, lr=0.3, fs_init=False)
    print("test")
    C,S, tau = AA.fit(verbose=True, max_iter=100, tau_thres=1e-3)

    print("tau: ", tau)
    
    recon = AA.recon.detach().resolve_conj().numpy()
    A = torch.fft.ifft(AA.A_F).detach().numpy()

    plt.figure()
    for arc in A:
        plt.plot(arc)
    plt.title("Archetypes")
    plt.show()
    plt.figure()
    plt.plot(X[1], label="First signal of X")
    plt.plot(recon[1], label="Reconstructed signal with shift AA")
    plt.legend()
    plt.show()

    plt.figure()
    plt.imshow(tau, aspect='auto', interpolation="none")
    ax = plt.gca()
    ax.set_xticks(np.arange(0, D, 1))
    plt.colorbar()
    plt.title("Tau")
    plt.show()
    #print(tau)