import torch
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper, ImprovementStopper
from helpers.losses import frobeniusLoss
from helpers.initializers import FurthestSum


class torchAA(torch.nn.Module):
    def __init__(self, X, rank, alpha=1e-9, lr = 0.2, factor = 0.9, patience = 5, min_imp = 1e-6):
        super(torchAA, self).__init__()

        # Shape of Matrix for reproduction
        N, M = X.shape
        self.X = torch.tensor(X, dtype=torch.double)

        self.softmax = torch.nn.Softmax(dim=1)
        self.lossfn = frobeniusLoss(self.X)
        Furthest = False
        if Furthest:
            noc = 10
            power = 1
            initial = 0
            exclude = []
            cols = FurthestSum(X.T, noc, initial, exclude)
            self.C = torch.zeros(N, rank)
            for i in cols:
                self.C[i] = power
            self.C = self.C.T
            self.C = self.C.clone().requires_grad_(True)
            self.C = torch.nn.Parameter(self.C)
        else:
            self.C = torch.nn.Parameter(torch.randn(rank, N, requires_grad=True, dtype=torch.double)*3)
            self.S = torch.nn.Parameter(torch.randn(N, rank, requires_grad=True, dtype=torch.double)*3)
        
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.stopper = ChangeStopper(alpha=alpha, patience=patience)
        self.improvement_stopper = ImprovementStopper(min_improvement=min_imp)
        
        
        
        if factor < 1:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience-2)
        else:
            self.scheduler = None

 
    def forward(self):

        # first matrix Multiplication with softmax
        CX = torch.matmul(self.softmax(self.C), self.X)

        # Second matrix multiplication with softmax
        SCX = torch.matmul(self.softmax(self.S), CX)

        return SCX

    def fit(self, verbose=False, return_loss=False):

        # Convergence criteria
        running_loss = []

        while not self.stopper.trigger() and not self.improvement_stopper.trigger():
            # zero optimizer gradient
            self.optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output)
            loss.backward()

            # Update A and B
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
                print(f"Epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        C = self.softmax(self.C)
        S = self.softmax(self.S)

        C = C.detach().numpy()
        S = S.detach().numpy()
        if return_loss:
            return C, S, running_loss
        else:
            return C, S


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io
    mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

    # Get X and Labels. Probably different for the other dataset, but i didn't check :)
    X = mat.get('xData')
    
    AA = torchAA(X, 3)
    C, S = AA.fit(verbose=True)
    CX = np.matmul(C, X)
    SCX = np.matmul(S, CX)
    plt.figure()
    for vec in CX:
        plt.plot(vec)
    plt.title("Archetypes")
    plt.show()
    plt.figure()
    plt.plot(X[1], label="First signal of X")
    plt.plot(SCX[1], label="Reconstructed signal with AA")
    plt.legend()
    plt.show()

    plt.plot(X[2])
    plt.plot(SCX[2])
    plt.show()
    