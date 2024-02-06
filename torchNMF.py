import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper, ImprovementStopper
from helpers.losses import frobeniusLoss, VolLoss
import scipy


class NMF(torch.nn.Module):
    def __init__(self, X, rank, alpha=1e-6, lr=0.1, patience=5, factor=0.9, min_imp=1e-6):
        super().__init__()

        n_row, n_col = X.shape
        self.softplus = torch.nn.Softplus()
        
        self.X = torch.tensor(X)
        self.std = torch.std(self.X)
        self.X = self.X/self.std
        
        self.lossfn = frobeniusLoss(torch.tensor(self.X))

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.randn(n_row, rank, requires_grad=True))
        # print(torch.mean(self.X, dim=0).shape)
        self.H = torch.nn.Parameter(torch.randn(rank, n_col, requires_grad=True))
        # print(torch.mean(self.X, dim=0).shape)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.stopper = ChangeStopper(alpha=alpha, patience=patience+5)
        self.improvement_stopper = ImprovementStopper(min_improvement=min_imp)
        
        
        if factor < 1:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience-2)
        else:
            self.scheduler = None

    def forward(self):
        WH = torch.matmul(self.softplus(self.W), self.softplus(self.H))
        return WH

    def fit(self, verbose=False, return_loss=False):
        running_loss = []
        while not self.stopper.trigger() and not self.improvement_stopper.trigger():
            # zero optimizer gradient
            self.optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward

            loss = self.lossfn.forward(output)
            loss.backward()

            # Update W and H
            self.optimizer.step()
            if self.scheduler != None:
                self.scheduler.step(loss)

            running_loss.append(loss.item())
            self.stopper.track_loss(loss)
            self.improvement_stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        W = self.softplus(self.W).detach().numpy()
        H = (self.softplus(self.H)*self.std).detach().numpy()

        if return_loss:
            return W, H, running_loss
        else:
            return W, H


class MVR_NMF(torch.nn.Module):
    def __init__(self, X, rank, regularization=1e-45, normalization=2, lr=0.1, alpha=1e-9, patience=5, factor=0.9):
        super().__init__()

        n_row, n_col = X.shape
        self.normalization = normalization
        self.rank = rank
        self.softmax = torch.nn.Softmax(dim=1) # dim = 1 is on the rows
        self.lossfn = VolLoss(torch.tensor(X), regularization)
        self.softplus = torch.nn.Softplus()

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.randn(n_row, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.randn(rank, n_col, requires_grad=True))

        self.optim = Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=factor, patience=patience)
        self.stopper = ChangeStopper(alpha=alpha, patience=patience+5)

    def normalize(self):


        if self.normalization == 2:
            W = F.normalize(self.softplus(self.W), p=1, dim=1)
        elif self.normalization == 1:
            W = self.softmax(self.W)
        else:
            raise ValueError(f"{self.normalization} is not a currently supported normalization technique (must be 1 or 2)")

        return W

    def forward(self):

        W = self.normalize()
        WH = torch.matmul(W, self.softplus(self.H))

        return WH


    def fit(self, verbose=False, return_loss=False):
        running_loss = []
        while not self.stopper.trigger():
            # zero optimizer gradient
            self.optim.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output, self.softplus(self.H))
            loss.backward()

            # Update W and H
            self.optim.step()
            self.scheduler.step(loss)

            running_loss.append(loss.item())
            self.stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        W = self.normalize().detach().numpy()
        H = self.softplus(self.H).detach().numpy()
        
        if return_loss:
            return W, H, running_loss
        else:
            return W, H


if __name__ == "__main__":
    from helpers.callbacks import explained_variance
    from helpers.data import X_clean
    import matplotlib.pyplot as plt
    import numpy as np

    # mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')
    
    # X = mat.get('xData')
    # targets = mat.get('yData')
    # target_labels = mat.get('yLabels')
    # axis = mat.get("Axis")
    X = X_clean
    alpha = 1e-5
    nmf = NMF(X, 6, lr=1, alpha = alpha, factor=1, patience=10)
    W, H = nmf.fit(verbose=True)
    print(f"Explained variance MVR_NMF: {explained_variance(X, np.matmul(W, H))}")
    plt.figure()
    for vec in H:
        plt.plot(vec)

    plt.title("H")
    plt.show()