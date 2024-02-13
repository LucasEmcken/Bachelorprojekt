
#start by making NMF withou any shift

import numpy as np
import pandas as pd
import torch
from torchNMF import NMF

class shiftNMF(NMF):
    def __init__(self, X, rank, alpha=0.000001, lr=0.1, patience=5, factor=0.9, min_imp=0.000001):
        super().__init__(X, rank, alpha, lr, patience, factor, min_imp)
    
    def forward(self):
        WH = torch.matmul(self.softplus(self.W), self.softplus(self.H))
        return WH
    

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Load data
    X  = pd.read_csv("X.csv").to_numpy()