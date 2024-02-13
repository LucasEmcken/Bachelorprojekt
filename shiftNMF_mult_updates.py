
#start by making NMF withou any shift

import numpy as np
import pandas as pd
import torch
from shiftNMF_frozen import ShiftNMF

import numpy as np

def update_T(T, P, TauW):
    nyT = P['nyT']
    Sf = P['Sf']
    A = P['A']
    sizeX2 = P['sizeX2']
    Xf = P['Xf']
    f = P['f']
    w = P['w']
    Recfd = np.zeros((A.shape[0], Sf.shape[1], A.shape[1]), dtype=np.complex128)

    for d in range(1, A.shape[1] + 1):
        Recfd[:, :, d - 1] = (np.tile(A[:, d - 1], (1, len(f))) * np.exp(T[:, d - 1] * f)) * np.tile(Sf[d - 1, :], (A.shape[0], 1))

    Recf = np.sum(Recfd, axis=2)
    Q = Recfd * np.tile(np.conj(Xf - Recf)[:, :, np.newaxis], (1, 1, Recfd.shape[2]))
    grad = np.squeeze(np.sum(np.tile((w * f), (Q.shape[0], 1, Q.shape[2])) * (np.conj(Q) - Q), axis=1)).T
    ind1 = np.where(w == 2)[0]
    ind2 = np.where(w == 1)[0]
    cost_old = np.linalg.norm(Xf[:, ind1] - Recf[:, ind1], 'fro') ** 2
    cost_old += 0.5 * np.linalg.norm(Xf[:, ind2] - Recf[:, ind2], 'fro') ** 2
    keepgoing = True
    Told = T.copy()

    while keepgoing:
        T = Told - nyT * grad
        for d in range(len(T)):
            if T[d] < TauW[d, 0]:
                T[d] = TauW[d, 0]
            if T[d] > TauW[d, 1]:
                T[d] = TauW[d, 1]

        for d in range(1, A.shape[1] + 1):
            Recfd[:, :, d - 1] = (np.tile(A[:, d - 1], (1, len(f))) * np.exp(T[:, d - 1] * f)) * np.tile(Sf[d - 1, :], (A.shape[0], 1))

        Recf = np.sum(Recfd, axis=2)
        cost = np.linalg.norm(Xf[:, ind1] - Recf[:, ind1], 'fro') ** 2
        cost += 0.5 * np.linalg.norm(Xf[:, ind2] - Recf[:, ind2], 'fro') ** 2

        if cost <= cost_old:
            keepgoing = False
            nyT *= 1.2
        else:
            keepgoing = True
            nyT /= 2

    T = np.mod(T, sizeX2)
    ind = np.where(T > np.floor(sizeX2 / 2))[0]
    T[ind] -= sizeX2

    return T, nyT, cost


def matricizing(X, mode):
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)

def krprod(A, B):
    return np.einsum('ik,jk->ijk', A, B).reshape(A.shape[0] * B.shape[0], -1)

def mult_update():
    pass




if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Load data
    X  = pd.read_csv("X.csv").to_numpy()