import numpy as np
from scipy.linalg import khatri_rao as krprod
# from krprod import krprod

def reconstructShiftCP(FACT, T):
    """
    Reconstructs the data from FACT obtained from the shiftCP model
    Written by Morten Mørup

    Parameters:
    - FACT: list of factors
    - T: array of delays

    Returns:
    - Rec: reconstructed data
    """
    noc = FACT[0].shape[1]
    nrmodes = len(FACT)
    krprt = np.ones((1,noc))
    
    for k in range(2, nrmodes):
        krprt = krprod(FACT[k],krprt)

    N = [FACT[k].shape[0] for k in range(nrmodes)]
    if nrmodes < 3:
        N.append(1)

    Ns = N[1]
    f = -1j * 2 * np.pi * np.arange(Ns) / Ns
    Nf = len(f) // 2 + 1
    Sf = np.fft.fft(FACT[1].T, axis=1)[:, :Nf]
    f = f[:Nf]

    Rec = np.zeros(N[:3], dtype=np.complex128)

    for i in range(N[0]):
        Sft = Sf * np.exp(np.outer(T[i], f))
        if N[1] % 2 == 0:
            Sft = np.hstack([Sft, np.conj(Sft[:, -2:0:-1])])
        else:
            Sft = np.hstack([Sft, np.conj(Sft[:, -1:0:-1])])
        St = np.real(np.fft.ifft(Sft, axis=1))
        
        Rec[i, ...] = np.dot(St.T, krprod(FACT[0][i, :].reshape(1, -1), krprt).T)


    return Rec