import numpy as np

def estTimeAutCor(Xf, A, Sf, krSf, krf, T, Nf, N, w, TauW, Lambda):
    """

    Args:
    Xf: 2D NumPy array representing the Fourier-transformed data of shape (number_of_samples, frequency_bins).
    It contains the Fourier-transformed data.

    A: 2D NumPy array representing the factor loadings of shape (number_of_samples, number_of_components).
    It contains the factor loadings.

    Sf: 2D NumPy array representing the Fourier-transformed shift component of shape (number_of_components, frequency_bins).
    It contains the Fourier-transformed shift component.

    krSf: 2D NumPy array representing the complex conjugate of the Fourier-transformed shift component of shape (number_of_components, frequency_bins).

    krf: 1D NumPy array representing the frequency values.

    T: 2D NumPy array representing the estimated time delays of shape (number_of_samples, number_of_components).

    Nf: 1D NumPy array representing the size of each dimension of the Fourier-transformed data.

    N: 1D NumPy array representing the size of each dimension of the original data.

    w: 1D NumPy array representing the windowing function.

    TauW: 2D NumPy array representing the shift constraints of the extracted components of shape (number_of_components, 2).

    Lambda: 1D NumPy array representing the regularization strength for each component.
    """
    #reshape to (1, number_of_components)
    Xf = np.expand_dims(Xf, axis=0)
    A = np.expand_dims(A, axis=0)
    T = np.expand_dims(T, axis=0)
    noc = A.shape[1]
    # noc = 3
    if N[1] % 2 == 0:
        sSf = 2 * Nf[1] - 2
    else:
        sSf = 2 * Nf[1] - 1
    t1 = np.random.permutation(A.shape[0])
    t2 = np.random.permutation(noc)
    for k in t1:
        Resf = Xf[k, :] - np.dot(A[k, :], (krSf * np.exp(np.outer(T[k, :], krf))))
        for d in t2:
            if np.sum(TauW[d, :]) > 0:
                Resfud = Resf + A[k, d] * (krSf[d, :] * np.exp(T[k, d] * krf))
                # Xft = np.squeeze(unmatricizing(Resfud, 1, [1, Nf[1], np.prod(Nf[2:])]))
                Xft = Resfud
                # if krpr.shape[0] == 1:
                #     Xd = Xft
                # else:
                #     Xd = np.dot(krpr[:, d].T, Xft.T)
                Xd = Xft
                
                C = Xd * np.conj(Sf[d, :])
                if N[1] % 2 == 0:
                    C = np.concatenate((C, np.conj(C[-2::-1])))
                else:
                    C = np.concatenate((C, np.conj(C[::-1])))
                    
                C = np.fft.ifft(C, axis=0, n=10000)
                C = C * TauW[d, :]
                
                ind = np.argmax(C)
                                
                # if constr:
                #     ind = np.argmax(C)
                # else:
                #     ind = np.argmax(np.abs(C))
                T[k, d] = ind - sSf - 1
                A[k, d] = C[ind] / (np.sum(w * (krSf[d, :] * np.conj(krSf[d, :]))) / sSf + Lambda[d])
                if abs(T[k, d]) > (sSf / 2):
                    if T[k, d] > 0:
                        T[k, d] = T[k, d] - sSf
                    else:
                        T[k, d] = T[k, d] + sSf
                Resf = Resfud - A[k, d] * (krSf[d, :] * np.exp(T[k, d] * krf))


def mgetopt(opts, key, default):
    if key in opts:
        return opts[key]
    else:
        return default


def generateTauWMatrix(TauW, N2):
    TauWMatrix = np.zeros((TauW.shape[0], N2))

    for d in range(TauW.shape[0]):
        TauWMatrix[d, 0:int(TauW[d, 1])] = 1
        TauWMatrix[d, -1:(-int(TauW[d, 0]) - 1):-1] = 1

    return TauWMatrix

def matricizing(X, n):
    # Turn tensor to matrix along n'th mode
    if n is None:
        return X
    else:
        sX = np.array(X.shape)
        N = X.ndim
        n2 = np.setdiff1d(np.arange(1, N+1), n)
        Y = np.reshape(np.transpose(X, axes=[n-1] + list(n2-1)), (np.prod(sX[n-1]), np.prod(sX[n2-1])))
        return Y

def unmatricizing(X, n, D):
    # Inverse function of matricizing
    ind = list(range(len(D)))
    del ind[n]
    
    if n == 1:
        perm = list(range(len(D)))
    else:
        perm = list(range(1, n)) + [0] + list(range(n+1, len(D)))

    X = np.transpose(np.reshape(X, [D[i] for i in [n] + list(range(n)) + list(range(n+1, len(D)))]), perm)
    return X

if __name__ == "__main__":
    print("This is a module. Not intended to be run standalone.")
    opts = None
    noc = 3
    N = X.shape
    TauW = mgetopt(opts, 'TauW', np.column_stack((np.ones(noc) * -N[1] / 2, np.ones(noc) * N[1] / 2)))
    TauWMatrix = generateTauWMatrix(TauW, N[1])
    
    