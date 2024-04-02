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
    noc = A.shape[1]
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
                    
                C = np.fft.ifft(C, axis=0)
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

if __name__ == "__main__":
    print("This is a module. Not intended to be run standalone.")
    