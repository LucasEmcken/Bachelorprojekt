import numpy as np 
import matlab

import matplotlib.pyplot as plt

def estTimeAutCor(Xf, A, Sf, krSf, krf, Tau, Nf, N, w, TauW, Lambda):
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

    TauW = generateTauWMatrix(TauW,N[1])
    #reshape to (1, number_of_components)
    Xf = np.expand_dims(Xf, axis=0)
    A = np.expand_dims(A, axis=0)
    #T = np.expand_dims(T, axis=1)
    noc = A.shape[1]
    # noc = 3
    if N[1] % 2 == 0:
        sSf = 2 * Nf[1] - 2
    else:
        sSf = 2 * Nf[1] - 1
    t1 = np.random.permutation(A.shape[0])
    t2 = np.random.permutation(noc)
    for k in t1:
        Resf = Xf[k, :] - np.dot(A[k, :], (krSf * np.exp(Tau[k].conj().T * krf)))
        for d in t2:
            if np.sum(TauW[d, :]) > 0:
                Resfud = Resf + A[k, d] * (krSf[d, :] * np.exp(Tau[d] * krf))
                # Xft = np.squeeze(unmatricizing(Resfud, 1, [1, Nf[1], np.prod(Nf[2:])]))
                Xft = Resfud
                # if krpr.shape[0] == 1:
                #     Xd = Xft
                # else:
                #     Xd = np.dot(krpr[:, d].T, Xft.T)
                Xd = Xft
                
                C = Xd * np.conj(Sf[d, :])
                if N[1] % 2 == 0:
                    C = np.concatenate((C, np.conj(C[-2:0:-1])))
                else:
                    C = np.concatenate((C, np.conj(C[-1:0:-1])))
                C = np.fft.ifft(C, axis=0)
                C = C * TauW[d, :]
                
                ind = np.argmax(C)
                                
                # if constr:
                #     ind = np.argmax(C)
                # else:
                #     ind = np.argmax(np.abs(C))
                Tau[d] = ind - sSf - 1
                A[k, d] = C[ind] / (np.sum(w * (krSf[d, :] * np.conj(krSf[d, :]))) / sSf + Lambda[d])
                if abs(Tau[d]) > (sSf / 2):
                    if Tau[d] > 0:
                        Tau[d] = Tau[d] - sSf
                    else:
                        Tau[d] = Tau[d] + sSf
                Resf = Resfud - A[k,d] * (krSf[d, :] * np.exp(Tau[d] * krf))
    return Tau



def generateTauWMatrix(TauW, N2):
    TauWMatrix = np.zeros((TauW.shape[0], N2))

    for d in range(TauW.shape[0]):
        TauWMatrix[d, 0:int(TauW[d, 1])] = 1
        TauWMatrix[d, N2+int(TauW[d, 0]):N2] = 1

    return TauWMatrix



def estT(X,W,H):
    N = [*X.shape,1]
    Xf = np.fft.fft(X)
    Xf = np.ascontiguousarray(Xf[:,:int(np.floor(Xf.shape[1]/2))+1])
    Nf = np.array(Xf.shape)
    A = np.copy(W)
    noc = A.shape[1]
    Sf = np.ascontiguousarray(np.fft.fft(H)[:,:Nf[1]])
    krpr = np.array([0,0,0])
    krSf = np.conj(Sf)
    krf = np.fft.fftfreq(Nf[1])
    T = np.zeros((N[0],noc))
    N = np.array(N)
    w = np.ones(Xf.shape[1])
    constr = False
    #TauW = np.column_stack((np.ones((3,1)) * -N[1]*2 / 2, np.ones(3) * N[1] / 2))
    TauW = np.ones((noc, 1))*np.array([-800,800])
    SST = np.sum(X**2)
    sigma_sq = SST / (11*np.prod(N) -X.shape[0]*X.shape[1])
    Lambda = np.ones(noc)*10#*sigma_sq.real
    for i in range(N[0]):
        T[i] = estTimeAutCor(Xf[i],A[i],Sf,krSf,krf,T[i],Nf,N,w,TauW,Lambda)
    #T = my_estTimeAutCor.estTimeAutCor(Xf,A,Sf,krpr,krSf,krf,T,Nf,N,w,constr,TauW,Lambda)
    #my_shiftCP.terminate()

    T = np.array(T,dtype=np.float64)
    return T


if __name__ == "__main__":
    np.random.seed(45)
    def shift_dataset(W, H, tau):
        # Get half the frequencies
        Nf = H.shape[1] // 2 + 1
        # Fourier transform of S along the second dimension
        Hf = np.fft.fft(H, axis=1)
        # Keep only the first Nf[1] elements of the Fourier transform of S
        Hf = Hf[:, :Nf]
        # Construct the shifted Fourier transform of S
        Hf_reverse = np.fliplr(Hf[:, 1:Nf - 1])
        # Concatenate the original columns with the reversed columns along the second dimension
        Hft = np.concatenate((Hf, np.conj(Hf_reverse)), axis=1)
        f = np.arange(0, M) / M
        omega = np.exp(-1j * 2 * np.pi * np.einsum('Nd,M->NdM', tau, f))
        Wf = np.einsum('Nd,NdM->NdM', W, omega)
        # Broadcast Wf and H together
        Vf = np.einsum('NdM,dM->NM', Wf, Hft)
        V = np.fft.ifft(Vf)
        return V

    N, M, d = 5, 10000, 3
    Fs = 1000  # The sampling frequency we use for the simulation
    t0 = 10    # The half-time interval we look at
    t = np.arange(-t0, t0, 1/Fs)  # the time samples
    f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples

    def gauss(mu, s, time):
        return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)

    W = np.random.dirichlet(np.ones(d), N)

    shift = 400
    # Random gaussian shifts
    tau = np.random.randint(-shift, shift, size=(N, d))

    mean = [1500, 5000, 8500]
    std = [30, 40, 50]
    t = np.arange(0, 10000, 1)

    H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])

    X = shift_dataset(W, H, tau)
    
    tau_est = estT(X,W,H)
    
    plt.subplot(1, 2, 1)
    plt.imshow(tau)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(tau_est)
    plt.colorbar()
    plt.show()


    plt.subplot(2,1,1)
    plt.plot(shift_dataset(W, H, tau).real.T)
    plt.subplot(2,1,2)
    plt.plot(shift_dataset(W, H, tau-tau_est).real.T)

    plt.show()
    