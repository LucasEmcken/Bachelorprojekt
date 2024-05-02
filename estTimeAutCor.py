import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import ifft


def estTimeAutCor(Xf, A, Sf, krSf, krf, Tau, Nf, N, w, TauW, Lambda):
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
                Xd = Xft
                
                C = Xd * np.conj(Sf[d, :])
                if N[1] % 2 == 0:
                    C = np.concatenate((C, np.conj(C[-2:0:-1])))
                else:
                    C = np.concatenate((C, np.conj(C[-1:0:-1])))
                C = np.fft.ifft(C, axis=0)
                # C = ifft(C, norm='ortho')
                
                C = C * TauW[d, :]
                
                ind = np.argmax(C)
                
                Tau[d] = ind - sSf - 1
                A[k, d] = C[ind] / (np.sum(w * (krSf[d, :] * np.conj(krSf[d, :]))) / sSf + Lambda[d])
                if abs(Tau[d]) > (sSf / 2):
                    if Tau[d] > 0:
                        Tau[d] = Tau[d] - sSf
                    else:
                        Tau[d] = Tau[d] + sSf
                Resf = Resfud - A[k,d] * (krSf[d, :] * np.exp(Tau[d] * krf))
    return Tau, A



def generateTauWMatrix(TauW, N2):
    TauWMatrix = np.zeros((TauW.shape[0], N2))

    for d in range(TauW.shape[0]):
        TauWMatrix[d, 0:int(TauW[d, 1])] = 1
        TauWMatrix[d, N2+int(TauW[d, 0]):N2] = 1

    return TauWMatrix



def estT(X,W,H, Tau=None):
    N = [*X.shape,1]
    Xf = np.fft.fft(X)
    Xf = np.ascontiguousarray(Xf[:,:int(np.floor(Xf.shape[1]/2))+1])
    Nf = np.array(Xf.shape)
    # A = np.array(np.copy(W), dtype = np.complex128)
    A = W
    # A = np.ones(shape=(W.shape[0],W.shape[1]),dtype=np.complex128)
    noc = A.shape[1]
    Sf = np.ascontiguousarray(np.fft.fft(H)[:,:Nf[1]])
    krSf = np.conj(Sf)
    krf = (-1j*2*np.pi * np.arange(0,N[1])/N[1])[:Nf[1]]
    # Tau = np.zeros((N[0],noc))
    if Tau is None:
        Tau = np.zeros((N[0],noc))
    N = np.array(N)
    w = np.ones(Xf.shape[1])
    #TauW = np.column_stack((np.ones((3,1)) * -N[1]*2 / 2, np.ones(3) * N[1] / 2))
    TauW = np.ones((noc, 1))*np.array([-800,800])
    SST = np.sum(X**2)
    sigma_sq = SST / (11*np.prod(N) -X.shape[0]*X.shape[1])
    Lambda = np.ones(noc)#*sigma_sq.real
    Lambda *= 1
    for i in range(N[0]):
        Tau[i], A[i] = estTimeAutCor(Xf[i],A[i],Sf,krSf,krf,Tau[i],Nf,N,w,TauW,Lambda)

    
    Tau = np.array(Tau,dtype=np.float64)
    return Tau


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
    print(W)
    W_before = np.copy(W)
    W=np.zeros_like(W)
    tau_est = estT(X,W,H)
    print(W)
    
    plt.subplot(1, 2, 1)
    plt.imshow(W_before)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(W)
    plt.colorbar()
    plt.show()


    plt.subplot(2,1,1)
    plt.plot(shift_dataset(W_before, H, tau).real.T)
    plt.subplot(2,1,2)
    plt.plot(shift_dataset(W, H, tau-tau_est).real.T)

    plt.show()
    