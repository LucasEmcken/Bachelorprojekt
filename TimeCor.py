import numpy as np 

import matplotlib.pyplot as plt


# import ShiftCPM
import estTimeAutCor
import matlab

def estT(X,W,H):
    my_estTimeAutCor = estTimeAutCor.initialize()
    N = [*X.shape,1]
    Xf = np.fft.fft(X)
    Xf = np.ascontiguousarray(Xf[:,:int(np.floor(Xf.shape[1]/2))+1])
    Nf = np.array(Xf.shape)
    A = W
    noc = A.shape[1]
    Sf = np.ascontiguousarray(np.fft.fft(H)[:,:Nf[1]])
    krpr = np.array([0,0,0])
    #krSf = np.conj(Sf)
    krSf = Sf
    krf = (-1j*2*np.pi * np.arange(0,N[1])/N[1])[:Nf[1]]

    Tau = np.zeros((N[0],noc))
    N = np.array(N)
    w = np.ones(Xf.shape[1])*2
    w[0] = 1
    if len(W)%2 == 1:
        w[-1] = 1
    constr = True
    #TauW = np.column_stack((np.ones((3,1)) * -N[1]*2 / 2, np.ones(3) * N[1] / 2))
    TauW = np.ones((noc, 1))*np.array([-400,400])

    SST = np.sum(X**2)
    sigma_sq = SST / (11*np.prod(N) -X.shape[0]*X.shape[1])
    Lambda = np.ones(noc)*0#*sigma_sq.real
    Tau = my_estTimeAutCor.estTimeAutCor(Xf,A,Sf,krpr,krSf,krf,Tau,Nf,N,w,constr,TauW,Lambda)
    #T = my_estTimeAutCor.estTimeAutCor(Xf,A,Sf,krpr,krSf,krf,T,Nf,N,w,constr,TauW,Lambda)
    #my_shiftCP.terminate()

    Tau = np.array(Tau,dtype=np.complex128)
    return Tau


if __name__ == "__main__":
    np.random.seed(45)

    def generateTauWMatrix(TauW, N2):
        TauWMatrix = np.zeros((TauW.shape[0], N2))

        for d in range(TauW.shape[0]):
            TauWMatrix[d, 0:int(TauW[d, 1])] = 1
            TauWMatrix[d, N2+int(TauW[d, 0]):N2] = 1

        return TauWMatrix

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

    shift = 200
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
    plt.imshow(np.array(tau_est).real)
    plt.colorbar()
    plt.show()


    plt.subplot(2,1,1)
    plt.plot(shift_dataset(W, H, tau).real.T)
    # plt.subplot(2,1,2)
    # plt.plot(np.dot(W, H).real.T)
    plt.subplot(2,1,2)
    plt.plot(shift_dataset(W, H, tau-tau_est).real.T)


    plt.show()