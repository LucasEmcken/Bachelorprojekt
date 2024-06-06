from scipy.stats import pearsonr

# Function to compute autocorrelation between H_true and H_est
def compute_autocorrelation(H_true, H_est):
    correlations = []
    for j in range(H_true.shape[1]):
        r, _ = pearsonr(H_true[:, j], H_est[:, j])
        correlations.append(r)
    return np.mean(correlations)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from shiftNMF_frozen import ShiftNMF
    
    from helpers.generators import *
    
    #Create data
    # Define random sources, mixings and shifts; H, W and tau
    N, M, d = 7, 10000, 3
    Fs = 1000  # The sampling frequency we use for the simulation
    t0 = 10    # The half-time interval we look at
    t = np.arange(-t0, t0, 1/Fs)  # the time samples
    f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples

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
        return V.real

    np.random.seed(42)

    # Random mixings:
    W = np.random.dirichlet(np.ones(d), N)
    W = np.append(W, [[1,0,0]], axis=0)
    W = np.append(W, [[0,1,0]], axis=0)
    W = np.append(W, [[0,0,1]], axis=0)
    N = N+3

    #W = np.random.rand(N, d)
    shift = 500
    # Random gaussian shifts
    tau = np.random.randint(-shift, shift, size=(N, d))
    tau[W==0] = 0
    #set tau to 0 where W is 0

    # tau = np.zeros((N,d))
    #tau = np.random.randint(0, 1000, size=(N, d))
    # Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
    mean = [2000, 5000, 8000]
    std = [100, 300, 50]
    t = np.arange(0, 10000, 1)

    H = np.array([multiplet(t, 1, m, s, 100) for m, s in list(zip(mean, std))])
    # H_lorentz = np.array([m(m, s, t) for m, s in list(zip(mean, std))])
    H_duplet = np.array([multiplet(t, 2, m, s, 1000) for m, s in list(zip(mean, std))])

    X = shift_dataset(W, H, tau)

    # Range of number of components
    components_range = range(1, 6)

    autocorrelations = []

    for k in components_range:
        print('fitting')
        model = ShiftNMF(X, k, lr=0.1, alpha=1e-6, patience=1000, min_imp=0)
        W_est,H_est,tau_est, running_loss_hybrid = model.fit(verbose=True, return_loss=True, max_iter=500, tau_iter=0, Lambda=0.3)

        
        autocorr = compute_autocorrelation(H, H_est.T)
        autocorrelations.append(autocorr)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(components_range, autocorrelations, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Autocorrelation (H_true, H_est)')
    plt.title('Autocorrelation vs Number of Components')
    plt.grid(True)
    plt.show()