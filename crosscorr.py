from latent_model_comparison import LatentModelComparer
#import pearsonr
# from scipy.stats import pearsonr
from scipy.signal import correlate

def normalized_cross_correlation(series1, series2, max_lag):
    series1 = (series1 - series1.mean()) / series1.std()
    series2 = (series2 - series2.mean()) / series2.std()
    
    cross_corr = correlate(series1, series2, mode='full')
    
    # Normalization factor (number of observations)
    n = len(series1)
    cross_corr /= n
    
    # Define the lags
    lags = np.arange(-n + 1, n)
    
    # Slice the cross-correlation result to get the relevant part
    mid = len(cross_corr) // 2
    relevant_corr = cross_corr[mid - max_lag: mid + max_lag + 1]
    relevant_lags = lags[mid - max_lag: mid + max_lag + 1]
    
    return relevant_lags, relevant_corr


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from shiftNMF_frozen import ShiftNMF
    
    from helpers.generators import *
    
    #Create data
    # Define random sources, mixings and shifts; H, W and tau
    N, M, d = 7, 10000, 5
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
    # W = np.append(W, [[1,0,0]], axis=0)
    # W = np.append(W, [[0,1,0]], axis=0)
    # W = np.append(W, [[0,0,1]], axis=0)
    # N = N+3

    #W = np.random.rand(N, d)
    shift = 100
    # Random gaussian shifts
    tau = np.random.randint(-shift, shift, size=(N, d))
    tau[W==0] = 0
    #set tau to 0 where W is 0

    # tau = np.zeros((N,d))
    #tau = np.random.randint(0, 1000, size=(N, d))
    # Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
    mean = [1000, 2800, 5000, 6500, 8000]
    std = [80, 50, 100, 70, 120]
    t = np.arange(0, 10000, 1)

    H = np.array([multiplet(t, 1, m, s, 100) for m, s in list(zip(mean, std))])
    # H_lorentz = np.array([m(m, s, t) for m, s in list(zip(mean, std))])

    X = shift_dataset(W, H, tau)

    # plt.plot(X.T)
    # plt.show()

    # Range of number of components
    components_range = range(1, 10)

    autocorrelations = []
    stds = []

    for k in components_range:
        print(f'fitting with components = {k}')
        temp_auto = []
        for repeat in range(3):
            print(f'round {repeat}')
            model = ShiftNMF(X, k, lr=0.3, alpha=1e-6, patience=1000, min_imp=0)
            W_est,H_est,tau_est, running_loss_hybrid = model.fit(verbose=True, return_loss=True, max_iter=1000, tau_iter=0, Lambda=0.5)
            
            #if k is less than H.shape[0], we add rows with noise to H
            # if k < H.shape[0]:
            #     H_est = np.vstack((H_est, np.random.rand(H.shape[0]-k, H.shape[1])))
            
            # if k > H.shape[0]:
            #     H = np.vstack((H, np.random.rand(k-H.shape[0], H.shape[1])))
            
            # print(H.shape, H_est.shape)
            
            # Create the comparer object
            comparer = LatentModelComparer(H.T, H_est.T)
            # Compare the true and estimated components
            C = comparer.match(type='exact', measure='crosscorr')
            temp_auto.append(np.mean(C[1]))
        
        autocorrelations.append(np.mean(temp_auto))
        stds.append(np.std(temp_auto))

    # Plotting the results
    plt.figure(figsize=(10, 6))
    # plt.plot(components_range, autocorrelations, marker='o')
    
    #error plot with std
    plt.errorbar(components_range, autocorrelations, yerr=stds, fmt='o', capsize=5)
    
    #make vline at true component number
    plt.axvline(x=H.shape[0], color='r', linestyle='--')
    
    plt.xlabel('Number of Components')
    plt.ylabel('Autocorrelation (H_true, H_est)')
    plt.title('Autocorrelation vs Number of Components')
    plt.grid(True)
    plt.show()