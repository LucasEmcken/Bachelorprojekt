from latent_model_comparison import LatentModelComparer
#import pearsonr
# from scipy.stats import pearsonr
from scipy.signal import correlate
from sklearn.metrics import r2_score

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
    N, M, d = 50, 10000, 5
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
    shift = 500
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

    #scale X up 1000 times
    

    #bootstrap samples of X along rows with replacement

    noise = np.random.normal(0, 0.000005, X.shape)
    
    # noise = np.abs(noise)
    X_noisy = X + noise

    # exit()
    best_fit = np.linalg.norm(X - X_noisy,'fro')

    T = 10
    S = 15
    X_boot = np.zeros((T,S,M))
    X_boot_noise = np.zeros((T,S,M))
    
    for i in range(T):
        for j in range(S):
            row = np.random.randint(0, N)
            X_boot[i,j] = X[row]
            X_boot_noise[i,j] = X_noisy[row]

    # Range of number of components
    components_range = range(1, 10)

    noise_errors = []
    stds = []

    for k in components_range:
        print(f'fitting with components = {k}')
        temp_noise = []
        for repeat in range(10):
            
            curr_boot = X_boot[repeat]
            
            #add noise
            # X_boot_noisy = X_boot + np.random.normal(0, 0.000005, X_boot.shape)
            
            print(f'round {repeat}')
            model = ShiftNMF(curr_boot, k, lr=0.1, alpha=1e-6, patience=1000, min_imp=0)
            W_est,H_est,tau_est, running_loss_hybrid = model.fit(verbose=True, return_loss=True, max_iter=1000, tau_iter=0, Lambda=0.1)
            
            X_est = shift_dataset(W_est, H_est, tau_est)
            noise_error = np.linalg.norm(X_boot[repeat] - X_est,'fro')
            temp_noise.append(noise_error)
            
        noise_errors.append(temp_noise)
        # stds.append(np.std(temp_noise))

    # Plotting the results
    plt.figure(figsize=(10, 6))
    # plt.plot(components_range, autocorrelations, marker='o')
    
    # plt.errorbar(components_range, noise_errors, yerr=stds, fmt='o')
    #mat a boxchart of the noise errors
    plt.boxplot(noise_errors)
    plt.title('Noise error vs number of components')
    plt.xlabel('Number of components')
    plt.ylabel('Noise error')
    #make a horizontal line at the best level
    best_fit = np.linalg.norm(X - X_noisy,'fro')
    plt.axhline(y=best_fit, color='r', linestyle='-')
    plt.legend(['Best theoretical fit'])
    plt.show()