
#start by making NMF withou any shift

import numpy as np
import pandas as pd
import torch
from shiftNMF_frozen import ShiftNMF

import numpy as np
import numpy as np

# Step 2: Define Functions
def fourier_transform(data):
    return np.fft.fft(data)
    
def vectorize(matrix):
    return matrix.flatten()

def least_square_error(Vtilde_f, Wtilde_f, Htilde_f, Xtilde_f):
    return np.linalg.norm(Vtilde_f - np.einsum('Nd,NdM->NM', Wtilde_f, Htilde_f))**2

def gradient(tau, Wtilde, Htilde, Vtilde, Xtilde):
    # Get the shapes of the matrices
    N, F = Vtilde.shape
    D = Wtilde.shape[1]
    M = N * D

    # Initialize the gradient matrix
    grad = np.zeros((M, 1), dtype=np.complex128)

    # Iterate over frequencies
    for f in range(F):
        # Calculate the omega value
        omega = 2 * np.pi * f / F

        # Calculate Qtilde and Etilde
        Qtilde = np.dot(Wtilde[:, :, f], Htilde[:, f])
        Etilde = Vtilde[:, f] - Qtilde

        # Iterate over sensors
        for n in range(N):
            for d in range(D):
                # Calculate the index
                t = n + d * N

                # Calculate the sum term
                sum_term = 0
                for n_prime in range(N):
                    sum_term += 2 * omega * np.conj(Qtilde[d, n]) * np.conj(Etilde[n_prime, f])

                # Update the gradient
                grad[t] -= sum_term / M

    return grad


def hessian(tau, Wtilde, Htilde, Vtilde, Xtilde):
    # Get the shapes of the matrices
    N, F = Vtilde.shape
    D = Wtilde.shape[1]

    # Initialize the Hessian matrix
    hess = np.zeros((N * D, N * D), dtype=np.complex128)

    # Iterate over frequencies
    for f in range(F):
        # Calculate the omega value
        omega = 2 * np.pi * f / F

        # Calculate Qtilde and Etilde
        Qtilde = np.dot(Wtilde[:, :, f], Htilde[:, f])
        Etilde = Vtilde[:, f] - Qtilde

        # Iterate over sensors
        for n in range(N):
            for d in range(D):
                for n_prime in range(N):
                    for d_prime in range(D):
                        # Calculate the indices
                        t = n + d * N
                        t_prime = n_prime + d_prime * N

                        # Calculate the omega_prime value
                        omega_prime = 2 * np.pi * f / F

                        # Calculate the first term
                        if n != n_prime or d != d_prime:
                            term1 = -2 / N * omega**2 * np.conj(Qtilde[d, n]) * Qtilde[d_prime, n_prime]
                        else:
                            term1 = 0

                        # Calculate the second term
                        if n == n_prime and d == d_prime:
                            term2 = -2 / N * omega**2 * np.conj(Qtilde[d, n]) * (np.conj(Qtilde[d, n]) + np.conj(Etilde[n, f]))
                        elif n == n_prime and d != d_prime:
                            term2 = -2 / N * omega**2 * np.conj(Qtilde[d, n]) * (np.conj(Qtilde[d_prime, n]) + np.conj(Etilde[n, f]))
                        elif n != n_prime and d == d_prime:
                            term2 = -2 / N * omega**2 * np.conj(Qtilde[d, n]) * np.conj(Etilde[n, f])
                        else:
                            term2 = 0

                        hess[t, t_prime] += term1 + term2

    return hess

def cross_correlation(Rtilde, Htilde):
    # Get the shape of Rtilde
    N, F = Rtilde.shape

    # Initialize the cross-correlation array
    cross_corr = np.zeros((N, F))

    # Iterate over frequencies
    for f in range(F):
        # Calculate the cross-correlation for each sensor
        for n in range(N):
            # Calculate the dot product between Rtilde_n and Htilde
            cross_corr[n, f] = np.dot(Rtilde[n, :], Htilde[:, f])

    return cross_corr

# Step 3: Newton-Rhapson Method
def newton_rhapson(tau, Wtilde, Htilde, Vtilde, Xtilde, learning_rate):
    # Calculate gradient and Hessian
    grad = gradient(tau, Wtilde, Htilde, Vtilde, Xtilde)
    hess = hessian(tau, Wtilde, Htilde, Vtilde, Xtilde)

    # Update tau using Newton-Rhapson method
    tau -= learning_rate * np.linalg.inv(hess) @ grad

    return tau

def re_estimate_tau(W, H, X, tau):
    # Get the shape of X
    N, F = X.shape[0], X.shape[1]

    # Initialize the new tau matrix
    new_tau = np.zeros_like(tau)

    # Calculate Rtilde for each sensor frequency
    Rtilde = np.zeros((N, F))
    for n in range(N):
        for f in range(F):
            Rtilde[n, f] = X[n, f] - np.sum(np.dot(W[n, :], H[:, f]))

    # Iterate over sensors and sources to estimate tau
    for n in range(N):
        for d_prime in range(W.shape[1]):
            # Calculate the cross-correlation
            ctilde = np.correlate(Rtilde[n], H[d_prime], mode='valid')

            # Find the index of the maximum cross-correlation
            t_max = np.argmax(ctilde)

            # Update the tau value
            new_tau[n, d_prime] = t_max - (W.shape[1] + 1)

    return new_tau


def optimize_tau(model, W, H, tau, X):
    learning_rate = 10
    max_iterations = 500
    convergence_threshold = 1e-6
    for i in range(max_iterations):
        # Calculate Vtilde, Wtilde, Htilde, Etilde
        Vtilde = np.fft.fft(X)
        Wtilde = np.fft.fft(W, axis=0)
        Htilde = np.fft.fft(H, axis=0)
        Etilde = Vtilde - np.tensordot(Wtilde, Htilde, axes=([1],[0]))

        # Calculate gradient and Hessian
        grad = gradient(tau, Wtilde, Htilde, Vtilde, X)
        hess = hessian(tau, Wtilde, Htilde, Vtilde, X)

        # Update tau using Newton-Raphson method
        tau -= learning_rate * np.linalg.inv(hess) @ grad

        # Re-estimate tau every 20th iteration
        if i % 20 == 0:
            tau = re_estimate_tau(W, H, X, tau)

        # Check convergence criteria
        if np.linalg.norm(learning_rate * np.linalg.inv(hess) @ grad) < convergence_threshold:
            break


def alt_update(model, X, n_alts = 10):
    
    for i in range(n_alts):
        #optimize W and H
        print("Fitting model...")
        W, H, tau = model.fit(max_iter=500)
        #optimize tau
        print("Optimizing tau...")
        optimize_tau(model, W, H, tau, X)

if __name__ == "__main__":
    import numpy as np
    import pandas as pd


    # Load data
    X  = pd.read_csv("X.csv").to_numpy()
    model = ShiftNMF(X, alpha=1e-5, rank = 3)
    
    alt_update(model, X)