import numpy as np
import matplotlib.pyplot as plt
from ShiftCP import ShiftCP
from reconstructShiftCP import reconstructShiftCP

# Simulate ShiftCP data
Fs = 512  # Sample rate
time_points = np.linspace(0, 1, Fs)

# Generate non-negative trial strengths
FACT = [np.random.rand(10, 4)]

# Generate Drift component
FACT.append(np.column_stack((time_points - np.mean(time_points),
                              np.cos(2 * np.pi * 24 * time_points * Fs),
                              np.cos(2 * np.pi * 12 * time_points * Fs),
                              np.cos(2 * np.pi * 50 * time_points * Fs))))
# Normalize generated time components
FACT[1] = FACT[1] / np.linalg.norm(FACT[1], axis=0)

# Generate random topography
FACT.append(np.random.randn(10, 4))

# Generate delays between -10 and 10 samples
T = np.ceil(20 * np.random.rand(10, 4) - 10)

# Reconstruct the data and add noise
# Define the function reconstructShiftCP(FACT, T) separately
# Add noise
Rec = reconstructShiftCP(FACT, T)  # Generate Data
X = Rec + 0.01 * np.random.randn(*Rec.shape)  # Add noise

# Set algorithm parameters
opts = {'maxiter': 250, 'constr': [1, 0, 0], 'SNR': 0, 'Sample_rate': Fs, 'ARD': True, 'InitRun': 3}

# Run the algorithm
FACT_est, T_est, nLogP, varexpl, Lambda = ShiftCP(X, 10, opts)

# Plot the results
noc_est = FACT_est[0].shape[1]
plt.figure(figsize=(12, 8))
for d in range(noc_est):
    plt.subplot(noc_est, 3, (d - 1) * 3 + 1)
    plt.plot(FACT_est[0][:, d] * np.exp(-1j * 2 * np.pi * T_est[:, d] / X.shape[1]), '.')
    plt.axis('tight')
    plt.ylabel('Component ' + str(d), fontweight='bold')

    plt.subplot(noc_est, 3, (d - 1) * 3 + 2)
    plt.plot(FACT_est[1][:, d])
    plt.axis('tight')
    plt.ylabel('Mode 2 Scores')
    plt.xlabel('Time points')

    plt.subplot(noc_est, 3, (d - 1) * 3 + 3)
    plt.bar(np.arange(len(FACT_est[2])), FACT_est[2][:, d])

plt.show()
