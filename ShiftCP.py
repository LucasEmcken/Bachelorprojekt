import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import svd

def ShiftCP(X, noc, opts=None):
    if opts is None:
        opts = {}

    # Extracted sizes of the data
    nrmodes = X.ndim
    N = X.shape
    if len(N) < 3:
        N = list(N) + [1]
    miss_ind = np.isnan(X)
    X[miss_ind] = 0
    SST = np.sum(X**2)

    # Extract algorithm parameters
    RemoveFrequencies = mgetopt(opts, 'RemoveFrequencies', [np.array([])] * noc)
    Sample_rate = mgetopt(opts, 'Sample_rate', np.nan)
    ARD = mgetopt(opts, 'ARD', True)
    TauW = mgetopt(opts, 'TauW', np.column_stack((np.ones(noc) * -N[1] / 2, np.ones(noc) * N[1] / 2)))
    TauWMatrix = generateTauWMatrix(TauW, N[1])
    conv_crit = mgetopt(opts, 'conv_crit', 1e-6)
    estWholeSamples = mgetopt(opts, 'estWholeSamples', True)
    nyT = mgetopt(opts, 'nyT', np.ones(N[0]))
    SNR = mgetopt(opts, 'SNR', 0)
    mu = np.ones(nrmodes)
    constr = mgetopt(opts, 'constr', np.zeros(nrmodes))
    maxiter = mgetopt(opts, 'maxiter', 500)
    InitRun = mgetopt(opts, 'InitRun', 0)

    sigma_sq = SST / ((1 + 10**(SNR / 10)) * (np.prod(N) - np.sum(miss_ind)))

    # Initialize variables
    if InitRun > 0:
        opts['maxiter'] = 50
        F_tmp, T_tmp, nLogP_tmp, varexpl, Lambda_tmp, RemoveFrequencies_tmp, _, const_tmp = [], [], [], [], [], [], [], []
        for k in range(InitRun):
            print('Finding best solution nr {} of {}'.format(k + 1, InitRun))
            opts['InitRun'] = 0
            res = ShiftCP(X, noc, opts)
            F_tmp.append(res[0])
            T_tmp.append(res[1])
            nLogP_tmp.append(res[2][-1])
            varexpl.append(res[3])
            Lambda_tmp.append(res[4])
            RemoveFrequencies_tmp.append(res[5])
            const_tmp.append(res[7][-1])

        ind = np.argmin(nLogP_tmp)
        opts['Lambda'] = Lambda_tmp[ind]
        RemoveFrequencies = RemoveFrequencies_tmp[ind]
        opts['const'] = const_tmp[ind]
        opts['FACT'] = F_tmp[ind]
        opts['T'] = T_tmp[ind]
        noc = F_tmp[ind][0].shape[1]

    FACT = []
    if 'FACT' in opts:
        FACT = opts['FACT']
        if constr[0] == 1:
            PP = np.zeros((N[0], noc))
            ZZ = np.tile(np.arange(1, noc + 1), (N[0], 1))
    else:
        for k in range(nrmodes):
            FACT.append(initializeFACT(N, constr, noc))
            if constr[k] == 1 and k == 0:
                PP = np.zeros((N[0], noc))
                ZZ = np.tile(np.arange(1, noc + 1), (N[0], 1))

    T = mgetopt(opts, 'T', np.zeros(FACT[0].shape))
    Lambda = mgetopt(opts, 'Lambda', np.ones(noc) * np.finfo(float).eps * sigma_sq)
    if not ARD:
        Lambda = np.zeros(noc)

    # Transform data to frequency domain
    Ns = N[1]
    f = -1j * 2 * np.pi * np.arange(Ns) / Ns
    Xf = fft(X, axis=1)
    Xf = Xf[:, :int(np.floor(Xf.shape[1] / 2)) + 1]
    Nf = Xf.shape
    Sf = fft(FACT[1].T, axis=1)
    Sf = Sf[:, :Nf[1]]
    f = f[:Nf[1]]

    includeFreq = np.ones((Nf[1], noc), dtype=bool)
    if RemoveFrequencies:
        Frequencies = np.arange(Nf[1]) / Sample_rate * Ns
        for d in range(noc):
            for k in range(RemoveFrequencies[d].shape[0]):
                ind1 = np.where(Frequencies < RemoveFrequencies[d][k, 0])[0]
                ind2 = np.where(Frequencies > RemoveFrequencies[d][k, 1])[0]
                includeFreq[np.intersect1d(ind1, ind2), d] = False

    for it in range(maxiter):
        print('Iteration: {}'.format(it))
        oldT = T.copy()
        oldFACT = FACT.copy()
        oldLambda = Lambda.copy()
        nLogP = 0
        Lambda_nT = Lambda[:, None] * T
        for i in range(noc):
            # Update components
            tmp1 = Xf - Lambda_nT[:, :, i].dot(Sf)
            tmp1[:, ~includeFreq[:, i]] = 0
            FACT[1][:, i] = svd(tmp1, False)[0][:, 0]

            # Update temporal shifts
            tmp1 = Xf - FACT[1].dot(Sf)
            tmp1[:, ~includeFreq[:, i]] = 0
            ttt = ifft(tmp1, axis=1)
            ttt[:, ~includeFreq[:, i]] = 0
            if estWholeSamples:
                T[:, i] = np.mod(T[:, i] + np.real(np.sum(ttt * np.conj(oldFACT[1][:, i]), axis=0)), Ns)
            else:
                T[:, i] = np.mod(T[:, i] + np.round(np.real(np.sum(ttt * np.conj(oldFACT[1][:, i]), axis=0))), Ns)

            # Update precision parameters
            tmp1 = Xf - FACT[1].dot(Sf)
            tmp1[:, ~includeFreq[:, i]] = 0
            tmp2 = fft(np.conj(FACT[1][:, i][:, None] * Sf), axis=1)
            if ARD:
                Lambda[i] = (Ns / (2 * np.pi * np.sqrt(sigma_sq))) / (np.real(np.mean(tmp1 * np.conj(tmp1), axis=0)) + (np.real(np.mean(tmp2 * np.conj(tmp2), axis=0)) + sigma_sq) * Lambda[i])
            else:
                Lambda[i] = (Ns / (2 * np.pi * np.sqrt(sigma_sq))) / (np.real(np.mean(tmp1 * np.conj(tmp1), axis=0)) + sigma_sq)

            # Update scale parameters
            tmp1 = Xf - Lambda_nT[:, :, i].dot(Sf)
            tmp1[:, ~includeFreq[:, i]] = 0
            nLogP = nLogP + np.sum(np.real(np.mean(tmp1 * np.conj(tmp1), axis=0)))
        
        # Convergence criteria
        if np.linalg.norm(T - oldT) / np.sqrt(np.linalg.norm(T) * np.linalg.norm(oldT)) < conv_crit and \
                np.linalg.norm(FACT[1] - oldFACT[1]) / np.sqrt(np.linalg.norm(FACT[1]) * np.linalg.norm(oldFACT[1])) < conv_crit and \
                np.linalg.norm(Lambda - oldLambda) / np.sqrt(np.linalg.norm(Lambda) * np.linalg.norm(oldLambda)) < conv_crit:
            break

    # Transform components to time domain
    Y = []
    for i in range(noc):
        Y.append(np.real(ifft(np.conj(FACT[0][:, i][:, None] * FACT[1][:, i][:, None] * Sf), axis=1)))

    if constr[0] == 1:
        for i in range(noc):
            PP += Y[i] ** 2
        for i in range(noc):
            a = np.where(PP[:, i] != 0)[0]
            ZZ[a, i] = Y[i][a] / np.sqrt(PP[a, i])

    return FACT, T, nLogP, Y, Lambda, RemoveFrequencies, includeFreq, ZZ

def mgetopt(opts, key, default):
    if key in opts:
        return opts[key]
    else:
        return default

def generateTauWMatrix(TauW, N):
    if len(TauW.shape) == 1:
        TauW = TauW[:, None]
    return np.mod(np.tile(np.arange(N), (TauW.shape[0], 1)) + np.transpose(TauW), N)

def initializeFACT(N, constr, noc):
    FACT = []
    for k in range(len(N)):
        FACT.append(np.random.randn(N[k], noc))
        if constr[k] == 1:
            for i in range(noc):
                FACT[k][:, i] = FACT[k][:, i] / np.linalg.norm(FACT[k][:, i])
    return FACT
