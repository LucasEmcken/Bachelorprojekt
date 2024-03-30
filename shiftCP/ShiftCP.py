import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import khatri_rao as krprod
import time

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
            F_tmp.append(res[0]) #error might be due to the script not being done
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
            # FACT.append(initializeFACT(N, constr, noc))
            FACT = initializeFACT(N, constr, noc)
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
        Frequencies = np.arange(0, Nf[1]) / Sample_rate * Ns
        for d in range(noc):
            for k in range(len(RemoveFrequencies[d])):
                ind1 = np.where(Frequencies < RemoveFrequencies[d][k][0])[0]
                if len(ind1) == 0:
                    ind1 = 0
                else:
                    ind1 = ind1[-1]
                ind2 = np.where(Frequencies > RemoveFrequencies[d][k][1])[0]
                if len(ind2) == 0:
                    ind2 = Nf[1]
                else:
                    ind2 = ind2[0]
                includeFreq[ind1:ind2, d] = False
        Sf[includeFreq.T == False] = 0
    t = time.process_time()
    iter = 0
    n = np.setdiff1d(np.arange(1, nrmodes + 1), [1, 2])  # Impute missing values
    # print(n)

    # Handle missing data and evaluate LS_error
    krprt = np.ones((1, noc))
    krkrt = np.ones(noc)
    for k in range(2, nrmodes):
        krprt = krprod(krprt, FACT[k])
        krkrt = krkrt * (FACT[k].T @ FACT[k])
    Rec = reconstruct(FACT[0], Sf, krprt, T, f, N)
    X[miss_ind] = Rec[miss_ind]
    X2 = matricizing(X, 2).T
    Xf = fft(X2, axis=1)
    Xf = Xf[:, :int(np.floor(Xf.shape[1] / 2)) + 1]
    Xf = unmatricizing(Xf.T, 0, [N[0], Xf.shape[1], *N[2:]]) #line 212 in matlab The 0 should be 1 in theory, but it returns the wrong shape
    LS_error = np.sum((X - Rec) ** 2)
    varexpl = (SST - LS_error) / SST
    NN = np.sum(N[constr != 2])

    # Evaluate prior
    nrmFACT = np.zeros(noc)
    for t in range(nrmodes):
        if constr[t] != 2:
            if t == 1:
                if N[1] % 2 == 0:
                    Sft = np.concatenate((Sf, np.conj(Sf[:, :-1][:, ::-1])), axis=1)
                else:
                    Sft = np.concatenate((Sf, np.conj(Sf[:, :-1][:, ::-1])), axis=1)
                nrmFACT = nrmFACT + np.sum(Sft * np.conj(Sft), axis=1) / N[1] ** 2
            else:
                nrmFACT = nrmFACT + np.sum(FACT[t] ** 2)

    const = mgetopt(opts, 'const', 0)
    nLogP = [np.inf]  # 0.5 * LS_error / sigma_sq + 0.5 * np.sum(Lambda * (nrmFACT + 1e-6 * SST / (np.prod(N) - np.sum(miss_ind)))) - 0.5 * NN * np.sum(np.log(Lambda)) + const
    dnLogP = np.inf
    
    #line 236 in matlab
    #Algorithm Display
    if False:
        print(' ')
        print('Shifted CP Analysis')
        if estWholeSamples:
            print('Using only integer delays estimated by cross-spectra')
        else:
            print('Using non-integer delays estimated by Newton-Rhapson')
        print('A {} component model will be fitted'.format(noc))
        print(' ')
        print('To stop algorithm press control C')
        print(' ')
        dheader = '{:<12s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} |'.format('Iteration', 'Expl. var.', 'Cost func.', 'd_cost/cost', ' Time(s)   ', ' Min step   ')
        dline = '-------------+--------------+--------------+--------------+--------------+--------------+'
        print(dline)
        print(dheader)
        print(dline)
    
    #main loop
    while iter < maxiter and dnLogP >= abs(nLogP[iter]) * conv_crit:
        if iter % 25 == 0 and False:
            print(dline)
            print(dheader)
            print(dline)
        nLogP_old = nLogP[iter]
        iter += 1
        told = t

        # Project out mode 3:end to speed up algorithm
        krprt = np.ones((1, noc))
        krkrt = np.ones(noc)
        for k in n-1:
            krprt = krprod(FACT[k], krprt)
            krkrt = krkrt * (FACT[k].T @ FACT[k])
            
        if nrmodes > 2:
            print(list(range(2, nrmodes)))
            # Xfp = (krprt.T @ matricizing(Xf, *list(range(2, nrmodes)))).T #here line 280 in matlab issue is with nrmodes, should return shape 10, 2570
            matricized = matricizing(Xf, 1)
            print(matricized.shape)
            exit()
            Xfp = (krprt.T @ matricized).T
            
            Xfp = unmatricizing(Xfp, 2, [Nf[0], Nf[1], noc])
            Xtp = (krprt.T @ matricizing(X, list(range(2, nrmodes + 1)))).T
            Xtp = unmatricizing(Xtp, 2, [N[0], N[1], noc])
            krprtt = np.eye(noc)
        else:
            Xfp = Xf
            Xtp = X
            krprtt = np.ones(noc)
        
        if ARD:
            nLogP.append(0.5 * LS_error / sigma_sq + 0.5 * np.sum(Lambda * (nrmFACT + 1e-6 * SST / (np.prod(N) - np.sum(miss_ind))) - 0.5 * NN * np.sum(np.log(Lambda)) + const))
        else:
            nLogP.append(0.5 * LS_error / sigma_sq)
    

def mgetopt(opts, key, default):
    if key in opts:
        return opts[key]
    else:
        return default

def generateTauWMatrix(TauW, N2):
    TauWMatrix = np.zeros((TauW.shape[0], N2))

    for d in range(TauW.shape[0]):
        TauWMatrix[d, 0:int(TauW[d, 1])] = 1
        TauWMatrix[d, -1:(-int(TauW[d, 0]) - 1):-1] = 1

    return TauWMatrix

def initializeFACT(N, constr, noc):
    FACT = {}

    for i in range(len(N)):
        if constr[i] == 1:
            FACT[i] = np.random.rand(N[i], noc)
        elif constr[i] == 2:
            U, S, V = np.linalg.svd(np.random.randn(N[i], noc))
            FACT[i] = U
        else:
            FACT[i] = np.random.randn(N[i], noc)

    return FACT

def reconstruct(A, Sf, Q, T, f, N):
    if N[1] % 2 == 0:
        X = np.zeros((A.shape[0], 2 * len(f) - 2, Q.shape[0]))
    else:
        X = np.zeros((A.shape[0], 2 * len(f) - 1, Q.shape[0]))

    for i in range(A.shape[0]):
        Sft = Sf * np.exp(np.outer(T[i, :], f))

        if N[1] % 2 == 0:
            Sft = np.hstack((Sft, np.conj(Sft[:, -2:0:-1])))
        else:
            Sft = np.hstack((Sft, np.conj(Sft[:, -1:0:-1])))

        St = np.real(np.fft.ifft(Sft, axis=1))
        X[i, :, :] = np.matmul(St.T, krprod(A[i, :].reshape(1, -1), Q).T)
    return X


def matricizing(X, n):
    # Turn tensor to matrix along n'th mode
    if n is None:
        return X
    else:
        sX = np.array(X.shape)
        N = X.ndim
        n2 = np.setdiff1d(np.arange(1, N+1), n)
        Y = np.reshape(np.transpose(X, axes=[n-1] + list(n2-1)), (np.prod(sX[n-1]), np.prod(sX[n2-1])))
        return Y


def unmatricizing(X, n, D):
    # Inverse function of matricizing
    ind = list(range(len(D)))
    del ind[n]
    
    if n == 1:
        perm = list(range(len(D)))
    else:
        perm = list(range(1, n)) + [0] + list(range(n+1, len(D)))

    X = np.transpose(np.reshape(X, [D[i] for i in [n] + list(range(n)) + list(range(n+1, len(D)))]), perm)
    return X

# def estTimeAutCor(Xf, A, Sf, krpr, krSf, krf, T, Nf, N, w, constr, TauW, Lambda):
def estTimeAutCor(Xf, A, Sf, krSf, krf, T, Nf, N, w, TauW, Lambda):

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