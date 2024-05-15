import numpy as np
import math
from sympy.utilities.iterables import multiset_permutations
from scipy import signal

class LatentModelComparer:
    
    def __init__(self, true_factor, est_factor):
        # Specify which factor is desired (true_factor) and what to compare against (est_factor)
        self.A = true_factor
        self.B = est_factor
        
        if np.ndim(self.A) == 1:
            self.A = self.A[:,np.newaxis]
            
        if np.ndim(self.B) == 1:
            self.B = self.B[:,np.newaxis]       
    
    def compare(self, measure):
        nA = self.A.shape[1]
        nB = self.B.shape[1]
        
        if measure == "corr":
            C = np.corrcoef(self.A.T, self.B.T)[0:nA, nA:]
        elif measure == "crosscorr":
            C = np.zeros((nA,nB))
            B = np.empty_like(C)
            for ia in range(nA):
                for ib in range(nB): 
                    C[ia,ib] = np.max(signal.correlate(self.A[:,ia], self.B[:,ib], mode="full") )

            # Scale to correlation
            C = C / np.sqrt((self.A**2).sum(axis=0)[:,np.newaxis] * (self.B**2).sum(axis=0))
                    
        elif measure == "MI":
            C = self._calculate_mutual_information(self.A, self.B)
        elif measure == "NMI":
            C = 2*self._calculate_mutual_information(self.A, self.B) / (
                self._calculate_mutual_information(self.A, self.A) +
                self._calculate_mutual_information(self.B, self.B)
            )
        elif measure == "Amari":
            pass  # TODO:FIXME: Does amari only work when nA==nB?
        elif measure == "RVcoeff":  # TODO:FIXME: Does RV only work when nA==nB
            AB = ((self.A.T*self.B) * (self.B.T * self.A)).sum()
            AA = (self.A.T*self.A)**2
            BB = (self.B.T*self.B)**2
            rv = AB/(np.sqrt(AA)*np.sqrt(BB))
            C = rv
        else:
            C = np.nan
           
        return C
    
    def _calculate_mutual_information(A,B):
            P = A.T * B
            PAB = P/P.sum()
            PAPB = PAB.sum(axis=1) * PAB.sum(axis=0)
            ind = np.where(PAB>0)
            MI = sum(PAB(ind) * np.log(PAB(ind)/PAPB(ind)))
            return MI
    
    def match(self, measure="corr", type="greedy"):
        
        C = self.compare(measure=measure)
        if np.any(C[:]<0):
            C = np.abs(C)
            #TODO: warning
            
        
        if type == "greedy":
            
            n_max = C.shape[0]
            n_max = np.min(C.shape)
            more_est_comp = C.shape[0] <= C.shape[1]
            
            
            idx = np.zeros((n_max,1), dtype=int)
            matched_measure = np.zeros((n_max,1))
            
            
            for d in range(n_max):
                # Find the value and column index of the maximum value
                tmatch = int(np.argmax(np.max(C,axis=0)))
                # print(tmatch)
                # Find the estimated components which produced this match
                test = int(np.argmax(C[:,tmatch]))
                # print(test)
                # save it
                if more_est_comp:
                    idx[test] = tmatch
                    matched_measure[test] = C[test, tmatch]
                else:
                    idx[tmatch] = test
                    matched_measure[tmatch] = C[test, tmatch]
                    
                # Set this component as matched
                C[:,tmatch] = 0
                C[test,:] = 0
            
            return idx, matched_measure
        elif type == 'exact':
            D1, D2 = C.shape

            '''print(f"D1={D1}, D2={D2}")
            if D1 == D2:
                pass
            elif D1 > D2:
                print("D1>D2")
            else:
                print("D1<D2")'''

            n_perm = math.factorial(np.max(C.shape))
            if np.max(C.shape) >= 11:
                print('Exact matching was not performed, as it would be too computationally demanding. The number of permutations scale as factorial(n).')
                for i in range(20):
                    print(f"n={i}\t #perm.={math.factorial(i)}")
                raise ValueError   # TODO: Should provide a more informative error

            # p_aucs = np.zeros((n_perm,1))
            j=0
            ms_best = -np.inf
            idx_best = []
            for p in multiset_permutations(range(np.max(C.shape)),int(np.min(C.shape))):
                
                if D2 >= D1:
                    #print([(i,p[i]) for i in range(D1)])
                    #print([C[i,p[i]] for i in range(D1)])
                    ms_score = [C[i,p[i]] for i in range(D1)]
                else:
                    #print([(p[i],i) for i in range(D2)])
                    #print([C[p[i],i] for i in range(D2)])
                    ms_score = [C[p[i],i] for i in range(D2)]  
                # p_aucs[j] = ms_score
                if np.sum(ms_score) > np.sum(ms_best):
                    ms_best = ms_score
                    idx_best = p
                j+=1
                
            # idx = np.argmax(p_aucs)
            return idx_best, ms_best
        
            




if __name__ == "__main__":
    print('Demo of Latent Comparison')
    # %%
    Atrue = np.random.randn(10,4)*3
    Aest = Atrue[:,[0,2,3,1]] + np.random.randn(10,4)
    mdl = LatentModelComparer(Atrue, Aest)
    
    # %% Compare
    measure = "corr"


    # %% Exact and greedy match 
    
    print('Eact and greedy match')
    print(mdl.compare("corr"))
    print(mdl.compare("crosscorr"))

    # print(mdl.match(type="greedy"))
    print(mdl.match(type="exact"))
    
    mdl = LatentModelComparer(Atrue, Aest[:,[0,3,1,2]])
    print(mdl.compare("corr"))
    
    # %% Fewer estimated than true components
    Atrue = np.random.randn(10,4)*3
    Aest = Atrue[:,[3,2]] + np.random.randn(10,2)
    mdl = LatentModelComparer(Atrue, Aest)
    print(mdl.match(type="greedy"))
    print(mdl.match(type="exact"))
    
     # %% Fewer true than estimated components
    Atrue = np.random.randn(10,4)*3
    Aest = Atrue + np.random.randn(10,4)
    Atrue = Atrue[:,[3,1]]
    mdl = LatentModelComparer(Atrue, Aest)
    print(mdl.match(type="greedy"))
    print(mdl.match(type="exact"))
    
# %%
