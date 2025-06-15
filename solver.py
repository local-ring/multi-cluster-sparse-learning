# === Standard Library ===
import os, time, sys
import numpy as np
from collections import defaultdict

from solvers.adaptive_grace import adaptive_grace
from solvers.lasso import lasso
from solvers.gfl_pqn import gfl_pqn
from solvers.gfl_proximal import gfl_proximal
from utils.communication import A_to_edges

# need to import the sparse_module.so in ./src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from solvers.signal_family import sparse_learning_solver

class Solver:    
    def __init__(self, models, c=1):   
        
        self.res = defaultdict(list)
        self.models = models
        self.n = None
        self.d = None
        self.k = None
        self.c = c
        # self.datafile = os.path.join(os.path.abspath(datafile), f'real_data')
        # self.resultfile = os.path.join(os.path.abspath(resultfile), f'real_data') # matlab does not like relative path
        # self.datafile_pqn = os.path.join(os.path.abspath('./src/PQN/data/'), f'real_data')
        # self.resultfile_pqn = os.path.join(os.path.abspath('./src/PQN/result/'), f'real_data')
        self.datafile = os.path.abspath('./data/data_gfl/')
        self.resultfile = os.path.abspath('./data/result_gfl/') 
        self.datafile_pqn = os.path.abspath('./data/data_PQN/')
        self.resultfile_pqn = os.path.abspath('./data/result_PQN/')
        self._init(self.datafile, self.resultfile)
        self._init(self.datafile_pqn, self.resultfile_pqn)


    def _init(self, datafile, resultfile):
        if not os.path.exists(datafile):
            os.makedirs(datafile)
        if not os.path.exists(resultfile):
            os.makedirs(resultfile)

    def _solver_lasso(self, X, y, alpha=0.1):
        return lasso(X, y, alpha=alpha)

    def _solver_adaptive_grace(self, X, y, W, lambda1=1.0, lambda2=1.0, max_iter=1000, tol=1e-4):
        return adaptive_grace(X, y, W, lambda1=lambda1, lambda2=lambda2, max_iter=max_iter, tol=tol)
    
    def _solver_gfl_proximal(self, X, y, A, i, rho1=0.5, rho2=0.5):
        return gfl_proximal(X, y, A, i, datafile=self.datafile, resultfile=self.resultfile, rho1=rho1, rho2=rho2)
    
    def _solver_gfl_pqn(self, X, y, L, i, k, rho=None, mu=0.01):
        if rho is None:
            return gfl_pqn(X, y, L, i, rho=np.sqrt(self.n), mu=mu, k=k, datafile=self.datafile_pqn, resultfile=self.resultfile_pqn)
        else:
            return gfl_pqn(X, y, L, i, rho=rho, mu=mu, k=k, datafile=self.datafile_pqn, resultfile=self.resultfile_pqn)

    def _solver_signal_family(self, X, y, i, s, c=1, g=1, max_epochs=50, tol_algo=1e-20, step=1, edges=None, costs=None,):
        # s is the number of sparsity level, w is x_star in their codecase, gamma=0.5 control the noise
        return sparse_learning_solver((i, X, y, edges, costs, s, g, max_epochs, tol_algo, step, c))
   

    def _single_runtime(self, model, X, y, k, c=1, L=None, A=None, i=None, rho=15.0, mu=0.1, rho1=0.5, rho2=0.5):
        start_time = time.time()
        self.solver(model, X, y, k, c=c, L=L, A=A, i=i, rho=rho, mu=mu, rho1=rho1, rho2=rho2)
        end_time = time.time()
        return end_time - start_time


    def solver(self, model, X, y, k, c=1, L=None, A=None, i=None, rho=15.0, mu=0.1, rho1=0.5, rho2=0.5):
        if model == "lasso":
            return self._solver_lasso(X, y)
        elif model == "adaptive_grace":
            return self._solver_adaptive_grace(X, y, L)
        elif model == "gfl_proximal":
            return self._solver_gfl_proximal(X, y, A, i, rho1=rho1, rho2=rho2)
        elif model == "gfl_pqn":
            return self._solver_gfl_pqn(X, y, L, i, k, rho=rho, mu=mu)
        elif model == "signal_family":
            edges, costs = A_to_edges(A)
            return self._solver_signal_family(X, y, i=i, s=k, c=c, edges=edges, costs=costs)
        else:   
            raise ValueError(f"Unknown model: {model}. Supported models are: lasso, adaptive_grace, gfl_proximal, gfl_pqn, signal_family.")
        

    def fit(self, X, y, L, A, k, i=None, verbose=False):
        self.n, self.d = X.shape
        self.k = k
        for model in self.models:
            if model == "signal_family":
                u = self.solver(model, X, y, k, c=self.c, L=L, A=A, i=i)
                # unpack the result
                _, results = u
                for method, (x_hat, *_) in results.items():
                    if verbose:
                        print(f"Running {method}")
                    self.res[method] = x_hat
            else:
                if verbose:
                    print(f"Running {model}")
                u = self.solver(model, X, y, k, L=L, A=A, i=i)
                self.res[model] = u

        return self.res

