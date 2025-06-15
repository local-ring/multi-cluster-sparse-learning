import numpy as np
import os, random, sys
from collections import defaultdict
from abc import ABC, abstractmethod

from solver import Solver
from utils.graph import generate_graph 
from utils.omse import compute_omse
class RandomEnsemble(ABC):
    def __init__(self, n, d, k, gamma, p=0.95, q=0.01):
        self.n , self.d, self.k, self.gamma = n, d, k, gamma
        self.p, self.q = p, q

    @abstractmethod
    def _generate_X(self):
        pass

    @abstractmethod
    def _generate_w(self):
        pass

    def _generate_y(self, X, w):
        signal = X @ w
        noise = np.random.normal(0, self.gamma, signal.shape)
        y = signal + noise
        return y

    def _generate_graph(self):
        return generate_graph(self.d, self.k, self.p, self.q)
    
    def _generate_data(self):
        L, A = self._generate_graph()
        w = self._generate_w()
        X = self._generate_X()
        y = self._generate_y(X, w)
        return L, w, X, y, A

    def _recovery_accuracy(self, u):
        # evaluate the support recovery accuracy
        selected_features_true = np.arange(self.k)
        selected_features_pred = np.argsort(np.abs(u))[-self.k:] 
        correct_pred = np.intersect1d(selected_features_true, selected_features_pred)
        accuracy = len(correct_pred) / self.k
        return accuracy
    
    def _report(self, model_accuracy):
        for model, accuracy in model_accuracy.items():
            avg_accuracy = np.mean(accuracy)
            std_accuracy = np.std(accuracy)
            print(f"Model: {model}, Avg. Accuracy: {avg_accuracy}, Std. Accuracy: {std_accuracy}")

    def runtime(self, num_replications=10,
                models=["gfl_pqn", "gfl_proximal", "lasso", "adaptive_grace", "signal_family"]):          
        runtime_results = {model: [] for model in models}  # Store runtimes

        for i in range(num_replications):
            L, w, X, y, A = self._generate_data()
            
            for model in models:
                solver = Solver(models=[model])
                single_runtime = solver._single_runtime(model, X, y, self.k, L=L, A=A, i=i)
                runtime_results[model].append(single_runtime)          
            print(f"Replication {i+1} completed.")

        # compute mean and standard deviation for each method
        runtime_summary = {model: (np.mean(times), np.std(times)) for model, times in runtime_results.items()}
        return runtime_summary

    def out_of_sample(self, k_values=np.arange(30, 100, 10), num_replications=10, model="gfl_pqn"):
        # out-of-sample MSE for proposed method
        mse_results = defaultdict(list)
        for i in range(num_replications):
            L, w, X, y, A = self._generate_data()
            omse = compute_omse(X, y, w, L, A, model, k_values, i, num_replications)
            for k in k_values:
                mse_results[k].append(omse[k])

        return mse_results
    

    def run(self, num_replications=10, models=["gfl_pqn", "gfl_proximal", "lasso", "adaptive_grace", "signal_family"]):
        model_accuracy = defaultdict(list)
        solver = Solver(models=models)
        for i in range(num_replications):
            L, w, X, y, A = self._generate_data()
            results = solver.fit(X, y, L, A, k=self.k, i=i, verbose=1)
            for model, u in results.items():
                accuracy = self._recovery_accuracy(u)
                model_accuracy[model].append(accuracy)
        
        self._report(model_accuracy)
        return model_accuracy
            
        

    
class RandomEnsembleCorrelation(RandomEnsemble):
    def __init__(self, *args, correlated_ratio=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.correlated_ratio = correlated_ratio

    def _generate_X(self):
        mean = np.zeros(self.d)
        cov = np.eye(self.d)
        correlated_ratio = self.correlated_ratio

        selected_features = np.arange(self.k)
        non_selected_features = np.arange(self.k, self.d)

        # Choose a subset of features to be correlated
        selected_correlated = random.sample(list(selected_features), int(correlated_ratio * self.k))
        non_selected_correlated = random.sample(list(non_selected_features), int(correlated_ratio * self.k))

        # Inject correlations (symmetric, high correlation)
        for i in selected_correlated:
            for j in non_selected_correlated:
                cov[i, j] = 0.9
                cov[j, i] = 0.9

        # ensure positive semi-definiteness
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        X = np.random.multivariate_normal(mean, cov_psd, size=self.n)
        return X

    def _generate_w(self):
        w = np.zeros(self.d)
        signs = np.random.choice([-1, 1], size=self.k)
        w[:self.k] = signs * (1 / np.sqrt(self.k))
        return w
class RandomEnsembleCorrelationWeight(RandomEnsembleCorrelation):
    # overide the w method
    def _generate_w(self):
        w = np.zeros(self.d)
        signs = np.random.choice([-1, 1], size=self.k)
        w[:self.k] = signs * (1 / np.sqrt(self.k))

        # add small Gaussian noise to all features
        w += np.random.normal(0, 0.01, size=self.d)

        return w
