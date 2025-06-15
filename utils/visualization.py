import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def support_recovery_analysis(accuracy, sample_sizes=None, file_name=None):
    if sample_sizes is None:
        sample_sizes = np.arange(50, 300, 100)

    # automatically detect model names from the first entry
    model_names = list(accuracy[0].keys())  
    acc_dict = {model: [] for model in model_names}
    ci_dict = {model: [] for model in model_names}

    for i in range(len(accuracy)):
        for model in model_names:
            values = accuracy[i][model]
            n = len(values)
            if n > 1:
                std = np.std(values, ddof=1)
                sem = std / np.sqrt(n)
                ci = t.ppf(0.975, df=n-1) * sem  # 95% CI
            else: 
                # no confidence interval if n <= 1
                std = 0
                sem = 0
                ci = 0  

            acc_dict[model].append(np.mean(values))
            ci_dict[model].append(ci)

    plt.figure(figsize=(6, 5))

    for model in model_names:
        plt.errorbar(
            sample_sizes[:len(accuracy)], 
            acc_dict[model], 
            yerr=ci_dict[model], 
            fmt='o-', 
            label=model, 
            capsize=5
        )

    plt.xlabel('Sample Size')
    plt.ylabel('Accuracy of Support Recovery')
    plt.grid(True)
    plt.legend(loc='lower right')  # fix the legend position
    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

    return acc_dict



def plot_out_of_sample_mse(mse_results, true_k, file_name=None):
    """
    Plots out-of-sample MSE vs k with error bars.
    
    Parameters:
    - mse_results: Dictionary {k: [list of MSEs]} with results from `out_of_sample()`.
    - true_k: The actual number of contributing features (to mark with a vertical line).
    """
    
    plt.figure(figsize=(6, 5))
    
    k_vals = sorted(mse_results.keys())  # Get sorted k values
    mse_means = [np.mean(mse_results[k]) for k in k_vals]  # Mean MSE for each k
    mse_stds = [np.std(mse_results[k]) for k in k_vals]  # Standard deviation
    k_vals = [k for k in k_vals]
    
    # Plot MSE with error bars
    plt.errorbar(k_vals, mse_means, yerr=mse_stds, fmt='o-', capsize=4, linestyle='--', color='r', label="Estimated MSE")

    # Add vertical black line at true k
    plt.axvline(x=true_k, color='k', linestyle='-', linewidth=2, label="True k")

    plt.xlabel(r'$k$', fontsize=14)
    plt.ylabel("Out of Sample MSE", fontsize=14)
    plt.legend()
    plt.grid(True)
    if file_name:
        plt.savefig(f'{file_name}.pdf', bbox_inches='tight')
    plt.show()