import networkx as nx
import numpy as np

def generate_cluster_sizes(num_selected_clusters, 
                           num_non_selected_clusters, 
                           num_selected_features, 
                           num_non_selected_features,):
    # we take the first num_selected_features as selected features
    # and the rest as non-selected features
    # return the list of sizes of selected clusters and non-selected clusters
    selected_clusters = []
    non_selected_clusters = []
    remaining_selected_features = num_selected_features
    remaining_non_selected_features = num_non_selected_features
    for i in range(num_selected_clusters):  
        if i == num_selected_clusters - 1:
            size = remaining_selected_features
        else:
            size = np.random.randint(1, remaining_selected_features - (num_selected_clusters - i - 1) + 1)
        selected_clusters.append(size)
        remaining_selected_features -= size
    for i in range(num_non_selected_clusters):
        if i == num_non_selected_clusters - 1:
            size = remaining_non_selected_features
        else:
            size = np.random.randint(1, remaining_non_selected_features - (num_non_selected_clusters - i - 1) + 1)
        non_selected_clusters.append(size)
        remaining_non_selected_features -= size
    return selected_clusters, non_selected_clusters

def generate_multi_way_graph(selected_clusters, non_selected_clusters, p, q):
    """
    Generates a multi-way graph based on the selected and non-selected clusters.

    Parameters:
    selected_clusters (list): List of sizes of selected clusters.
    non_selected_clusters (list): List of sizes of non-selected clusters.
    p (float): Probability for intra-cluster edge creation.
    q (float): Probability for inter-cluster edge creation.

    Returns:
    A (np.ndarray): Adjacency matrix representing the multi-way graph.
    L (np.ndarray): Laplacian matrix of the graph.
    """

    cluster_sizes = selected_clusters + non_selected_clusters
    total_nodes = sum(cluster_sizes)
    cluster_labels = []

    # Assign each node a cluster label
    label = 0
    for size in cluster_sizes:
        cluster_labels.extend([label] * size)
        label += 1

    cluster_labels = np.array(cluster_labels)
    A = np.zeros((total_nodes, total_nodes))

    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            same_cluster = cluster_labels[i] == cluster_labels[j]
            prob = p if same_cluster else q
            if np.random.rand() < prob:
                A[i, j] = A[j, i] = 1

    G = nx.from_numpy_array(A)
    L = nx.laplacian_matrix(G).toarray()

    return A, L

def generate_w(selected_clusters, non_selected_clusters):
    d = sum(selected_clusters) + sum(non_selected_clusters)
    k = sum(selected_clusters)
    w = np.zeros(d)
    start_index = 0
    for size in selected_clusters:
        cluster_weight = 1/k
        random_sign = np.random.choice([-1, 1])
        cluster_weight *= random_sign
        w[start_index:start_index + size] = cluster_weight
        start_index += size

    for size in non_selected_clusters:
        w[start_index:start_index + size] = 0
        start_index += size   
    return w

def generate_X(n, d):
    # generate the random design matrix X with shape (n, d)
    return np.random.normal(0, 1, (n, d))

def generate_y(X, w, gamma=0.01):
    signal = X @ w
    noise = np.random.normal(0, gamma, signal.shape)
    y = signal + noise
    return y


import matlab.engine
import os
from utils.communication import save_data, read_result_gfl_multi
def call_matlab(datafile, resultfile, rho, mu, k=None, l=None):
    eng = matlab.engine.start_matlab()
    try:
        eng.cd(os.path.abspath('./src/PQN/'))
        eng.addpath(os.path.abspath('./src/PQN/'))
        eng.addpath(eng.genpath(os.path.abspath('./src/PQN/')))
        eng.addpath(eng.genpath(os.path.abspath('./src/PQN/minConF/')))
        eng.gfl_multi_new_inv(datafile, resultfile, rho, mu, float(k), float(l), nargout=0)
    finally:
        eng.quit()

def gfl_multi(X, y, L, i, k=None, l=None, rho=None, mu=0.01, datafile=None, resultfile=None):
    datafile_name = os.path.join(datafile, f'data_{i}.mat')
    resultfile_name = os.path.join(resultfile, f'result_{i}.mat')
    save_data(X=X, y=y, L=L, filename=datafile_name)
    call_matlab(datafile_name, resultfile_name, rho, mu, float(k), float(l))
    M, s, u, _ = read_result_gfl_multi(resultfile_name)
    return M, s, u

def call_matlab_same(datafile, resultfile, rho, mu, k=None, l=None):
    eng = matlab.engine.start_matlab()
    try:
        eng.cd(os.path.abspath('./src/PQN/'))
        eng.addpath(os.path.abspath('./src/PQN/'))
        eng.addpath(eng.genpath(os.path.abspath('./src/PQN/')))
        eng.addpath(eng.genpath(os.path.abspath('./src/PQN/minConF/')))
        eng.gfl_multi_new_inv_same(datafile, resultfile, rho, mu, float(k), float(l), nargout=0)
    finally:
        eng.quit()

def gfl_multi_same_weights(X, y, L, i, k=None, l=None, rho=None, mu=0.01, datafile=None, resultfile=None):
    datafile_name = os.path.join(datafile, f'data_{i}.mat')
    resultfile_name = os.path.join(resultfile, f'result_{i}.mat')
    save_data(X=X, y=y, L=L, filename=datafile_name)
    call_matlab_same(datafile_name, resultfile_name, rho, mu, float(k), float(l))
    M, s, u, _ = read_result_gfl_multi(resultfile_name)
    return M, s, u

if __name__ == "__main__":
    # get the argments from command line which methods we want to test
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "gfl_multi"
    print(f"Testing method: {method}")
    np.random.seed(42)
    p, q = 0.95, 0.01
    n = 150
    rho, mu = 15.0, 0.1
    num_selected_clusters, num_non_selected_clusters = 2, 5
    num_selected_features, num_non_selected_features = 10, 50
    d = num_selected_features + num_non_selected_features
    k = num_selected_features
    l = num_selected_clusters + num_non_selected_clusters
    selected_clusters, non_selected_clusters = generate_cluster_sizes(num_selected_clusters, num_non_selected_clusters, num_selected_features, num_non_selected_features)
    print("Selected clusters sizes:", selected_clusters)
    print("Non-selected clusters sizes:", non_selected_clusters)
    A, L = generate_multi_way_graph(selected_clusters, non_selected_clusters, p, q)
    w = generate_w(selected_clusters, non_selected_clusters)
    print("True w:", w)
    X = generate_X(n, d)
    y = generate_y(X, w, gamma=0.5)
    datafile = os.path.abspath("./data/data_multi/")
    resultfile = os.path.abspath("./data/result_multi/")
    if not os.path.exists(datafile):
        os.makedirs(datafile)
    if not os.path.exists(resultfile):
        os.makedirs(resultfile)
    if method == "gfl_multi_same":
        M, s, u = gfl_multi_same_weights(X, y, L, i=0, k=k, l=l, rho=rho, mu=mu, datafile=datafile, resultfile=resultfile)
    else:
        M, s, u = gfl_multi(X, y, L, i=0, k=k, l=l, rho=rho, mu=mu, datafile=datafile, resultfile=resultfile)
    print("M shape:", M.shape)
    print("s shape:", s.shape)
    print("u shape:", u.shape)
    # print the value of M, s, u
    print("M:", M)
    print("s:", s)
    print("u:", u)
