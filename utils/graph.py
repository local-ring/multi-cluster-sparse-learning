import numpy as np
import scipy.sparse as sp

def generate_graph(d, k, p, q):
    # here we generate the adjacency matrix and laplacian of the graph
    selected_features = np.arange(k)
    non_selected_features = np.arange(k, d)

    # adjacency matrix
    A = sp.lil_matrix((d, d))
    
    selected_block = (np.random.rand(k, k) < p).astype(int)
    np.fill_diagonal(selected_block, 0) # no self-loop
    selected_block = np.triu(selected_block) + np.triu(selected_block, 1).T # make it symmetric
    for i, node_i in enumerate(selected_features):
        for j, node_j in enumerate(selected_features):
            A[node_i, node_j] = selected_block[i, j]

    non_selected_block = (np.random.rand(d-k, d-k) < p).astype(int)
    np.fill_diagonal(non_selected_block, 0)
    non_selected_block = np.triu(non_selected_block) + np.triu(non_selected_block, 1).T 
    for i, node_i in enumerate(non_selected_features):
        for j, node_j in enumerate(non_selected_features):
            A[node_i, node_j] = non_selected_block[i, j]


    # generate the connections between selected and non-selected features
    inter_block = (np.random.rand(k, d - k) < q).astype(int)
    for i, node_i in enumerate(selected_features):
        for j, node_j in enumerate(non_selected_features):
            A[node_i, node_j] = inter_block[i, j]
            A[node_j, node_i] = inter_block[i, j]

    # degree matrix
    D = sp.diags(np.ravel(A.sum(axis=1)))

    # laplacian matrix
    L = D - A
    
    return L, A


# get the normalized laplacian matrix from the laplacian matrix
def normalized_laplacian(L):
    D = np.array(L.diagonal())

    # handle zero degrees by replacing them with 1 (which will become 0 after inversion)
    D_sqrt_inv = np.zeros_like(D)
    mask = D > 0
    D_sqrt_inv[mask] = 1 / np.sqrt(D[mask])
    D_inv_sqrt = sp.diags(D_sqrt_inv)

    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    return L_normalized
    