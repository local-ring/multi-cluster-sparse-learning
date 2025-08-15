import scipy.io as sio
import scipy.sparse as sp
import numpy as np


# save the data to .mat file so that the matlab code of gfl_pqn or gfl_proximal can use it
def save_data(X, y, L=None, A=None, filename=None):
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if L is not None: # save the data for gfl_pqn
        data = {
            "X": X,
            "y": y,
            "L": L.toarray() if sp.issparse(L) else L,  
        }
    elif A is not None: # save the data for gfl_proximal
        data = {
            "X": X,
            "y": y,
            "AdjMat": A.toarray() if sp.issparse(A) else A,  #
        }
    sio.savemat(filename, data)


# read the result of gfl_pqn or gfl_proximal
def read_result(resultfile):
    result = sio.loadmat(resultfile)
    beta, funcVal = result['beta'], result['funcVal']
    return beta, funcVal

def read_result_gfl_multi(resultfile):
    result = sio.loadmat(resultfile)
    M, s, u, funcVal = result['M'], result['s'], result['u'], result['funcVal']
    return M, s, u, funcVal

# convert adjacency matrix to edges and costs for signal family
def A_to_edges(A):
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    A_coo = A.tocoo()
    edges = np.vstack((A_coo.row, A_coo.col)).T
    costs = A_coo.data.astype(np.float64)
    return edges, costs