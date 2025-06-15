import os
import sys
import time
import pickle
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from itertools import product
import pickle as pkl
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from collections import ChainMap, defaultdict
import scipy.sparse as sp

import pickle as pkl
from scipy.sparse import lil_matrix
import logging

# # ensure repo root is on PYTHONPATH
# sys.path.insert(0, os.path.dirname(__file__))


# current_script_path = os.path.abspath(__file__)
# project_root_path = os.path.abspath(os.path.join(os.path.dirname(current_script_path), '..'))
# src_path = os.path.join(project_root_path, 'src')

# if src_path not in sys.path:
#     sys.path.insert(0, src_path) # Insert at the beginning to prioritize this path


try:
    import sparse_module
    try:
        from sparse_module import wrap_head_tail_bisearch
    except ImportError:
        print('cannot find wrap_head_tail_bisearch method in sparse_module')
        sparse_module = None
        exit(0)
except ImportError:
    print('\n'.join([
        'cannot find the module: sparse_module',
        'try run: \'python setup.py build_ext --inplace\' first! ']))

import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

os.system("export OMP_NUM_THREADS=1")
os.system("export OPENBLAS_NUM_THREADS=1")
os.system("export MKL_NUM_THREADS=1")
os.system("export VECLIB_MAXIMUM_THREADS=1")
os.system("export NUMEXPR_NUM_THREADS=1")

# Detect available CPUs dynamically - AL
NUM_CPUS = multiprocessing.cpu_count()


np.random.seed(17)
root_p = 'results/'
if not os.path.exists(root_p):
    os.mkdir(root_p)


def algo_head_tail_bisearch(
        edges, x, costs, g, root, s_low, s_high, max_num_iter, verbose=0):
    """ This is the wrapper of head/tail-projection proposed in [2].
    :param edges:           edges in the graph.
    :param x:               projection vector x.
    :param costs:           edge costs in the graph.
    :param g:               the number of connected components.
    :param root:            root of subgraph. Usually, set to -1: no root.
    :param s_low:           the lower bound of the sparsity.
    :param s_high:          the upper bound of the sparsity.
    :param max_num_iter:    the maximum number of iterations used in
                            binary search procedure.
    :param verbose: print out some information.
    :return:            1.  the support of the projected vector
                        2.  the projected vector
    """
    prizes = x * x
    # length of prizes is length of x
    # to avoid too large upper bound problem.
    if s_high >= len(prizes) - 1:
        s_high = len(prizes) - 1
    re_nodes = wrap_head_tail_bisearch(
        edges, prizes, costs, g, root, s_low, s_high, max_num_iter, verbose)
    proj_w = np.zeros_like(x)
    proj_w[re_nodes[0]] = x[re_nodes[0]]
    return re_nodes[0], proj_w


def algo_graph_iht(
        x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s,
        root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    """
    :param x_mat: design matrix.
    :param y: response vector.
    :param max_epochs: maximal number of iterations for outer layer algorithm
    :param x_star: ground truth vector
    :param edges: edges in the graph
    :param g: connected component
    :param s: sparsity level
    :param gamma: to control the range of the sparsity since it cannot be the exact value
    :return:
    1. x_hat: the estimator of the vector
    """
    start_time = time.time()
    x_hat = np.copy(x0)
    xtx = np.dot(np.transpose(x_mat), x_mat)
    xty = np.dot(np.transpose(x_mat), y)

    # graph projection para
    h_low = int(len(x0) / 2)
    h_high = int(h_low * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    p = len(x0)
    beta = eigh(xtx, eigvals_only=True, subset_by_index=[p - 1, p - 1])[0]
    lr = 1. / beta
    for tt in range(max_epochs):
        num_epochs += 1
        grad = -1. * (xty - np.dot(xtx, x_hat))
        head_nodes, proj_gradient = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high,
            proj_max_num_iter, verbose)
        bt = x_hat - lr * proj_gradient
        tail_nodes, proj_bt = algo_head_tail_bisearch(
            edges, bt, costs, g, root, t_low, t_high,
            proj_max_num_iter, verbose)
        x_hat = proj_bt
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_graph_cosamp(
        x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        h_g, t_g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)

    h_low, h_high = int(2 * s), int(2 * s * (1.0 + gamma))
    t_low, t_high = int(s), int(s * (1.0 + gamma))

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = -2. * (np.dot(xtx, x_hat) - xty)  # proxy
        head_nodes, proj_grad = algo_head_tail_bisearch(
            edges, grad, costs, h_g, root,
            h_low, h_high, proj_max_num_iter, verbose)
        gamma = np.union1d(x_hat.nonzero()[0], head_nodes)
        bt = np.zeros_like(x_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_mat[:, gamma]), y)
        tail_nodes, proj_bt = algo_head_tail_bisearch(
            edges, bt, costs, t_g, root,
            t_low, t_high, proj_max_num_iter, verbose)
        x_hat = proj_bt
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_gen_mp(
        x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    h_low, h_high = int(s), int(s * (1.0 + gamma))
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)
    p = len(x0)
    beta = eigh(xtx, eigvals_only=True, subset_by_index=[p - 1, p - 1])[0]
    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = np.dot(xtx, x_hat) - xty
        dmo_nodes, proj_vec = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high, proj_max_num_iter, verbose)
        norm_vt = np.linalg.norm(proj_vec[dmo_nodes])
        vt = (-c / norm_vt) * proj_vec
        x_hat = x_hat - (np.dot(vt, grad) / beta) * vt
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
            # print(tt, loss, list_est_err[-1])
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_cosamp(x_mat, y, max_epochs, x_star, x0, tol_algo, step, s):
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    m, p = x_mat.shape

    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = -(2. / float(m)) * (np.dot(xtx, x_hat) - xty)  # proxy
        gamma = np.argsort(abs(grad))[-2 * s:]  # identify
        gamma = np.union1d(x_hat.nonzero()[0], gamma)
        bt = np.zeros_like(x_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_mat[:, gamma]), y)
        gamma = np.argsort(abs(bt))[-s:]
        x_hat = np.zeros_like(x_hat)
        x_hat[gamma] = bt[gamma]
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_dmo_acc_fw(
        x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    h_low, h_high = int(s), int(s * (1.0 + gamma))
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)
    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = np.dot(xtx, x_hat) - xty
        eta_t = 2. / (tt + 2.)
        dmo_nodes, proj_vec = algo_head_tail_bisearch(
            edges, -x_hat + grad / eta_t, costs, g, root, h_low, h_high, proj_max_num_iter, verbose)
        norm_vt = np.linalg.norm(proj_vec[dmo_nodes])

        vt = (-c / norm_vt) * proj_vec
        x_hat += eta_t * (vt - x_hat)
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_dmo_fw(
        x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    h_low, h_high = int(s), int(s * (1.0 + gamma))
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = np.dot(xtx, x_hat) - xty
        eta_t = 2. / (tt + 2.)
        dmo_nodes, proj_vec = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high, proj_max_num_iter, verbose)
        norm_vt = np.linalg.norm(proj_vec[dmo_nodes])
        vt = (-c / norm_vt) * proj_vec
        x_hat += eta_t * (vt - x_hat)
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def run_single_solver(para):
    method, img_name, trial_i, y, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c= para
    n, p = x_mat.shape
    x0 = np.zeros(p, dtype=np.float64)
    # dummy x_star here
    x_star = np.zeros(p, dtype=np.float64)

    if method == 'graph-iht':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_graph_iht(
            x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    elif method == 'graph-cosamp':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_graph_cosamp(
            x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs, h_g=g, t_g=g, s=s)
    elif method == 'dmo-fw':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_dmo_fw(
            x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    elif method == 'dmo-acc-fw':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_dmo_acc_fw(
            x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    elif method == 'cosamp':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_cosamp(
            x_mat, y, max_epochs, x_star, x0, tol_algo, step, s)
    elif method == 'gen-mp':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_gen_mp(
            x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    else:
        print('something must wrong.')
        exit()
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = 0.0, x_star, [0.0], [0.0], [0.0]
    # print('%-13s trial_%03d n: %03d w_error: %.3e num_epochs: %03d run_time: %.3e' %
    #       (method, trial_i, n, list_est_err[-1], num_epochs, list_run_time[-1]))
    return method, img_name, trial_i, list_est_err[-1], num_epochs, x_hat, list_run_time, list_loss, list_est_err

def sparse_learning_solver(para):
    """
    What we need:
    read from the para;  
    
    run_single_test( ('gen-mp', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    """
    # we need pass by the x_mat, s, g, x_star, edges, costs
    trial_i, x_mat, y, edges, costs, s, g, max_epochs, tol_algo, step, c = para
    np.random.seed(trial_i)

    # img_name, s, g, l1_norm, l2_norm, x_star = pkl.load(open(f'data/grid_img_angio.npz', 'rb'))
    img_name = 'dummy'

    results = {}

    re = run_single_solver(
        ('gen-mp', img_name, trial_i, y, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['gen-mp'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_solver(
        ('dmo-acc-fw', img_name, trial_i, y, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['dmo-acc-fw'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_solver(
        ('graph-cosamp', img_name, trial_i, y, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['graph-cosamp'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_solver(
        ('cosamp', img_name, trial_i, y, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['cosamp'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_solver(
        ('graph-iht', img_name, trial_i, y, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['graph-iht'] = [x_hat, list_run_time, list_loss, list_est_err]

    return trial_i, results