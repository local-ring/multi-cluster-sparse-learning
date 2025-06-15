import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNetCV

def adaptive_grace(X, y, W, lambda1=1.0, lambda2=1.0, max_iter=1000, tol=1e-4):
    # Standardize X and center y
    n, p = X.shape
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # avoid division by zero
    X = (X - X_mean) / X_std
    y_mean = y.mean()
    y = y - y_mean

    # Compute initial estimate beta_tilde
    if p < n:
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X, y)
        beta_tilde = lr.coef_
    else:
        enet = ElasticNetCV(l1_ratio=0.5, fit_intercept=False, cv=5, max_iter=10000)
        enet.fit(X, y)
        beta_tilde = enet.coef_

    # Construct modified Laplacian matrix Lstar
    d = W.sum(axis=1).A1 if hasattr(W, 'A1') else W.sum(axis=1)  # handle sparse matrices
    Lstar = np.zeros((p, p))
    rows, cols = W.nonzero()
    for i in range(len(rows)):
        u, v = rows[i], cols[i]
        if u >= v:
            continue  # process each edge once
        if d[u] == 0 or d[v] == 0:
            Lstar[u, v] = Lstar[v, u] = 0
        else:
            sign_u = np.sign(beta_tilde[u]) if beta_tilde[u] != 0 else 0
            sign_v = np.sign(beta_tilde[v]) if beta_tilde[v] != 0 else 0
            weight = W[u, v] if isinstance(W, np.ndarray) else W.data[i]
            Lstar_uv = -sign_u * sign_v * weight / np.sqrt(d[u] * d[v])
            Lstar[u, v] = Lstar_uv
            Lstar[v, u] = Lstar_uv
    np.fill_diagonal(Lstar, 1 * (d > 0))  # set diagonal to 1 if degree > 0

    # Precompute adjacency list
    adjacency_list = [[] for _ in range(p)]
    for u, v in zip(rows, cols):
        if u != v:
            adjacency_list[u].append(v)

    # Initialize beta and residual
    beta = np.zeros(p)
    residual = y.copy()
    prev_beta = np.inf * np.ones(p)
    iter = 0

    # Coordinate descent
    while iter < max_iter and np.linalg.norm(beta - prev_beta) > tol:
        prev_beta = beta.copy()
        for u in range(p):
            xu = X[:, u]
            current_beta_u = beta[u]

            # Compute xuTr and neighbor_sum
            xuTr = xu @ residual
            xuTr_plus = xuTr + n * current_beta_u  # since xu.T @ xu = n

            neighbor_sum = 0
            for v in adjacency_list[u]:
                neighbor_sum += Lstar[u, v] * beta[v]

            # Update beta_u
            z = (xuTr_plus - lambda2 * neighbor_sum) / (n + lambda2)
            threshold = lambda1 / (2 * (n + lambda2))
            beta_u_new = np.sign(z) * max(abs(z) - threshold, 0)

            # Update residual and beta
            delta = beta_u_new - current_beta_u
            residual -= xu * delta
            beta[u] = beta_u_new

        iter += 1

    return beta