function [f, g] = L0Obj(X, m, y, L, pho, mu, d, h, n)
    % Compute the objective function value and gradient for the L0 regularized least squares problem.
    %
    % Parameters:
    %   X: n x d matrix of features
    %   m: d*h vector of weights
    %   y: n x 1 vector of labels
    %   L: d x d Laplacian matrix
    %   pho: regularization parameter for L2 penalty
    %   mu: regularization parameter for L0-graph penalty
    %   d: number of features
    %   h: number of clusters
    %   n: number of samples
    %
    % Returns:
    %   f: objective function value
    %   g: gradient

    % Check dimensions
    [nX, dh] = size(X);
    if d * h ~= dh
        error('The dimensions of X and d*h do not match.');
    end

    % Ensure m is double
    m = double(m);

    % Construct diagonal sparse matrix
    SpDiag = spdiags(m(:), 0, dh, dh);

    % Ensure X and y are double
    X = double(X);
    y = double(y);
    pho = double(pho);

    if size(y, 2) > 1
        y = y(:); % Reshape to column vector
    end


    % Compute B and precision penalty
    regularization = 1e-8;  % Small regularization to ensure numerical stability
    B_inv = (1 / pho) * X * SpDiag * X' + eye(n) + regularization * eye(n);
    B = B_inv \ eye(n);  % Equivalent to inv(B_inv)
    precision_penalty = y' * B * y;

    % Compute graph penalty
    epsilon = 0.1;
    r = 1 + epsilon;

    eta = 2 * r * mu + 2 * d + 0.4 * (pho^2);
    L = double(L + r * eye(d));  % Add identity matrix scaled by r
    M = reshape(m, d, h);  % Reshape m into the assignment matrix
    graph_penalty = 0.5 * mu * trace(M' * L * M);

    % Compute correction term
    MTM = M' * M;
    correction_term = 0.5 * eta * (sum(MTM(:)) - sum(diag(MTM)));

    % Compute objective function value
    f = precision_penalty + graph_penalty + correction_term;

    % Compute gradients
    row_sums = sum(M, 2);  % Sum over columns (axis=1 in Python)
    A_grad = -(1 / pho) * ((X' * B * y).^2);  % Gradient of the first term
    B_grad = mu * (L * M);  % Gradient of the second term
    C_grad = eta * (M - row_sums);  % Gradient of the third term

    % Flatten gradients
    B_grad = B_grad(:);
    C_grad = C_grad(:);

    % Combine gradients
    g = A_grad + B_grad + C_grad;
end