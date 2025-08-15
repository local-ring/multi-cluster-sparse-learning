function gfl_multi(datafile, resultfile, rho, mu, k)
    % gfl_multi: Solves the matrix-assignment-based Generalized Fused Lasso
    % Inputs:
    %   datafile   - .mat file containing X, y, L
    %   resultfile - file to save results (beta, funcVal, funEvals)
    %   rho        - regularization parameter
    %   mu         - smoothness parameter
    %   d, k       - dimensions of M (d x k)

    % Load data
    data = load(datafile);
    X = data.X;
    y = data.y;
    L = data.L;

    if size(y, 2) > 1
        y = y(:);
    end

    % Initial assignment matrix as a vector (uniform doubly stochastic guess)
    [n, d] = size(X);
    m_init = ones(d * k, 1) / k;

    % Objective and projection function handles
    funObj = @(m) AssignmentObjective(m, X, y, rho, L, mu, d, k);
    funProj = @(m) ProjectToDoublyStochasticGurobi(m, d, k);

    % Optimization options
    options.verbose = 0;
    options.optTol = 1e-6;
    options.maxIter = 500;
    options.SPGiters = 100;

    % Solve
    [m_opt, fval, evals] = minConF_PQN(funObj, m_init, funProj, options);

    % Reshape to d x k matrix and save using expected variable names
    beta = reshape(m_opt, [d, k]);
    funcVal = fval;
    funEvals = evals;

    save(resultfile, 'beta', 'funcVal', 'funEvals');
end



function [f, g] = AssignmentObjective(m_vec, X, y, rho, L, mu, d, k)
    % Reshape m_vec to M
    M = reshape(m_vec, [d, k]);

    % Compute inverse term
    XMMTXT = (1 / rho) * (X * M * M' * X');
    A = XMMTXT + eye(size(X, 1));
    A_inv_y = A \ y;

    % Objective value
    f1 = 0.5 * y' * A_inv_y;
    f2 = 0.5 * mu * trace(M' * L * M);
    f = f1 + f2;

    % Gradient computation
    B = A_inv_y * A_inv_y';
    G = (1 / rho) * (X' * B * X);  % d×d matrix
    grad_M = G * M + mu * L * M;   % d×k matrix

    g = grad_M(:); % vectorize the gradient
end


function m_proj = ProjectToDoublyStochasticGurobi(m_vec, d, k)
    m = length(m_vec);
    model.modelname = 'ProjectToDoublyStochastic';
    model.modelsense = 'min';
    model.Q = sparse(eye(m));
    model.obj = -2 * m_vec;
    model.lb = zeros(m, 1);
    model.ub = ones(m, 1);

    % Constraint: sum over rows and cols == 1
    A_eq = [];
    for j = 1:k
        e = zeros(d, d * k);
        for i = 1:d
            e(i, (j - 1) * d + i) = 1;
        end
        A_eq = [A_eq; sum(e, 1)];
    end
    for i = 1:d
        idx = i:d:d * k;
        row = zeros(1, d * k);
        row(idx) = 1;
        A_eq = [A_eq; row];
    end

    model.A = sparse(A_eq);
    model.rhs = ones(d + k, 1);
    model.sense = repmat('=', d + k, 1);

    params.OutputFlag = 0;
    result = gurobi(model, params);

    if strcmp(result.status, 'OPTIMAL')
        m_proj = result.x;
    else
        error('Projection failed.');
    end
end
