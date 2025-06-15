function gfl_pqn(datafile, resultfile, rho, mu, k)
    % gfl_pqn: MATLAB implementation for solving the Generalized Fused Lasso (GFL) problem
    % using the PQN optimization framework.
    % 
    % Inputs:
    %   datafile   - Path to the .mat file containing X, y, and L
    %   resultfile - Path to save the result
    %   rho        - Regularization parameter
    %   mu         - Smoothness parameter
    
    % Load data
    data = load(datafile);
    X = data.X;
    y = data.y;
    L = data.L; % Laplacian matrix

    % Ensure y is a column vector
    if size(y, 2) > 1
        y = y(:);
    end

    % Set initial solution
    [n, d] = size(X);
    u_init = zeros(d, 1);

    % Define the objective function
    funObj = @(u) GeneralizedFusedLasso(u, X, y, rho, L, mu);

    % Define the projection function
    funProj = @(u) ProjGeneralizedFusedLassoGurobi(u, k, d);

    % Optimization options
    options.verbose = 0; % Verbosity level
    options.optTol = 1e-6; % Optimality tolerance
    options.maxIter = 1000; % Maximum iterations
    options.SPGiters = 100;

    % Solve the optimization problem using minConf_PQN
    [u_opt, fval, funEvals] = minConF_PQN(funObj, u_init, funProj, options);

    % Save the results
    beta = u_opt; % Optimal solution
    funcVal = fval; % Objective value
    save(resultfile, 'beta', 'funcVal', 'funEvals');
end



function [f, g] = GeneralizedFusedLasso(u, X, y, rho, L, mu)
    % GeneralizedFusedLasso: Computes the objective function value and gradient
    % Inputs:
    %   u   - Current solution vector (d x 1)
    %   X   - Feature matrix (n x d)
    %   y   - Target vector (n x 1)
    %   rho - Regularization parameter
    %   L   - Laplacian matrix (d x d)
    %   mu  - Smoothness parameter
    %
    % Outputs:
    %   f   - Objective function value (scalar)
    %   g   - Gradient vector (d x 1)

    % Dimensions
    [n, d] = size(X);

    % Create diagonal matrix from u
    D_u = spdiags(u, 0, d, d);

    % Compute M = inv((1/rho) * X * D_u * X' + I)
    M = inv((1 / rho) * (X * D_u * X') + eye(n));

    % Compute objective value
    f = (1 / 2) * y' * M * y + mu * u' * L * u;

    % Compute gradient
    g_loss = -(1 / (2 * rho)) * ((X' * M * y).^2); % Gradient of loss
    g_smooth = 2 * mu * L * u;                     % Gradient of smoothness
    g = g_loss + g_smooth;
end

function u_proj = ProjGeneralizedFusedLassoGurobi(u, k, d)
    % ProjGeneralizedFusedLassoGurobi: Solves the projection problem for GFL
    % Inputs:
    %   u - Current solution vector (d x 1)
    %   k - Sparsity level (scalar)
    %   d - Dimensionality of the problem (scalar)
    %
    % Output:
    %   u_proj - Projected solution vector (d x 1)

    % Create optimization model
    model.modelname = 'GeneralizedFusedLasso';
    model.modelsense = 'min';

    % Objective function: minimize (1/2) x'Qx + f'x
    Q = speye(d);                      % Quadratic term (identity matrix)
    f = -2 * u;                        % Linear term
    model.Q = sparse(Q);
    model.obj = f;

    % Constraints
    model.A = sparse(ones(1, d));      % Sum of entries constraint
    model.rhs = k;                     % Right-hand side for constraint
    model.sense = '<';                 % Less-than constraint
    model.lb = zeros(d, 1);            % Lower bound (non-negative entries)
    model.ub = ones(d, 1);             % Upper bound (entries <= 1)

    % Gurobi parameters
    params.OutputFlag = 0;             % Suppress Gurobi output
    params.IterationLimit = 500;       % Iteration limit

    % Solve the optimization problem
    result = gurobi(model, params);

    % Extract solution
    if strcmp(result.status, 'OPTIMAL')
        u_proj = result.x;
    else
        error('Projection optimization did not converge!');
    end
end
