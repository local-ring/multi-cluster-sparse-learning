function gfl_multi_new_inv(datafile, resultfile, rho, mu, k)
% gfl_multi_new_inv:
%   Minimizes: y' * (I + (1/rho) X diag(M*s) X')^{-1} * y + 0.5 * tr(M' L M)
%   subject to (u,M,s) in \tilde{\Gamma} (box, row-stochastic M, cardinality on u,
%   and u_i >= m_{ij} + s_j - 1).
%
% Inputs:
%   datafile   - .mat file with X (n x d), y (n x 1), L (d x d, typically PSD)
%   resultfile - .mat to save M, s, u, funcVal, funEvals
%   rho        - > 0
%   L_unused_mu - kept for interface compatibility (unused; weight = 0.5 in obj)
%   k          - number of columns in M and cardinality cap in \tilde{\Gamma}
%
% Requires: minConF_PQN, Gurobi MATLAB API

    S = load(datafile);
    X = S.X;              % n x d
    y = S.y(:);           % n x 1
    L = S.L;              % d x d
    [n, d] = size(X);

    % ----- initialization (feasible after projection) -----
    M0 = ones(d, k) / k;       % row-stochastic guess
    s0 = 0.5 * ones(k,1);      % inside [0,1]
    u0 = zeros(d,1);           % inside [0,1]
    z0 = [M0(:); s0; u0];

    % ----- objective (smooth) and projection (Gurobi) -----
    funObj  = @(z) Objective_MS_inv(z, X, y, rho, mu, L, d, k);
    funProj = @(z) Project_UMS_Gurobi(z, d, k);

    % ----- PQN options -----
    options.verbose  = 0;
    options.optTol   = 1e-6;
    options.maxIter  = 500;
    options.SPGiters = 100;

    % ----- solve -----
    [z_opt, fval, evals] = minConF_PQN(funObj, z0, funProj, options);

    % unpack
    M_opt = reshape(z_opt(1:d*k), [d, k]);
    s_opt = z_opt(d*k + (1:k));
    u_opt = z_opt(d*k + k + (1:d));

    % save
    M = M_opt; s = s_opt; u = u_opt;              %#ok<NASGU>
    funcVal = fval; funEvals = evals;             %#ok<NASGU>
    save(resultfile, 'M', 's', 'u', 'funcVal', 'funEvals');
end


function [f, g] = Objective_MS_inv(z, X, y, rho, mu, L, d, k)
% Objective_MS_inv:
%   f(M,s) = y'*(I + (1/rho) X diag(M*s) X')^{-1} y + 0.5 * tr(M' L M)
%   z = [vec(M); s; u], but u does not enter f; gradient wrt u is zero.

    % unpack
    M = reshape(z(1:d*k), [d, k]);
    s = z(d*k + (1:k));
    % u = z(d*k + k + (1:d));  % present but not used in f

    % v = M s
    v = M * s;                        % d x 1

    % Build A = I + (1/rho) X diag(v) X' without forming diag explicitly
    % XvX' = X * diag(v) * X' = (X .* repmat(v',n,1)) * X'
    [n, ~] = size(X);
    Xv = bsxfun(@times, X, v');       % n x d
    A  = eye(n) + (1/rho) * (Xv * X');

    % Solve A z = y  (SPD; use Cholesky)
    % f1 = y' * A^{-1} * y
    [R, pflag] = chol(A, 'lower');
    if pflag ~= 0
        % fallback to backslash if numerical issues
        zsol = A \ y;
    else
        w = R \ y;
        zsol = R' \ w;
    end

    f1 = y' * zsol;
    f2 = 0.5 * mu * trace(M' * L * M);
    f  = f1 + f2;

    % Gradient wrt v_i: g_v(i) = -(1/rho) * (x_i' zsol)^2
    t    = X' * zsol;                 % d x 1
    g_v  = -(1/rho) * (t.^2);         % d x 1

    % Chain rule to M and s
    gM = g_v * (s.');                 % d x k
    gM = gM + mu * L * M;                  % add quadratic term

    gs = M' * g_v;                    % k x 1

    gu = zeros(d,1);                  % u does not enter f

    % pack gradient
    g = [gM(:); gs; gu];
end


function z_proj = Project_UMS_Gurobi(z0, d, k)
% Euclidean projection of z=[vec(M); s; u] onto:
%   0<=M<=1,  sum_j M(i,j)=1  (i=1..d)
%   0<=s<=1
%   0<=u<=1,  sum_i u_i <= k           (convex surrogate of ||u||_0<=k)
%   u_i >= m_ij + s_j - 1   for all i,j
%
% INPUT:  z0 length d*k + k + d (pack of M, s, u)
% OUTPUT: z_proj of same length

    nM = d*k; ns = k; nu = d;
    n_core = nM + ns + nu;

    z0 = z0(:);
    if numel(z0) ~= n_core
        error('z0 must have length d*k + k + d.');
    end

    model.modelname  = 'Proj_tildeGamma_L1Cap';
    model.modelsense = 'min';

    % Objective: 0.5 * ||[M;s;u] - z0||^2
    model.Q   = speye(n_core);
    model.obj = -2 * z0;   % 0.5 x'Qx + obj'x

    % Bounds: 0 <= variables <= 1
    model.lb = zeros(n_core,1);
    model.ub = ones(n_core,1);

    % All variables continuous
    model.vtype = repmat('C', n_core, 1);

    % Indices
    idxM = 1:nM;
    idxs = nM + (1:ns);
    idxu = nM + ns + (1:nu);

    % ----- constraints -----
    rows = {}; rhs = []; sense = [];

    % (a) Row-stochastic M: sum_j M(i,j) = 1
    for i = 1:d
        row = sparse(1, n_core);
        idx = (0:(k-1))*d + i;      % positions of M(i,1..k) in vec(M)
        row(1, idxM(idx)) = 1;
        rows{end+1,1} = row; rhs(end+1,1) = 1; sense(end+1,1) = '=';
    end

    % (b) Sum cap on u: sum_i u_i <= k
    row = sparse(1, n_core);
    row(1, idxu) = 1;
    rows{end+1,1} = row; rhs(end+1,1) = k; sense(end+1,1) = '<';

    % (c) Coupling: u_i >= m_ij + s_j - 1    for all i,j
    for j = 1:k
        mcol = (j-1)*d + (1:d);  % column j of M in vec(M)
        for i = 1:d
            row = sparse(1, n_core);
            row(1, idxu(i))         = -1;
            row(1, idxM(mcol(i)))   = 1;
            row(1, idxs(j))         = 1;
            % -u_i + m_ij + s_j <= 1
            rows{end+1,1} = row; rhs(end+1,1) = 1; sense(end+1,1) = '<';
        end
    end

    % Stack
    model.A     = vertcat(rows{:});
    model.rhs   = rhs;
    model.sense = sense;

    % Solve
    params.OutputFlag = 0;
    result = gurobi(model, params);

    if ~isfield(result,'x') || ~strcmp(result.status,'OPTIMAL')
        error('Gurobi projection failed: status=%s', result.status);
    end

    z_proj = result.x;  % returns [vec(M); s; u]
end
