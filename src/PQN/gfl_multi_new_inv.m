function gfl_multi_new_inv(datafile, resultfile, rho, mu, k, l)
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
%   l          - number of clusters the model assume
%
% Requires: minConF_PQN, Gurobi MATLAB API

    S = load(datafile);
    X = S.X;              % n x d
    y = S.y(:);           % n x 1
    L = S.L;              % d x d
    [n, d] = size(X);

    % ----- initialization (feasible after projection) -----
    M0 = ones(d, l) / l;       % row-stochastic guess
    s0 = 0.5 * ones(l,1);      % inside [0,1]
    u0 = zeros(d,1);           % inside [0,1]
    z0 = [M0(:); s0; u0];

    % ----- objective (smooth) and projection (Gurobi) -----
    funObj  = @(z) Objective_MS_inv(z, X, y, rho, mu, L, d, k, l);
    funProj = @(z) Project_UMS_Gurobi(z, d, k, l);

    % ----- PQN options -----
    options.verbose  = 0;
    options.optTol   = 1e-6;
    options.maxIter  = 1000;
    options.SPGiters = 100;

    % ----- solve -----
    [z_opt, fval, evals] = minConF_PQN(funObj, z0, funProj, options);

    % unpack
    M_opt = reshape(z_opt(1:d*l), [d, l]);
    s_opt = z_opt(d*l + (1:l));
    u_opt = z_opt(d*l + l + (1:d));

    % save
    M = M_opt; s = s_opt; u = u_opt;              %#ok<NASGU>
    funcVal = fval; funEvals = evals;             %#ok<NASGU>
    save(resultfile, 'M', 's', 'u', 'funcVal', 'funEvals');
end

function [f, g] = Objective_MS_inv(z, X, y, rho, mu, L, d, k, l)
% Objective_MS_inv:
%   f(M,s) = y'*(I + (1/rho) X diag(M*s) X')^{-1} y + 0.5 * tr(M' L M)
%   z = [vec(M); s; u], but u does not enter f; gradient wrt u is zero.

    % unpack
    M = reshape(z(1:d*l), [d, l]);
    s = z(d*l + (1:l));
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


function z_proj = Project_UMS_Gurobi(z0, d, k, l)
% Euclidean projection of z=[vec(M); s; u] onto:
%   0<=M<=1,  sum_j M(i,j)=1  (i=1..d)
%   0<=s<=1
%   0<=u<=1,  sum_i u_i <= k
%   u_i >= m_ij + s_j - 1   for all i,j
%
% INPUT:  z0 length d*l + l + d (pack of M, s, u)
% OUTPUT: z_proj of same length

    nM = d*l; ns = l; nu = d;
    n_core = nM + ns + nu;

    z0 = z0(:);
    if numel(z0) ~= n_core
        error('z0 must have length d*l + l + d.');
    end

    model.modelname  = 'Project_UMS_Gurobi';
    model.modelsense = 'min';

    % Quadratic projection objective: 0.5*||x - z0||^2
    model.Q   = speye(n_core);
    model.obj = -2 * z0;        % constant term omitted

    % Bounds: 0 <= variables <= 1
    model.lb = zeros(n_core,1);
    model.ub = ones(n_core,1);

    % All variables continuous
    model.vtype = repmat('C', n_core, 1);

    % Indices
    idxM = 1:nM;
    idxs = nM + (1:ns);
    idxu = nM + ns + (1:nu);

    % ----- constraints assembly -----
    rows = {}; 
    rhs  = [];
    sense = '';   % IMPORTANT: start as empty CHAR row, not numeric

    % (a) Row-stochastic M: sum_j M(i,j) = 1   (d constraints, '=')
    for i = 1:d
        row = sparse(1, n_core);
        idx = (0:(l-1))*d + i;            % positions of M(i,1..l) in vec(M)
        row(1, idxM(idx)) = 1;
        rows{end+1,1} = row; 
        rhs(end+1,1)  = 1;
        sense = [sense '='];
    end

    % (b) Sum cap on u: sum_i u_i <= k   (1 constraint, '<')
    row = sparse(1, n_core);
    row(1, idxu) = 1;
    rows{end+1,1} = row; 
    rhs(end+1,1)  = k; 
    sense = [sense '<'];

    % (c) Coupling: u_i >= m_ij + s_j - 1  for all i,j
    %     equivalently: -u_i + m_ij + s_j <= 1   (d*l constraints, '<')
    for j = 1:l
        mcol = (j-1)*d + (1:d);           % column j of M in vec(M)
        for i = 1:d
            row = sparse(1, n_core);
            row(1, idxu(i))        = -1;
            row(1, idxM(mcol(i)))  = 1;
            row(1, idxs(j))        = 1;
            rows{end+1,1} = row; 
            rhs(end+1,1)  = 1; 
            sense = [sense '<'];
        end
    end

    % Stack
    Astack = vertcat(rows{:});
    if ~issparse(Astack), Astack = sparse(Astack); end
    model.A   = Astack;
    model.rhs = rhs(:);

    % Finalize senses as 1×m char row
    model.sense = reshape(sense, 1, []);

    % ---- sanity checks ----
    m_expected = d + 1 + d*l;
    m = size(model.A, 1);
    assert(m == m_expected, 'A has %d rows; expected %d (= d + 1 + d*l).', m, m_expected);
    assert(numel(model.rhs) == m, 'rhs length %d != #rows(A) %d', numel(model.rhs), m);
    assert(ischar(model.sense) && isrow(model.sense) && numel(model.sense) == m, ...
        'model.sense must be 1x%d char; got class=%s size=%s', ...
        m, class(model.sense), mat2str(size(model.sense)));

    % Solve
    params.OutputFlag    = 0;
    params.IterationLimit = 500;
    result = gurobi(model, params);

    if ~isfield(result,'x') || ~strcmp(result.status,'OPTIMAL')
        error('Gurobi projection failed: status=%s', result.status);
    end

    z_proj = result.x;  % returns [vec(M); s; u]
end
