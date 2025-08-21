function gfl_multi_new_inv_same(datafile, resultfile, rho, mu, k, l)
% gfl_multi_new_inv_same
%   Minimizes over (M,s,u) with projection-based constraints:
%
%       f(M,s) = y' * ( I_n + (1/rho) * X * M * D(s) * M' * X' )^{-1} * y ...
%                 + 0.5 * mu * tr(M' * L * M)
%
%   where D(s) = diag(s). The vector u appears only in the projection
%   constraints (convex surrogate); it does not enter f.
%
%   Constraints (handled by Project_UMS_Gurobi):
%     0 <= M <= 1,    sum_j M(i,j) = 1  (row-stochastic, i = 1..d)
%     0 <= s <= 1
%     0 <= u <= 1,    sum_i u_i <= k
%     u_i >= m_ij + s_j - 1  for all i,j
%
% Inputs:
%   datafile   - .mat with X (n x d), y (n x 1), L (d x d; not necessarily symmetric)
%   resultfile - .mat to save M, s, u, funcVal, funEvals
%   rho        - positive scalar
%   mu         - nonnegative scalar (weight on tr(M' L M))
%   k          - cardinality cap used in projection on u
%   l          - number of columns (clusters) in M and length of s
%
% Solver:     minConF_PQN  (smooth objective + projection)
% Projection: Gurobi MATLAB API
%
% ---------- Mathematical notes (how f and gradients are evaluated) ----------
% Let
%   A(M,s) = I_n + (1/rho) * X * M * D(s) * M' * X'  (SPD)
%   w      = A^{-1} y     (solved by Cholesky; do not form A^{-1})
%   u      = X' * w
% Then
%   f1 = y' * w
%   f2 = 0.5 * mu * tr(M' L M)
%   f  = f1 + f2
%
% Gradients:
%   H := u u' (rank-1, d-by-d, not formed explicitly)
%   ∇_M f = -(2/rho) * H * M * D(s) + 0.5 * mu * (L + L') * M
%         = -(2/rho) * u * (u' * (M*D(s))) + 0.5 * mu * (L + L') * M
%   ∇_s f = -(1/rho) * diag(M' * H * M)
%         = -(1/rho) * (M' * u) .^ 2              (elementwise square)
%   ∇_u f = 0  (u is not in the smooth objective)
%
% Efficient evaluation:
%   - Build A via XM = X*M (n-by-l); XMD = XM .* (ones(n,1)*s') ;
%     then A = I_n + (1/rho) * (XMD) * (XM')   (never form D(s) explicitly).
%   - Solve A w = y with chol(A,'lower') if possible, else backslash.
%   - Compute u = X' * w once; reuse it in both ∇_M and ∇_s.
%
% ---------------------------------------------------------------------------

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
    funObj  = @(z) Objective_MS_inv(z, X, y, rho, mu, L, d, l);
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

function [f, g] = Objective_MS_inv(z, X, y, rho, mu, L, d, l)
% Objective_MS_inv
%   f(M,s) = y' * ( I_n + (1/rho) * X * M * D(s) * M' * X' )^{-1} * y ...
%            + 0.5 * mu * tr(M' * L * M)
%
%   INPUT packing:
%     z = [vec(M); s; u], with:
%       M : d x l, s : l x 1, u : d x 1  (u DOES NOT enter f)
%
%   OUTPUT:
%     f : scalar objective value
%     g : gradient packed as [vec(∇_M f); ∇_s f; ∇_u f], where ∇_u f = 0.
%
%   NOTES ON IMPLEMENTATION:
%     - We avoid forming A^{-1}. Solve A w = y (SPD) via Cholesky/backslash.
%     - Reuse u = X' * w; note H = u*u' is rank-1. Use matrix-vector products
%       to apply H without explicitly forming it.

    % unpack
    M = reshape(z(1:d*l), [d, l]);
    s = z(d*l + (1:l));
    % u_var = z(d*l + l + (1:d));  % present for projection only; not used here

    % Precompute XM, XMD to build A efficiently
    XM  = X * M;                        % n x l
    XMD = bsxfun(@times, XM, s');       % n x l  (scale columns by s_j)

    % Build A = I + (1/rho) * X M D(s) M' X' = I + (1/rho) * (XMD) * (XM)'
    [n, ~] = size(X);
    A  = eye(n) + (1/rho) * (XMD * XM');

    % Solve A w = y  (SPD; use Cholesky if possible)
    [R, pflag] = chol(A, 'lower');
    if pflag ~= 0
        w = A \ y;                      % robust fallback
    else
        w = R' \ (R \ y);
    end

    % Objective value
    f1 = y' * w;                        % y' * A^{-1} * y
    % Symmetrize L inside the quadratic term to be safe when L ~= L'
    f2 = 0.5 * mu * trace(M' * ((L + L')/2) * M);
    f  = f1 + f2;

    % Common vector u = X' * w
    u = X' * w;                         % d x 1

    % ----- Gradients -----

    % ∇_M f = -(2/rho) * (u*u') * M * D(s) + 0.5 * mu * (L + L') * M
    % Implement as: u * ( u' * (M .* s') ) to avoid forming H explicitly.
    MD   = bsxfun(@times, M, s');       % d x l   (columns m_j scaled by s_j)
    row  = (u' * MD);                   % 1 x l
    gM1  = -(2/rho) * (u * row);        % d x l
    gM2  = 0.5 * mu * ((L + L') * M);   % d x l
    gM   = gM1 + gM2;

    % ∇_s f = -(1/rho) * diag(M' * (u*u') * M) = -(1/rho) * (M' * u).^2
    Mu   = M' * u;                      % l x 1
    gs   = -(1/rho) * (Mu .^ 2);        % l x 1

    % ∇_u f = 0  (u is not in f)
    gu = zeros(d,1);

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
