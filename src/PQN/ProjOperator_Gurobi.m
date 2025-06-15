function solution = ProjOperator_Gurobi(m, k, d, h)
    % Projects the input vector m onto the simplex using Gurobi, ensuring constraints.
    
    % Inputs:
    %   m: Input vector of size (d * h).
    %   k: Sparsity level (number of non-zero entries allowed).
    %   d: Number of features.
    %   h: Number of clusters.
    
    % Outputs:
    %   solution: Projected vector of size (d * h).
    
    % Define the constraint matrices
    A = ones(1, d * h);       % Sparsity constraint
    b = k;                    % Sparsity constraint bound

    Aa = -1 * ones(1, d * h);       % Sparsity constraint
    bb = -(k - 1);                  % Sparsity constraint bound

    A = [A; Aa];
    b = [b; bb];
    
    % Feature assignment constraints
    B = zeros(d, d * h);
    for i = 1:d
        B(i, (i - 1) * h + 1:i * h) = 1;
    end
    c = ones(d, 1);
    
    % Combine all constraints
    C = [A; B];
    Cb = [b; c];

    % Define the additional Cluster constraint
    Cluster = zeros(h, d * h);
    for j = 1:h
        Cluster(j, j:h:end) = -1; % Set every h-th element in row j
    end
    Cluster_b = -1 * ones(h, 1); % RHS of the constraint
    
    % Update the constraint matrix and RHS
    C = [C; Cluster];    % Add Cluster constraints to the existing constraints
    Cb = [Cb; Cluster_b]; % Add corresponding RHS values
    
    % Create Gurobi model
    model.modelsense = 'min';      % Minimization problem
    model.Q = sparse(eye(d * h));  % Quadratic terms
    model.obj = -2 * m(:)';        % Linear terms
    model.A = sparse(C);           % Constraints matrix
    model.rhs = full(double(Cb(:))); % Ensure RHS is a dense double vector
    model.sense = '<';             % All constraints are inequalities
    model.lb = zeros(d * h, 1);    % Lower bound for x
    model.ub = ones(d * h, 1);     % Upper bound for x
    
    % Gurobi options
    params.outputflag = 0;         % Suppress solver output
    
    % Solve using Gurobi
    result = gurobi(model, params);
    
    % Handle infeasibility
    if strcmp(result.status, 'INFEASIBLE')
        warning('Model is infeasible. Returning default solution.');
        solution = zeros(d * h, 1);
        return;
    end
    
    % Return the solution
    solution = result.x;
end