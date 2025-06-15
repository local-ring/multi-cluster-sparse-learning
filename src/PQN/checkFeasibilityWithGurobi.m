function isFeasible = checkFeasibilityWithGurobi(solution, k, d, h)
    % Check if a solution is feasible under the given constraints using Gurobi.
    %
    % Parameters:
    %   solution: The solution vector to check (d*h x 1).
    %   k: Sparsity level (number of non-zero entries allowed).
    %   d: Number of features.
    %   h: Number of clusters.
    %
    % Returns:
    %   isFeasible: True if the solution is feasible, false otherwise.

    % Ensure the solution is a column vector
    solution = solution(:);

    % Define the constraint matrices
    A = ones(1, d * h);  % Sparsity constraint
    b = k;               % Sparsity bound

    % Feature assignment constraints
    B = zeros(d, d * h);
    for i = 1:d
        B(i, (i - 1) * h + 1:i * h) = 1;
    end
    c = ones(d, 1);

    % Combine all constraints
    C = [A; B];
    Cb = [b; c];

    % Create Gurobi model
    model.modelsense = 'min';       % Minimization problem
    model.obj = zeros(d * h, 1);    % Dummy objective function (minimize 0)
    model.A = sparse(C);
    model.rhs = Cb;
    model.sense = '<';              % All constraints are inequalities
    model.lb = zeros(d * h, 1);     % Lower bound for variables
    model.ub = ones(d * h, 1);      % Upper bound for variables

    % Fix variables to the provided solution
    model.vtype = 'C';              % Continuous variables
    model.start = solution;         % Start point for optimization
    model.A = [model.A; eye(d * h)];
    model.rhs = [model.rhs; solution];
    model.sense = [model.sense; '='];  % Equality constraints for fixing variables

    % Gurobi options
    params.outputflag = 0;  % Suppress solver output

    % Solve using Gurobi
    result = gurobi(model, params);

    % Check feasibility
    if strcmp(result.status, 'OPTIMAL') || strcmp(result.status, 'FEASIBLE')
        isFeasible = true;
    else
        isFeasible = false;
    end
end