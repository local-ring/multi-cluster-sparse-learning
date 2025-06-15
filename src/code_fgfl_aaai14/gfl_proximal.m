function [beta, funcVal] = gfl_proximal(datafile, resultfile, rho1, rho2)
    % Load data from .mat file
    data = load(datafile);

    X = data.X;            % Design matrix
    y = data.y;            % Response vector
    AdjMat = data.AdjMat;  % Adjacency matrix

    % Convert adjacency matrix to graph structure
    [nE, E_in, E_out, E_w] = adj_matrix_to_graph(AdjMat);

    % Graph structure required by fast_gfl
    Graph = {nE, E_w, E_in, E_out};

    % disp(['Class of X: ', class(X)]);
    % disp(['Class of y: ', class(y)]);
    % disp(['Graph structure type: ', class(Graph)]);


    % Options for fast_gfl
    opts.maxIter = 1000;
    opts.tol = 1e-4;

    % Call the fast_gfl function
    [beta, funcVal] = fast_gfl(X, y, Graph, rho1, rho2, opts);

    % Save results
    save(resultfile, 'beta', 'funcVal');

    % Display results
    % fprintf('Results saved to %s\n', resultfile);

    % Function to process adjacency matrix
    function [nE, E_in, E_out, E_w] = adj_matrix_to_graph(AdjMat)
        [rows, cols] = find(AdjMat > 0); % Find nonzero entries
        nE = length(rows);              % Number of edges
        E_in = rows;                    % Starting nodes
        E_out = cols;                   % Ending nodes
        E_w = AdjMat(sub2ind(size(AdjMat), rows, cols)); % Edge weights
    end
end


