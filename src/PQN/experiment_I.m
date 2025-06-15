clc;
clear;

% Timing start
tStart = tic;

% Generate synthetic data
n = 1000; % Number of samples
d = 500;  % Number of features
k = 20;   % Number of non-zero features
h_total = 40; % Number of clusters in the graph
h = 5;        % Number of selected clusters
nVars = d * h; % Number of Boolean variables in m
inter_cluster = 1; % Probability of inter-cluster edges in the graph
outer_cluster = 0.05; % Probability of outer-cluster edges in the graph
gamma = 1.5; % Noise standard deviation

mu = 1;

SNR = 1;

fixed_seed = true;
random_rounding = false;
connected = false;
correlated = true;
random_graph = true;
visualize = true;

function [X, w, y, adjMatrix, laplacianMatrix, clusters, k] = readSyntheticData(filePath)
    % Load the data
    data = load(filePath);

    % Extract fields
    X = data.X;
    w = data.w;
    y = data.y;
    adjMatrix = data.adj_matrix; % Use the correct field name
    laplacianMatrix = data.laplacian_matrix;
    clusters = data.clusters;
    k = data.k;

    % Print for debugging (optional)
    fprintf('Loaded data from %s:\n', filePath);
    fprintf('  X: %d x %d\n', size(X));
    fprintf('  w: %d\n', numel(w));
    fprintf('  y: %d\n', numel(y));
    fprintf('  adjMatrix: %d x %d\n', size(adjMatrix));
    fprintf('  laplacianMatrix: %d x %d\n', size(laplacianMatrix));
    fprintf('  clusters: %d clusters\n', numel(clusters));
    fprintf('  k: %d\n', k);
end

% Generate or read synthetic data
if fixed_seed
    filePath = "data.mat";
    [X, w, y, adjMatrix, L, clusters_true, k]  = readSyntheticData(filePath);
else
    % Generate synthetic data
    [X, w_true, y, adj_matrix, L, clusters_true, k] = generateSyntheticData( ...
        n, d, h_total, h, inter_cluster, outer_cluster, gamma);
    % Save synthetic data
end

% Define constants
clusters_size = cellfun(@length, clusters_true);
clusters_size = sort(clusters_size);
C = (2 * clusters_size(end) + outer_cluster * (d - clusters_size(1) - clusters_size(2))) + 2 * d;

% Modify X for objective function
X_hat = repmat(X, 1, h);

fprintf('Check!\n');
fprintf('Execution time (data generation): %.2f seconds\n', toc(tStart));

% Initial guess of parameters
m_initial = ones(nVars, 1) * (double(k) / double(nVars));

pho = d * 4 * k;
% Objective function
funObj = @(m) L0Obj(X_hat, m, y, L, pho, mu, d, h, n);

% Projection function (Gurobi or other solver)
funProj = @(m) ProjOperator_Gurobi(m, k, d, h);

% Solve with PQN
options.maxIter = 100;
options.verbose = 3;

[mout, obj, fun_evals] = minConF_PQN(funObj, m_initial, funProj, options);

fprintf('Solution:\n');
% Reshape mout into a matrix of size (d, h)
mout_matrix = reshape(mout, [d, h]);

% Example: Compute sums for features and clusters
m_features_sum = sum(mout_matrix, 2); % Sum over clusters for each feature
m_clusters_sum = sum(mout_matrix, 1); % Sum over features for each cluster

% Display results
disp('Sum over features for each cluster:');
disp(m_features_sum);

disp('Sum over clusters for each feature:');
disp(m_clusters_sum);

% Save results
save('mout.mat', 'mout');

% Additional analysis and metrics (to be implemented based on MATLAB capabilities)