function [X, w, y, adj_matrix, laplacian_matrix, clusters, k] = generateSyntheticData(n, d, h_total, h, inter_cluster, outer_cluster, gamma)
    % Main function
    [adj_matrix, clusters, k] = generate_random_graph(d, h_total, h, inter_cluster, outer_cluster);
    laplacian_matrix = diag(sum(adj_matrix, 2)) - adj_matrix;
    selected_clusters = clusters(1:h);
    w = generate_weight(d, selected_clusters, k);
    X = randn(n, d);
    y = generate_Y(X, w, gamma);
end

function [adj_matrix, clusters, k] = generate_random_graph(d, h_total, h, inter_cluster_prob, outer_cluster_prob)
    % Subfunction
    % Partition features into clusters
    breakpoints = random_partition(d, h_total);
    clusters = arrayfun(@(i) breakpoints(i) + 1:breakpoints(i + 1), 1:h_total, 'UniformOutput', false);
    selected_clusters = clusters(1:h);
    k = clusters{h}(end);

    % Initialize adjacency matrix
    adj_matrix = sparse(d, d);

    % Intra-cluster edges
    for cluster = clusters
        cluster = cluster{1};
        size_c = numel(cluster);
        block = rand(size_c) < inter_cluster_prob;
        block = triu(block, 1) + triu(block, 1)';
        adj_matrix(cluster, cluster) = block;
    end

    % Inter-cluster edges
    for i = 1:h_total
        for j = i + 1:h_total
            cluster_i = clusters{i};
            cluster_j = clusters{j};
            block = rand(numel(cluster_i), numel(cluster_j)) < outer_cluster_prob;
            adj_matrix(cluster_i, cluster_j) = block;
            adj_matrix(cluster_j, cluster_i) = block';
        end
    end
end

function w = generate_weight(d, selected_clusters, k)
    % Subfunction
    w = zeros(d, 1);
    for i = 1:numel(selected_clusters)
        cluster = selected_clusters{i};
        feature_weight = sign(randn(1));
        w(cluster) = feature_weight;
        fprintf('Cluster %d: Features [%d:%d], Weight: %.2f\n', ...
            i, cluster(1), cluster(end), feature_weight);
    end
end

function y = generate_Y(X, w, gamma)
    % Subfunction
    n = size(X, 1);
    epsilon = randn(n, 1) * gamma;
    signal = X * w;
    y = signal + epsilon;
    fprintf('SNR: %.2f dB\n', compute_snr(signal, epsilon));
end

function snr_db = compute_snr(signal, noise)
    % Subfunction
    signal_power = mean(signal.^2);
    noise_power = mean(noise.^2);
    snr = signal_power / noise_power;
    snr_db = 10 * log10(snr);
end

function breakpoints = random_partition(d, h_total)
    % Subfunction
    if d < h_total
        error('The number of clusters exceeds the total number.');
    end
    breakpoints = sort(randperm(d - 1, h_total - 1));
    breakpoints = [0, breakpoints, d];
    for i = 1:h_total
        fprintf('Cluster %d contains features from %d to %d, with size %d\n', ...
            i, breakpoints(i) + 1, breakpoints(i + 1), breakpoints(i + 1) - breakpoints(i));
    end
end