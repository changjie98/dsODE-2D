function block_conn_mat = generate_block_matrix_gpu(row_counts, col_counts, prob_mat)
    % GENERATE_BLOCK_MATRIX_GPU 根据概率矩阵生成随机连接矩阵 (GPU加速版)
    % 输入:
    %   row_counts: 列向量，表示每个行区块(Post)有多少神经元
    %   col_counts: 列向量，表示每个列区块(Pre)有多少神经元
    %   prob_mat:   概率矩阵，维度与 numel(row_counts) x numel(col_counts) 一致
    
    n_row_blocks = length(row_counts);
    n_col_blocks = length(col_counts);
    
    % 计算最终矩阵的总维度
    Total_Rows = sum(row_counts);
    Total_Cols = sum(col_counts);
    
    % 预计算每个区块在最终矩阵中的起始索引偏移量
    row_offsets = [0; cumsum(row_counts(1:end-1))];
    col_offsets = [0; cumsum(col_counts(1:end-1))];
    
    % 检查是否有可用的 GPU
    try
        gpu_dev = gpuDevice;
        use_gpu = true;
        % fprintf('Using GPU: %s\n', gpu_dev.Name);
    catch
        use_gpu = false;
    end
    
    % 初始化存储稀疏矩阵索引的容器
    all_rows = cell(n_row_blocks, n_col_blocks);
    all_cols = cell(n_row_blocks, n_col_blocks);
    
    % --- 并行/循环生成 ---
    % 由于 prob_mat 通常很小 (121x121)，直接双重循环开销很小
    % 主要开销在 rand 生成。
    
    for i = 1:n_row_blocks
        n_r = row_counts(i);
        if n_r == 0, continue; end
        
        for j = 1:n_col_blocks
            n_c = col_counts(j);
            if n_c == 0, continue; end
            
            p = prob_mat(i, j);
            
            % 如果概率为0，直接跳过
            if p == 0, continue; end
            
            % 生成随机连接
            if use_gpu
                % GPU 模式：生成 -> 比较 -> 提取索引 -> 传回 CPU
                % 注意：如果 Block 很大，可能需要分批，但 11x11 网格下 Block 通常不大
                rand_gpu = rand(n_r, n_c, 'gpuArray');
                mask_gpu = rand_gpu < p;
                [r_local, c_local] = find(mask_gpu);
                r_local = gather(r_local);
                c_local = gather(c_local);
            else
                % CPU 模式
                if p < 0.05 % 稀疏时用 sprand 可能更省内存，但速度较慢；直接 rand < p 比较快
                    mask = rand(n_r, n_c) < p;
                    [r_local, c_local] = find(mask);
                else
                    mask = rand(n_r, n_c) < p;
                    [r_local, c_local] = find(mask);
                end
            end
            
            % 加上偏移量，转换为全局索引
            if ~isempty(r_local)
                all_rows{i,j} = r_local + row_offsets(i);
                all_cols{i,j} = c_local + col_offsets(j);
            end
        end
    end
    
    % --- 构建最终稀疏矩阵 ---
    % 将 cell 数组展开合并
    rows_vec = vertcat(all_rows{:});
    cols_vec = vertcat(all_cols{:});
    
    % 创建稀疏矩阵
    block_conn_mat = sparse(rows_vec, cols_vec, true, Total_Rows, Total_Cols);
end