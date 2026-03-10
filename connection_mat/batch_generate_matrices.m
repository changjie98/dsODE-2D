function batch_generate_matrices()
    %% 1. 全局参数设置与坐标生成
    % 保持与原代码一致的参数
    L = sqrt(10);
    k = 1/100; 
    ne = round(300/k);   % 兴奋神经元数量 (约30000)
    ni = round(100/k);   % 抑制神经元数量 (约10000)
    
    % 连接概率 (原代码中所有类型的基础概率都是0.002，这里统一使用)
    p_base = 0.002*L^2; 
    
    % 生成并保存坐标 (确保所有矩阵基于同一套位置)
    disp('Generating neuron positions...');
    [e_pos, i_pos] = generate_positions_local(ne, ni, L);
    % 保存位置数据，以备后用
    save('neuron_positions.mat', 'e_pos', 'i_pos');
    
    %% 2. 循环生成不同Sigma的矩阵
    sigma_values = 0.05:0.01:0.5; % 0.05 到 0.3
    
    for sigma = sigma_values
        fprintf('Processing sigma = %g ...\n', sigma);
        
        % 定义文件名后缀 (去除小数点，或保留)
        % 为了文件名美观，使用 %g 格式化
        
        % --- 1. 生成 EE 矩阵 (Top-Left: Post=E, Pre=E) ---
        disp('  Generating EE...');
        EE_conn_mat = build_block_matrix(e_pos, e_pos, p_base, sigma, L);
        save_name = sprintf('EE_sig%g_conn_mat.mat', sigma);
        save(save_name, 'EE_conn_mat');
        
        % --- 2. 生成 EI 矩阵 (Top-Right: Post=E, Pre=I) ---
        % 注意：这是从 I 连向 E，但在矩阵中位置是右上角 (行是E, 列是I)
        disp('  Generating EI...');
        EI_conn_mat = build_block_matrix(e_pos, i_pos, p_base, sigma, L);
        save_name = sprintf('EI_sig%g_conn_mat.mat', sigma);
        save(save_name, 'EI_conn_mat');
        
        % --- 3. 生成 IE 矩阵 (Bottom-Left: Post=I, Pre=E) ---
        % 注意：这是从 E 连向 I，矩阵位置左下角 (行是I, 列是E)
        disp('  Generating IE...');
        IE_conn_mat = build_block_matrix(i_pos, e_pos, p_base, sigma, L);
        save_name = sprintf('IE_sig%g_conn_mat.mat', sigma);
        save(save_name, 'IE_conn_mat');
        
        % --- 4. 生成 II 矩阵 (Bottom-Right: Post=I, Pre=I) ---
        disp('  Generating II...');
        II_conn_mat = build_block_matrix(i_pos, i_pos, p_base, sigma, L);
        save_name = sprintf('II_sig%g_conn_mat.mat', sigma);
        save(save_name, 'II_conn_mat');
    end
    
    disp('All matrices generated successfully!');
end

%% 核心子函数：构建单个块矩阵
function mat = build_block_matrix(pos_post, pos_pre, p, sigma, L)
    % 转换为 GPU 数组 (如果显存够大，可以一次性转；为了稳妥，我们在循环内转)
    % 这里只获取维度
    n_post = size(pos_post, 1);
    n_pre = size(pos_pre, 1);
    
    % 预分配内存 (CPU端) 用于存储稀疏矩阵的索引
    % 预估非零元素数量：n_post * n_pre * p * (截断面积比例)
    % 为了安全，可以先用动态数组，或者分块构建 sparse 然后相加
    
    % === 核心策略：分块处理 ===
    % GPU 显存有限，我们每次处理 4000 个 Post 神经元 (根据显存大小调整)
    block_size = 4000; 
    num_blocks = ceil(n_post / block_size);
    
    truncation_factor = 3; 
    integral_2d = 2*pi*sigma^2*(1 - exp(-truncation_factor^2/2));
    C = 1 / integral_2d;
    
    % 将 Pre 位置一次性放入 GPU (通常 Pre 数量也就是几万，显存放得下)
    try
        g_pos_pre = gpuArray(pos_pre);
    catch
        error('GPU内存不足或未检测到支持的GPU，请改用CPU并行方案。');
    end
    
    rows_all = cell(num_blocks, 1);
    cols_all = cell(num_blocks, 1);
    
    for b = 1:num_blocks
        % 1. 确定当前块的范围
        idx_start = (b-1)*block_size + 1;
        idx_end = min(b*block_size, n_post);
        current_indices = idx_start:idx_end;
        
        % 2. 将当前块的 Post 位置移入 GPU
        g_pos_post = gpuArray(pos_post(current_indices, :));
        
        % 3. 利用 GPU 的广播机制计算距离矩阵 (无需循环)
        % g_pos_post: [M x 2], g_pos_pre: [N x 2]
        % reshape 让维度匹配: [M x 1 x 2] - [1 x N x 2]
        
        d_raw = abs(reshape(g_pos_post, [], 1, 2) - reshape(g_pos_pre, 1, [], 2));
        
        % 周期性边界 (GPU 上执行)
        d_raw = min(d_raw, L - d_raw);
        
        % 欧氏距离平方 (避免开根号，直接比较平方更在 GPU 上更快)
        dist_sq = sum(d_raw.^2, 3); 
        
        % 4. 计算概率并生成连接
        % 筛选有效范围 (dist < 3*sigma) => dist^2 < 9*sigma^2
        valid_mask = dist_sq < (truncation_factor * sigma)^2;
        
        % 只对 mask 内的元素计算 exp，节省计算量
        % 注意：直接生成随机数矩阵可能会很大，我们利用 mask 索引
        
        % 找到有效索引
        [r_local, c_local] = find(valid_mask); 
        
        if ~isempty(r_local)
            % 提取有效距离平方
            valid_dist_sq = dist_sq(valid_mask);
            
            % 计算概率 P
            probs = p * C * exp(-valid_dist_sq / (2*sigma^2));
            probs = min(probs, 1);
            
            % 生成随机数并判定
            rand_vals = gpuArray.rand(size(probs));
            connected = rand_vals < probs;
            
            % 5. 收集结果 (转回 CPU)
            r_final = r_local(connected);
            c_final = c_local(connected);
            
            % 因为是分块，行索引需要加上偏移量
            rows_all{b} = gather(r_final) + (idx_start - 1);
            cols_all{b} = gather(c_final);
        end
    end
    
    % 合并所有块的结果
    rows = vertcat(rows_all{:});
    cols = vertcat(cols_all{:});
    
    % 构建稀疏矩阵
    mat = sparse(rows, cols, true(size(rows)), n_post, n_pre);
end


%% 辅助函数：生成坐标
function [e_pos, i_pos] = generate_positions_local(ne, ni, L)
    e_grid_size = ceil(sqrt(ne));
    x_e = linspace(0, L-L/e_grid_size, e_grid_size);
    [X_e, Y_e] = meshgrid(x_e);
    e_pos = [X_e(:), Y_e(:)];
    e_pos = e_pos(1:ne,:);
    
    i_grid_size = ceil(sqrt(ni));
    x_i = linspace(0, L-L/i_grid_size, i_grid_size);
    [X_i, Y_i] = meshgrid(x_i);
    i_pos = [X_i(:), Y_i(:)];
    i_pos = i_pos(1:ni,:);
end