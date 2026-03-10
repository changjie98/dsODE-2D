function main_process_blocks()
    %% 1. 设置路径和参数
    data_path = '/home/changj/MATLAB/project/conn_mat'; 
    data_path2 = '/home/changj/MATLAB/project/block_conn_mat'; 
    L = sqrt(10); % 空间大小，请确保与生成positions时一致
    step = 0.3;  % 网格步长
    
    sigma_values = 0.05:0.01:0.5;
    types = {'EE', 'EI', 'IE', 'II'};
    
    %% 2. 加载位置数据
    fprintf('Loading neuron positions...\n');
    pos_file = fullfile(data_path, 'neuron_positions.mat');
    if ~exist(pos_file, 'file')
        error('Position file not found: %s', pos_file);
    end
    load(pos_file, 'e_pos', 'i_pos');
    
    % 位置结构体
    pos_struct.E = e_pos;
    pos_struct.I = i_pos;
    
    %% 3. 循环处理所有 Sigma 和连接类型
    for sigma = sigma_values
        fprintf('\n================ Processing Sigma = %g ================\n', sigma);
        
        for t_idx = 1:length(types)
            type_name = types{t_idx}; % 例如 'EE', 'EI'
            
            % 解析 Post 和 Pre 类型
            post_type = type_name(1); 
            pre_type = type_name(2);  
            
            % 获取对应的坐标
            current_pos_row = pos_struct.(post_type);
            current_pos_col = pos_struct.(pre_type);
            
            % 构建输入文件名 (例如 EE_sig0.05_conn_mat.mat)
            input_filename = sprintf('%s_sig%g_conn_mat.mat', type_name, sigma);
            input_full_path = fullfile(data_path, input_filename);
            
            if ~exist(input_full_path, 'file')
                warning('File not found: %s. Skipping.', input_full_path);
                continue;
            end
            
            fprintf('  -> [%s] Loading raw matrix...\n', type_name);
            loaded_data = load(input_full_path);
            field_names = fieldnames(loaded_data);
            raw_conn_mat = loaded_data.(field_names{1}); % 自动获取变量名
            
            %% 4. 计算并保存概率矩阵
            fprintf('     [%s] Calculating probability matrix...\n', type_name);
            [prob_mat, row_groups, col_groups] = calc_block_probabilities(...
                current_pos_row, current_pos_col, raw_conn_mat, step, L);
            
            % --- 关键修改：动态重命名变量 ---
            prob_save_struct = struct();
            
            % 1. 定义特定的变量名 (如 EE_prob_mat)
            specific_prob_name = sprintf('%s_prob_mat', type_name);
            prob_save_struct.(specific_prob_name) = prob_mat;
            
            % 2. 同时也保存 groups 信息 (建议也加上前缀，或者保持通用)
            % 这里我加上前缀以防万一，或者你可以保持 row_groups
            prob_save_struct.row_groups = row_groups;
            prob_save_struct.col_groups = col_groups;
            
            % 保存
            prob_save_name = sprintf('%s_sig%g_prob_mat.mat', type_name, sigma);
            save(fullfile(data_path2, prob_save_name), '-struct', 'prob_save_struct');
            
            %% 5. 生成并保存区块化连接矩阵 (GPU加速)
            fprintf('     [%s] Generating stochastic block matrix...\n', type_name);
            
            block_conn_mat = generate_block_matrix_gpu(row_groups, col_groups, prob_mat);
            
            % --- 关键修改：动态重命名变量 ---
            block_save_struct = struct();
            
            % 定义特定的变量名 (如 EE_block_conn_mat)
            specific_block_name = sprintf('%s_block_conn_mat', type_name);
            block_save_struct.(specific_block_name) = block_conn_mat;
            
            % 保存 (大文件使用 v7.3)
            block_save_name = sprintf('%s_sig%g_block_conn_mat.mat', type_name, sigma);
            save(fullfile(data_path2, block_save_name), '-struct', 'block_save_struct', '-v7.3');
            
        end
    end
    
    fprintf('\nAll processing complete!\n');
end