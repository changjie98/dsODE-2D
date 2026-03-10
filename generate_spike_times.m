function [E_sp_dsODE,I_sp_dsODE] = generate_spike_times(fr_e, fr_i, ne, ni)
    % 参数设置
    dt = 0.1; % 内部设置时间分辨率
    fr_e = fr_e*dt;
    fr_i = fr_i*dt;
    
    % 确定群体数量
    if size(fr_e, 2) > 0
        N_e = size(fr_e, 2); % E群体数量
    else
        error('fr_e 必须包含至少一个群体');
    end
    
    if size(fr_i, 2) > 0
        N_i = size(fr_i, 2); % I群体数量
    else
        N_i = 0;
        warning('未检测到I群体数据，仅生成E群体尖峰数据');
    end
    
    % 验证输入参数
    if length(ne) ~= N_e
        error('ne参数长度必须与fr_e的群体数一致');
    end
    if length(ni) ~= N_i && N_i > 0
        error('ni参数长度必须与fr_i的群体数一致');
    end
    
    % 神经元数量设置
    ne_per_group = ne; % 每个E群体的神经元数向量
    ni_per_group = ni; % 每个I群体的神经元数向量
    
    % 计算神经元总数
    total_e_neurons = sum(ne_per_group);
    total_i_neurons = sum(ni_per_group);
    total_neurons = total_e_neurons + total_i_neurons;
    
    % 确定模拟时长
    time_steps = size(fr_e, 1); % 时间步数
    duration_time = time_steps * dt; % 总模拟时间(ms)
    
    % 确保放电率非负
    fr_e(fr_e < 0) = 0;
    fr_i(fr_i < 0) = 0;
    
    % 初始化尖峰时间收集器
    E_sp_list = []; % 存储E神经元尖峰时间 [神经元ID, 时间]
    I_sp_list = []; % 存储I神经元尖峰时间 [神经元ID, 时间]
    
    % 神经元ID分配计数器
    current_e_id = 0;
    current_i_id = total_e_neurons; % I神经元ID从E神经元总数之后开始
    
    % ===========================================
    % 处理每个E群体的放电率
    % ===========================================
    for group_idx = 1:N_e
        current_ne = ne_per_group(group_idx); % 当前E群体的神经元数
        group_fr = fr_e(:, group_idx); % 当前E群体的放电率
        
        % 为该群体分配神经元ID范围
        neuron_ids = (current_e_id + 1):(current_e_id + current_ne);
        current_e_id = current_e_id + current_ne;
        
        % 计算每个时间点的发放神经元数
        spike_counts = min(round(group_fr), current_ne);
        
        % 随机分配发放事件
        for t = 1:time_steps
            if spike_counts(t) > 0
                % 确保spike_counts不超过神经元数量
                valid_spike_count = min(spike_counts(t), current_ne);
                
                % 随机选择要发放的神经元
                selected_neurons = randperm(current_ne, valid_spike_count);
                selected_ids = neuron_ids(selected_neurons);
                current_time = t * dt;
                
                % 添加到尖峰列表
                new_events = [selected_ids; repmat(current_time, 1, valid_spike_count)];
                E_sp_list = [E_sp_list, new_events];
            end
        end
    end
    
    % ===========================================
    % 处理每个I群体的放电率
    % ===========================================
    for group_idx = 1:N_i
        current_ni = ni_per_group(group_idx); % 当前I群体的神经元数
        group_fr = fr_i(:, group_idx); % 当前I群体的放电率
        
        % 为该群体分配神经元ID范围
        neuron_ids = (current_i_id + 1):(current_i_id + current_ni);
        current_i_id = current_i_id + current_ni;
        
        % 计算每个时间点的发放神经元数
        spike_counts = min(floor(group_fr), current_ni);
        
        % 随机分配发放事件
        for t = 1:time_steps
            if spike_counts(t) > 0
                % 确保spike_counts不超过神经元数量
                valid_spike_count = min(spike_counts(t), current_ni);
                
                % 随机选择要发放的神经元
                selected_neurons = randperm(current_ni, valid_spike_count);
                selected_ids = neuron_ids(selected_neurons);
                current_time = t * dt;
                
                % 添加到尖峰列表
                new_events = [selected_ids; repmat(current_time, 1, valid_spike_count)];
                I_sp_list = [I_sp_list, new_events];
            end
        end
    end
    
    % ===========================================
    % 构建输出矩阵
    % ===========================================
    if isempty(E_sp_list)
        E_sp_dsODE = zeros(2, 0); % 空矩阵
    else
        % 按时间排序
        [~, sort_idx] = sort(E_sp_list(2, :));
        E_sp_dsODE = E_sp_list(:, sort_idx);
    end
    
    if isempty(I_sp_list)
        I_sp_dsODE = zeros(2, 0); % 空矩阵
    else
        % 按时间排序
        [~, sort_idx] = sort(I_sp_list(2, :));
        I_sp_dsODE = I_sp_list(:, sort_idx);
    end

    %res_dsODE_sp.E_sp_dsODE= E_sp_dsODE;
    %res_dsODE_sp.I_sp_dsODE= I_sp_dsODE;
end