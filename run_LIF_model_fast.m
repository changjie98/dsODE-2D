function [res] = run_LIF_model_fast(params, connection_mat)
    res = struct;
    
    %% 参数准备
    dt = params.dt;
    ne = params.ne;
    ni = params.ni;
    t_end = round(params.duration_time / dt);
    E_sp = [];
    I_sp = [];
    
    % 记录参数设置
    record_t1 = 0;          % 记录开始时间(ms)
    record_t2 = params.duration_time; % 记录结束时间(ms)
    record_interval = 5;    % 记录间隔(ms)
    record_steps = record_t1/dt : record_interval/dt : record_t2/dt;
    res.record = struct('t', {}, 'V_e', {}, 'V_i', {}, 'H_ee', {}, 'H_ei', {}, ...
                       'H_ie', {}, 'H_ii', {});
    record_count = 1;
    
    %% 初始化状态变量（单时间步向量）
    % 电压状态
    V_e = normrnd(20, 10, 1, ne);
    V_i = normrnd(20, 10, 1, ni);
%     V_e = zeros(1, ne);
%     V_i = zeros(1, ni);
    % 中间变量初始化
%     V_e(1, ne/3+1:2*ne/3) = normrnd(70, 10, 1, ne/3);
%     V_e(1, 2*ne/3+1:ne) = normrnd(10, 10, 1, ne/3);
    
    % H变量（单时间步）
    H_ee = zeros(1, ne);
    H_ei = zeros(1, ne);
    H_ie = zeros(1, ni);
    H_ii = zeros(1, ni);
    H_e_adapt = zeros(1, ne);
    H_i_adapt = zeros(1, ni);
    
    % 不应期状态
    ref_e = zeros(1, ne);
    ref_i = zeros(1, ni);
    
    % 脉冲计数
    nf_e = zeros(t_end, 1);
    nf_i = zeros(t_end, 1);
    
    %% 预先生成所有随机输入（优化：避免在循环中生成）
    J_eex = random('normal', params.Ex_Poisson_lambda*dt, ...
                  sqrt(params.Ex_Poisson_lambda*dt), t_end, ne);
    J_iex = random('normal', params.Ex_Poisson_lambda*dt, ...
                  sqrt(params.Ex_Poisson_lambda*dt), t_end, ni);
              
   % 将连接矩阵分为四个子矩阵
%    conn_ee = connection_mat(1:ne, 1:ne);             % E->E
%    conn_ei = connection_mat(1:ne, (ne+1):ne+ni);   % E->I
%    conn_ie = connection_mat((ne+1):ne+ni, 1:ne);   % I->E
%    conn_ii = connection_mat((ne+1):ne+ni, (ne+1):ne+ni); % I->I
    
    %% 主循环
    for i = 2:t_end
        % 获取当前时间步的输入
        I_eex = J_eex(i, :);
        I_iex = J_iex(i, :);
        
        % 计算J值（使用上一时间步的电压）
        J_ei = (V_e + params.Mr) * params.s_ei / (params.M + params.Mr);    
        J_ii = (V_i + params.Mr) * params.s_ii / (params.M + params.Mr);
        
        % 计算电流
        I_ee = params.s_ee .* H_ee / params.tau_ee * dt;
        I_ei = J_ei .* H_ei / params.tau_i * dt;    
        I_ie = params.s_ie .* H_ie / params.tau_ie * dt;
        I_ii = J_ii .* H_ii / params.tau_i * dt;
        
        % 确定可更新的神经元（不在不应期）
        e_index = ref_e == 0;
        i_index = ref_i == 0;
        
        % 泄漏电流
        if params.tau_m ~= 0
            I_leak_e = -V_e ./ params.tau_m * dt;
            I_leak_i = -V_i ./ params.tau_m * dt;
        else
            I_leak_e = zeros(1, ne);
            I_leak_i = zeros(1, ni);
        end
        
        % 自适应电流
        if params.tau_adapt ~= 0
            I_adapt_e = -H_e_adapt / params.tau_adapt / 5;
            I_adapt_i = -H_i_adapt / params.tau_adapt / 5;
        else
            I_adapt_e = zeros(1, ne);
            I_adapt_i = zeros(1, ni);
        end
        
        % 更新电压（仅更新不在不应期的神经元）
        V_e(e_index) = V_e(e_index) + I_eex(e_index) + I_leak_e(e_index) + ...
                       I_adapt_e(e_index) + I_ee(e_index) - I_ei(e_index);
        V_i(i_index) = V_i(i_index) + I_iex(i_index) + I_leak_i(i_index) + ...
                       I_adapt_i(i_index) + I_ie(i_index) - I_ii(i_index);
        
        % 检测脉冲
        new_spike_e = sparse(V_e > params.M);
        new_spike_i = sparse(V_i > params.M);
        ft_cur = [new_spike_e new_spike_i];
        nf_e(i) = sum(new_spike_e);
        nf_i(i) = sum(new_spike_i);
        
        % 构建完整的脉冲向量
        E_sp = [E_sp, [find(new_spike_e);ones(size(find(new_spike_e)))*i*dt]];
        I_sp = [I_sp, [find(new_spike_i);ones(size(find(new_spike_i)))*i*dt]];
        
        % 重置发放神经元的电压
        V_e(new_spike_e) = 0;
        V_i(new_spike_i) = 0;
        
        % 更新不应期状态
        if params.tau_r ~= 0
            % 消耗不应期
            ref_e = max(0, ref_e - dt/params.tau_r);
            ref_i = max(0, ref_i - dt/params.tau_r);
            % 设置新发放神经元的不应期
            ref_e(new_spike_e) = 1;
            ref_i(new_spike_i) = 1;
        end
        
        % 计算H生成项
        fire_index = find(ft_cur);
        efire_index = fire_index(fire_index <= ne);
        ifire_index = fire_index(fire_index > ne);  % 调整索引
        
        Hee_generate = sum(connection_mat(efire_index, 1:ne), 1);
        Hie_generate = sum(connection_mat(efire_index, ne+1:ne+ni), 1);
        Hei_generate = sum(connection_mat(ifire_index, 1:ne), 1);
        Hii_generate = sum(connection_mat(ifire_index, ne+1:ne+ni), 1);

%       % 创建稀疏脉冲向量
%        spike_vector_e = sparse(zeros(1, ne));
%        spike_vector_i = sparse(zeros(1, ni));

%        % 填充脉冲向量
%        spike_vector_e(new_spike_e) = 1;
%        spike_vector_i(new_spike_i) = 1;
% 
%        % 矩阵乘法计算H生成项
%        Hee_generate = spike_vector_e * conn_ee;
%        Hie_generate = spike_vector_e * conn_ei;
%        Hei_generate = spike_vector_i * conn_ie;
%        Hii_generate = spike_vector_i * conn_ii;
        
        
        % 更新H变量（消耗+生成）
        H_ee = H_ee + Hee_generate - H_ee * dt / params.tau_ee;
        H_ie = H_ie + Hie_generate - H_ie * dt / params.tau_ie;
        H_ei = H_ei + Hei_generate - H_ei * dt / params.tau_i;
        H_ii = H_ii + Hii_generate - H_ii * dt / params.tau_i;
        
        % 确保H非负
        H_ee = max(0, H_ee);
        H_ie = max(0, H_ie);
        H_ei = max(0, H_ei);
        H_ii = max(0, H_ii);
        
        %% 记录功能
        if ismember(i, record_steps)
            res.record(record_count).t = i * dt;
            res.record(record_count).V_e = V_e;
            res.record(record_count).V_i = V_i;
            res.record(record_count).H_ee = H_ee;
            res.record(record_count).H_ei = H_ei;
            res.record(record_count).H_ie = H_ie;
            res.record(record_count).H_ii = H_ii;
            record_count = record_count + 1;
        end
    end
    
    %% 后处理
    res.fr_e = nf_e / dt;
    res.fr_i = nf_i / dt;
    res.E_sp = E_sp;
    res.I_sp = I_sp;
end
