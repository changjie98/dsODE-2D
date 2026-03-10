tic

%% 参数设置 
params = struct;
params.J_ex = 5;
params.M = 100;
params.Mr = 66;
k = 1/100;
params.dt = 0.1;
params.duration_time = 200;
params.t_end = params.duration_time / params.dt;

params.tau_ee = 3;
params.tau_ie = 3;
params.tau_i  = 10;
params.tau_r  = 2;
params.tau_m  = 20;

params.s_ee = 3;
params.s_ie = 5;
params.s_ei = 9;
params.s_ii = 10;

params.V_bin     = 10;
params.V_bin_min = -4;
params.V_bin_num = params.M / params.V_bin - params.V_bin_min;
params.digit_num = 10;

Te = params.t_end;

% 外部输入（工作区需提供）
params.N_e = length(E_group);
params.N_i = length(I_group);
N_e = params.N_e;
N_i = params.N_i;
params.Ne = E_group(:)';  % 1 x N_e
params.Ni = I_group(:)';  % 1 x N_i
params.p_ee = block_prob_mat(1:N_e, 1:N_e);
params.p_ei = block_prob_mat(1:N_e, N_e+1:N_e+N_i);
params.p_ie = block_prob_mat(N_e+1:N_e+N_i, 1:N_e);
params.p_ii = block_prob_mat(N_e+1:N_e+N_i, N_e+1:N_e+N_i);

%% 预计算常量
denom_M = (params.M + params.Mr);
s_ee_over_tau          = params.s_ee / params.tau_ee;
s_ee_sq_over_tau_sq    = params.s_ee^2 / params.tau_ee^2;
s_ei_over_tau_i        = params.s_ei / params.tau_i;
s_ei_sq_over_tau_i_sq  = params.s_ei^2 / params.tau_i^2;
s_ie_over_tau_ie       = params.s_ie / params.tau_ie;
s_ie_sq_over_tau_ie_sq = params.s_ie^2 / params.tau_ie^2;
s_ii_over_tau_i        = params.s_ii / params.tau_i;
s_ii_sq_over_tau_i_sq  = params.s_ii^2 / params.tau_i^2;
pee_var = params.p_ee .* (1 - params.p_ee);
pei_var = params.p_ei .* (1 - params.p_ei);
pie_var = params.p_ie .* (1 - params.p_ie);
pii_var = params.p_ii .* (1 - params.p_ii);

Vb = params.V_bin_num;

%% 当前状态（无时间维）
% n_e_cur / n_i_cur: Vb x N
n_e_cur = zeros(Vb, N_e);
n_i_cur = zeros(Vb, N_i);

% V_all & V_mean 单步缓冲（Vb x N）
V_e_all_cur  = zeros(Vb, N_e);
V_e_mean_cur = zeros(Vb, N_e);
V_i_all_cur  = zeros(Vb, N_i);
V_i_mean_cur = zeros(Vb, N_i);

% fr/ref 历史（保留时间维）
fr_e_hist  = zeros(Te, N_e);
ref_e_hist = zeros(Te, N_e);
fr_i_hist  = zeros(Te, N_i);
ref_i_hist = zeros(Te, N_i);

% 当前 fr/ref（1 x N）
fr_e_cur  = zeros(1, N_e);
ref_e_cur = zeros(1, N_e);
fr_i_cur  = zeros(1, N_i);
ref_i_cur = zeros(1, N_i);

%%  初始化
init_bin = 1 - params.V_bin_min;

n_e_cur(init_bin, :) = params.Ne;   % 每个E群体的神经元数放入该 bin
n_i_cur(init_bin, :) = params.Ni;

% 随机分配初始电压总和（V_all）
V_e_all_cur(init_bin, :) = (params.V_bin * rand(1, N_e)) .* params.Ne;
V_i_all_cur(init_bin, :) = (params.V_bin * rand(1, N_i)) .* params.Ni;

% 平均电压（安全除以 1）
V_e_mean_cur = V_e_all_cur ./ max(n_e_cur, 1);
V_i_mean_cur = V_i_all_cur ./ max(n_i_cur, 1);

% 初始 fr/ref 写入历史第 1 步（可以是 0）
fr_e_hist(1, :)  = fr_e_cur;
ref_e_hist(1, :) = ref_e_cur;
fr_i_hist(1, :)  = fr_i_cur;
ref_i_hist(1, :) = ref_i_cur;

%% 就地中间量（无时间维）
H_ee_mean = zeros(1, N_e);
H_ee_var  = zeros(1, N_e);
H_ei_mean = zeros(1, N_i);
H_ei_var  = zeros(1, N_i);
H_ie_mean = zeros(1, N_e);
H_ie_var  = zeros(1, N_e);
H_ii_mean = zeros(1, N_i);
H_ii_var  = zeros(1, N_i);

% 外部输入常量缓冲
I_eex_mean_const = ones(Vb, N_e) * params.J_ex;
I_eex_var_const  = ones(Vb, N_e) * params.J_ex;
I_iex_mean_const = ones(Vb, N_i) * params.J_ex;
I_iex_var_const  = ones(Vb, N_i) * params.J_ex;

% I 输入缓冲（Vb x N）
I_e_mean_cur = zeros(Vb, N_e);
I_e_var_cur  = zeros(Vb, N_e);
I_i_mean_cur = zeros(Vb, N_i);
I_i_var_cur  = zeros(Vb, N_i);


%% 主循环
for it = 2:Te
    % 计算 E 输入
    I_ee_mean_val = s_ee_over_tau * H_ee_mean;   % N_e x 1
    I_ee_var_val  = s_ee_sq_over_tau_sq * H_ee_var;
    V_e_mean_T = V_e_mean_cur;                         % N_e x Vb
    H_ei_sum   = H_ei_mean;                      % N_e x 1
    I_ei_mean_val = s_ei_over_tau_i .* (repmat(H_ei_sum,Vb,1) .* ((V_e_mean_T + params.Mr) / denom_M) ); % N_e x Vb
    H_ei_var_sum  = H_ei_var;
    I_ei_var_val  = s_ei_sq_over_tau_i_sq .* (repmat(H_ei_var_sum,Vb,1) .* ((V_e_mean_T + params.Mr).^2 / denom_M^2) ); % N_e x Vb
    I_e_mean_cur = I_eex_mean_const + repmat(I_ee_mean_val, Vb, 1) - I_ei_mean_val;
    I_e_var_cur  = I_eex_var_const  + repmat(I_ee_var_val, Vb, 1) + I_ei_var_val;
    I_e_var_cur(I_e_var_cur < 0) = 0;

    % 计算 I 输入
    I_ie_mean_val = s_ie_over_tau_ie * H_ie_mean; % N_i x 1
    I_ie_var_val = s_ie_sq_over_tau_ie_sq * H_ie_var;
    V_i_mean_T = V_i_mean_cur; % N_i x Vb
    H_ii_sum = H_ii_mean; % N_i x 1
    I_ii_mean_val = s_ii_over_tau_i .* (repmat(H_ii_sum,Vb,1) .* ((V_i_mean_T + params.Mr) / denom_M) ); % N_i x Vb
    H_ii_var_sum = H_ii_var;
    I_ii_var_val = s_ii_sq_over_tau_i_sq .* (repmat(H_ii_var_sum,Vb,1) .* ((V_i_mean_T + params.Mr).^2 / denom_M^2) ); % N_i x Vb
    I_i_mean_cur = I_iex_mean_const + repmat(I_ie_mean_val, Vb, 1) - I_ii_mean_val;
    I_i_var_cur = I_iex_var_const + repmat(I_ie_var_val, Vb, 1) + I_ii_var_val;
    I_i_var_cur(I_i_var_cur < 0) = 0;

    % 3) 调用 E 的批量模块（当前时刻输入/状态） 
    E_state.V_n_all  = V_e_all_cur;    % Vb x N_e
    E_state.n_n      = n_e_cur;        % Vb x N_e
    E_state.fr_n     = fr_e_cur;       % 1 x N_e
    E_state.ref_n    = ref_e_cur;      % 1 x N_e
    E_state.I_n_mean = I_e_mean_cur * params.dt;
    E_state.I_n_var  = I_e_var_cur  * params.dt;

    E_output = dsODE_module_2D(E_state, params);
    %E_output = dsODE_gaussian_batch(E_state, params);

    % 加噪声（就地）并写入历史 fr/ref
    %noise = random('Normal', 0, sqrt(max(E_output.fr_n,0)));
    %fr_e_cur = max(E_output.fr_n + noise, 0);
    fr_e_cur = E_output.fr_n;
    ref_e_cur = E_output.ref_n;

    fr_e_hist(it, :)  = fr_e_cur;
    ref_e_hist(it, :) = ref_e_cur;

    % 最少写回：更新当前 n_e/V_e_all/V_e_mean
    n_e_cur = E_output.n_n;
    V_e_all_cur  = E_output.V_n_all;
    V_e_mean_cur = E_output.V_n_mean;

    % 调用 I 的批量模块（当前时刻输入/状态）
    I_state.V_n_all  = V_i_all_cur;
    I_state.n_n      = n_i_cur;
    I_state.fr_n     = fr_i_cur;
    I_state.ref_n    = ref_i_cur;
    I_state.I_n_mean = I_i_mean_cur * params.dt;
    I_state.I_n_var  = I_i_var_cur  * params.dt;

    I_output = dsODE_module_2D(I_state, params);
    %I_output = dsODE_gaussian_batch(I_state, params);

    %noise = random('Normal', 0, sqrt(max(I_output.fr_n,0)));
    %fr_i_cur = max(I_output.fr_n + noise, 0);
    fr_i_cur = I_output.fr_n;
    ref_i_cur = I_output.ref_n;

    fr_i_hist(it, :)  = fr_i_cur;
    ref_i_hist(it, :) = ref_i_cur;

    n_i_cur = I_output.n_n;
    V_i_all_cur  = I_output.V_n_all;
    V_i_mean_cur = I_output.V_n_mean;

    % 更新 H 矩阵（只保留当前）
    % E->E
    if N_e > 0
        dH_ee_mean = -H_ee_mean / params.tau_ee + fr_e_cur * params.p_ee;
        H_ee_mean  = H_ee_mean + dH_ee_mean * params.dt;
        dH_ee_var = -2*H_ee_var / params.tau_ee + H_ee_mean / params.tau_ee + fr_e_cur * pee_var;
        H_ee_var  = H_ee_var + dH_ee_var * params.dt;
    end

    % I->E
    if N_e > 0 && N_i > 0
        dH_ei_mean = -H_ei_mean / params.tau_i + fr_i_cur * params.p_ei;
        H_ei_mean  = H_ei_mean + dH_ei_mean * params.dt;
        dH_ei_var = -2*H_ei_var / params.tau_i + H_ei_mean / params.tau_i + fr_i_cur * pei_var;
        H_ei_var  = H_ei_var + dH_ei_var * params.dt;
    end
    
    % E->I
    if N_i > 0 && N_e > 0
        dH_ie_mean = -H_ie_mean / params.tau_ie + fr_e_cur * params.p_ie;
        H_ie_mean  = H_ie_mean + dH_ie_mean * params.dt;
        dH_ie_var = -2*H_ie_var / params.tau_ie + H_ie_mean / params.tau_ie + fr_e_cur * pie_var;
        H_ie_var  = H_ie_var + dH_ie_var * params.dt;
    end

    % I->I
    if N_i > 0
        dH_ii_mean = -H_ii_mean / params.tau_i + fr_i_cur * params.p_ii;
        H_ii_mean  = H_ii_mean + dH_ii_mean * params.dt;
        dH_ii_var = -2*H_ii_var / params.tau_i + H_ii_mean / params.tau_i + fr_i_cur * pii_var;
        H_ii_var  = H_ii_var + dH_ii_var * params.dt;
    end
    
end

%%保存结果
res_dsODE = struct();
res_dsODE.fr_e  = fr_e_hist;        % Te x N_e
res_dsODE.ref_e = ref_e_hist;       % Te x N_e
res_dsODE.fr_i  = fr_i_hist;        % Te x N_i
res_dsODE.ref_i = ref_i_hist;       % Te x N_i
[res_dsODE.E_sp, res_dsODE.I_sp] = generate_spike_times(res_dsODE.fr_e, res_dsODE.fr_i, E_group, I_group);
clearvars -except res_lif res_dsODE params E_group I_group block_conn_mat block_prob_mat block_positions index_map connection_mat positions;

toc
