function res = run_LIF_model_core(params, connection_mat)
% Internal LIF solver used by the public single- and grid-population entry points.
% Mini LIF simulation with fixed neuron-level connectivity (CPU only).
%
% Inputs:
%   params.ne, params.ni, params.dt, params.duration_time
%   params.M, params.Mr, params.tau_m, params.tau_r
%   params.tau_ee, params.tau_ei, params.tau_i
%   params.s_ee, params.s_ei, params.s_ie, params.s_ii
%   params.Ex_Poisson_lambda
%   connection_mat: (ne+ni) x (ne+ni), row=pre, col=post
%
% Output:
%   res.fr_e, res.fr_i, res.E_sp, res.I_sp, res.record
% params.record_times optionally requests exact snapshot times in ms.
% params.refractory_mode may be 'fixed' (default) or 'exponential'.

res = struct();

dt = params.dt;
ne = params.ne;
ni = params.ni;
t_end = round(params.duration_time / dt);

if nargin < 2 || isempty(connection_mat)
    error('run_LIF_model_mini requires connection_mat.');
end

if size(connection_mat, 1) ~= ne + ni || size(connection_mat, 2) ~= ne + ni
    error('connection_mat size mismatch. Expect (%d x %d).', ne + ni, ne + ni);
end

if ~isfield(params, 'record_interval')
    params.record_interval = 5; % ms
end
if ~isfield(params, 'init_v_mu')
    params.init_v_mu = 20;
end
if ~isfield(params, 'init_v_sigma')
    params.init_v_sigma = 10;
end
if ~isfield(params, 'refractory_mode') || isempty(params.refractory_mode)
    params.refractory_mode = 'fixed';
end
refractory_mode = lower(char(params.refractory_mode));
if ~ismember(refractory_mode, {'fixed','exponential'})
    error('params.refractory_mode must be fixed or exponential.');
end

% Keep sparse/logical to reduce memory and speed row-aggregation.
if ~issparse(connection_mat)
    connection_mat = sparse(connection_mat);
end
connection_mat = spones(connection_mat);

conn_ee = connection_mat(1:ne, 1:ne);
conn_ei = connection_mat(1:ne, ne+1:ne+ni);      % E -> I
conn_ie = connection_mat(ne+1:ne+ni, 1:ne);      % I -> E
conn_ii = connection_mat(ne+1:ne+ni, ne+1:ne+ni);

if isfield(params, 'init_v_min') && isfield(params, 'init_v_max')
    V_e = params.init_v_min + (params.init_v_max - params.init_v_min) * rand(1, ne);
    V_i = params.init_v_min + (params.init_v_max - params.init_v_min) * rand(1, ni);
else
    V_e = normrnd(params.init_v_mu, params.init_v_sigma, 1, ne);
    V_i = normrnd(params.init_v_mu, params.init_v_sigma, 1, ni);
end
res.initial_V_e = V_e;
res.initial_V_i = V_i;

H_ee = zeros(1, ne);
H_ie = zeros(1, ne);
H_ei = zeros(1, ni);
H_ii = zeros(1, ni);
ref_e = zeros(1, ne);
ref_i = zeros(1, ni);

nf_e = zeros(t_end, 1);
nf_i = zeros(t_end, 1);

% Spike list format: [neuron_id_local, time_ms]
E_sp = zeros(0, 2);
I_sp = zeros(0, 2);

J_eex = random('normal', params.Ex_Poisson_lambda * dt, ...
    sqrt(params.Ex_Poisson_lambda * dt), t_end, ne);
J_iex = random('normal', params.Ex_Poisson_lambda * dt, ...
    sqrt(params.Ex_Poisson_lambda * dt), t_end, ni);

if isfield(params, 'record_times') && ~isempty(params.record_times)
    record_times = double(params.record_times(:).');
    if any(~isfinite(record_times)) || any(record_times < 0) || ...
            any(record_times > params.duration_time)
        error('params.record_times must lie inside [0, duration_time].');
    end
    rec_indices = round(record_times / dt);
    mismatch = abs(rec_indices * dt - record_times);
    if any(mismatch > max(1e-9, 10*eps(max(1, params.duration_time))))
        error('Each record time must be an integer multiple of params.dt.');
    end
    rec_indices = unique(rec_indices(rec_indices >= 2 & rec_indices <= t_end));
else
    record_step = max(1, round(params.record_interval / dt));
    rec_indices = 2:record_step:t_end;
end
n_recs = numel(rec_indices);
res.record = struct('t', cell(n_recs, 1), 'V_e', cell(n_recs, 1), ...
    'V_i', cell(n_recs, 1), 'H_ee', cell(n_recs, 1), 'H_ie', cell(n_recs, 1), ...
    'H_ei', cell(n_recs, 1), 'H_ii', cell(n_recs, 1), ...
    'ref_e', cell(n_recs, 1), 'ref_i', cell(n_recs, 1));
rec_ptr = 1;

for it = 2:t_end
    I_eex = J_eex(it, :);
    I_iex = J_iex(it, :);

    J_ie = (V_e + params.Mr) * params.s_ie / (params.M + params.Mr);
    J_ii = (V_i + params.Mr) * params.s_ii / (params.M + params.Mr);

    I_ee = params.s_ee .* H_ee / params.tau_ee * dt;
    I_ie = J_ie .* H_ie / params.tau_i * dt;
    I_ei = params.s_ei .* H_ei / params.tau_ei * dt;
    I_ii = J_ii .* H_ii / params.tau_i * dt;

    active_e = (ref_e <= 0);
    active_i = (ref_i <= 0);

    if params.tau_m ~= 0
        I_leak_e = -V_e ./ params.tau_m * dt;
        I_leak_i = -V_i ./ params.tau_m * dt;
    else
        I_leak_e = zeros(1, ne);
        I_leak_i = zeros(1, ni);
    end

    V_e(active_e) = V_e(active_e) + I_eex(active_e) + I_leak_e(active_e) + ...
        I_ee(active_e) - I_ie(active_e);
    V_i(active_i) = V_i(active_i) + I_iex(active_i) + I_leak_i(active_i) + ...
        I_ei(active_i) - I_ii(active_i);

    spk_e = find(V_e > params.M);
    spk_i = find(V_i > params.M);
    nf_e(it) = numel(spk_e);
    nf_i(it) = numel(spk_i);

    if ~isempty(spk_e)
        E_sp = [E_sp; [spk_e(:), repmat(it * dt, numel(spk_e), 1)]];
    end
    if ~isempty(spk_i)
        I_sp = [I_sp; [spk_i(:), repmat(it * dt, numel(spk_i), 1)]];
    end

    V_e(spk_e) = 0;
    V_i(spk_i) = 0;

    if params.tau_r ~= 0
        ref_e = max(0, ref_e - dt / params.tau_r);
        ref_i = max(0, ref_i - dt / params.tau_r);
        if strcmp(refractory_mode, 'fixed')
            ref_e(spk_e) = 1;
            ref_i(spk_i) = 1;
        else
            ref_e(spk_e) = -log(max(rand(1,numel(spk_e)),realmin));
            ref_i(spk_i) = -log(max(rand(1,numel(spk_i)),realmin));
        end
    end

    if ~isempty(spk_e)
        sv_e = sparse(1, spk_e, 1, 1, ne);
        Hee_generate = full(sv_e * conn_ee);
        Hei_generate = full(sv_e * conn_ei);
    else
        Hee_generate = zeros(1, ne);
        Hei_generate = zeros(1, ni);
    end

    if ~isempty(spk_i)
        sv_i = sparse(1, spk_i, 1, 1, ni);
        Hie_generate = full(sv_i * conn_ie);
        Hii_generate = full(sv_i * conn_ii);
    else
        Hie_generate = zeros(1, ne);
        Hii_generate = zeros(1, ni);
    end

    H_ee = max(0, H_ee + Hee_generate - H_ee * dt / params.tau_ee);
    H_ie = max(0, H_ie + Hie_generate - H_ie * dt / params.tau_i);
    H_ei = max(0, H_ei + Hei_generate - H_ei * dt / params.tau_ei);
    H_ii = max(0, H_ii + Hii_generate - H_ii * dt / params.tau_i);

    if rec_ptr <= n_recs && it == rec_indices(rec_ptr)
        res.record(rec_ptr).t = it * dt;
        res.record(rec_ptr).V_e = V_e;
        res.record(rec_ptr).V_i = V_i;
        res.record(rec_ptr).H_ee = H_ee;
        res.record(rec_ptr).H_ie = H_ie;
        res.record(rec_ptr).H_ei = H_ei;
        res.record(rec_ptr).H_ii = H_ii;
        res.record(rec_ptr).ref_e = ref_e;
        res.record(rec_ptr).ref_i = ref_i;
        rec_ptr = rec_ptr + 1;
    end
end

res.fr_e = nf_e / dt;
res.fr_i = nf_i / dt;
res.E_sp = E_sp;
res.I_sp = I_sp;
res.connection_mat = connection_mat;
end
