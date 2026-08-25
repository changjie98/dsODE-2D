function res_dsODE = dsODE_fvm_single_core(block_prob_mat, params, lif_result, lif_time_ms)
% Internal solver for main_dsODE_fvm.
%
% The voltage Fokker-Planck equations, refractory states, and synaptic
% mean/variance states are integrated simultaneously by ode15s.
% Cross-population names use presynaptic-to-postsynaptic order:
% IE = I -> E and EI = E -> I.
%
% Optional LIF initialization:
%   res = main_dsODE_fvm_ode(P, params, lif_result, lif_time_ms)
% lif_result may be the output of main_LIF_mini/run_LIF_model_mini, or a
% snapshot struct containing V_e, V_i, H_ee, H_ie, H_ei, and H_ii. For a
% recorded LIF result, lif_time_ms must exactly match a recorded time.
% params.refractory_stages controls an Erlang-chain approximation of the
% fixed LIF refractory time. The default 1 preserves the original
% exponential refractory pool; values such as 16 or 64 approach a fixed
% delay while remaining an ODE system.

if nargin < 2 || isempty(params)
    params = struct();
end
if nargin < 3
    lif_result = [];
end
if nargin < 4
    lif_time_ms = [];
end
params = local_defaults(params);

if isequal(size(block_prob_mat), [2 2])
    params.p_ee = local_probability(block_prob_mat(1, 1));
    params.p_ei = local_probability(block_prob_mat(1, 2)); % E -> I
    params.p_ie = local_probability(block_prob_mat(2, 1)); % I -> E
    params.p_ii = local_probability(block_prob_mat(2, 2));
elseif size(block_prob_mat, 1) >= 122 && size(block_prob_mat, 2) >= 122
    params.p_ee = local_probability(block_prob_mat(1, 1));
    params.p_ei = local_probability(block_prob_mat(1, 122)); % E -> I
    params.p_ie = local_probability(block_prob_mat(122, 1)); % I -> E
    params.p_ii = local_probability(block_prob_mat(122, 122));
else
    error('block_prob_mat must be 2-by-2 or contain E block 1 and I block 122.');
end

K_exact = params.M / params.V_bin - params.V_bin_min;
if abs(K_exact - round(K_exact)) > 1e-12 || K_exact < 2
    error('M/V_bin - V_bin_min must be an integer greater than one.');
end
params.V_bin_num = round(K_exact);
K = params.V_bin_num;
edges = params.V_bin * (params.V_bin_min + (0:K));
if abs(edges(end) - params.M) > 1e-12
    error('The upper voltage edge must equal params.M.');
end

reset_bin = find(params.V_reset >= edges(1:end-1) & ...
    params.V_reset < edges(2:end), 1, 'last');
if isempty(reset_bin)
    error('params.V_reset must lie inside the voltage domain.');
end

if strcmp(params.time_integrator, 'fixed_queue_ssprk3')
    if ~isempty(lif_result) || ~isempty(lif_time_ms)
        error('fixed_queue_ssprk3 currently supports reset initialization only.');
    end
    res_dsODE = local_fixed_queue_solver(params, edges, reset_bin);
    [res_dsODE.E_sp,res_dsODE.I_sp] = local_generate_spikes( ...
        res_dsODE.t,res_dsODE.fr_e,res_dsODE.fr_i,params,'interval_endpoint');
    return
end

L = local_layout(K, params.refractory_stages);
[y0, init_info] = local_initial_state(lif_result, lif_time_ms, ...
    params, L, edges, reset_bin);

tout = (0:params.dt:params.duration_time).';
if tout(end) < params.duration_time
    tout = [tout; params.duration_time];
end
ode_options = odeset('RelTol', params.rel_tol, ...
    'AbsTol', params.abs_tol, 'MaxStep', params.max_step, ...
    'NonNegative', L.nonnegative);
if params.use_jpattern
    ode_options = odeset(ode_options, 'JPattern', ...
        local_jacobian_pattern(L, K, reset_bin));
end

rhs = @(t, y) local_full_rhs(t, y, params, L, edges, reset_bin);
solver = str2func(params.ode_solver);
[t, Y] = solver(rhs, tout, y0, ode_options);
res_dsODE = local_pack_result(t, Y, params, L, edges, reset_bin);
[res_dsODE.E_sp,res_dsODE.I_sp] = local_generate_spikes( ...
    res_dsODE.t,res_dsODE.fr_e,res_dsODE.fr_i,params,'trapezoidal');
res_dsODE.initial_state = init_info.state;
res_dsODE.meta.initialization = rmfield(init_info, 'state');
end


function params = local_defaults(params)
defaults.J_ex = 5;
defaults.M = 100;
defaults.Mr = 66;
defaults.ne = 289;
defaults.ni = 100;
defaults.dt = 0.1;
defaults.duration_time = 300;
defaults.tau_ee = 1;
defaults.tau_ei = 1;
defaults.tau_i = 10;
defaults.tau_r = 2;
defaults.tau_m = 20;

% defaults.s_ee = 5;
% defaults.s_ie = 5;
% defaults.s_ei = 5;
% defaults.s_ii = 5;
defaults.s_ee = 5.162;
defaults.s_ei = 3.463;
defaults.s_ie = 3.293;
defaults.s_ii = 4.674;

defaults.V_bin = 5;
defaults.V_bin_min = -10;
defaults.V_reset = 0;
defaults.rel_tol = 1e-6;
defaults.abs_tol = 1e-8;
defaults.max_step = 0.1;
defaults.stochastic_synaptic_decay = true;
defaults.include_synaptic_diffusion = true;
defaults.refractory_stages = 1;
defaults.ode_solver = 'ode15s';
defaults.use_jpattern = false;
defaults.time_integrator = 'ode';
defaults.fixed_cfl = 0.2;
defaults.fixed_max_step = 0.1;
defaults.rng_seed = 1;

names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(params, name) || isempty(params.(name))
        params.(name) = defaults.(name);
    end
end

positive_names = {'M','ne','ni','dt','duration_time','tau_ee','tau_ei', ...
    'tau_i','tau_r','tau_m','V_bin','rel_tol','abs_tol','max_step', ...
    'refractory_stages','fixed_cfl','fixed_max_step'};
for i = 1:numel(positive_names)
    name = positive_names{i};
    if params.(name) <= 0
        error('params.%s must be positive.', name);
    end
end
if params.refractory_stages ~= round(params.refractory_stages)
    error('params.refractory_stages must be a positive integer.');
end
if ~isscalar(params.rng_seed) || ~isfinite(params.rng_seed) || ...
        params.rng_seed < 0 || params.rng_seed ~= round(params.rng_seed)
    error('params.rng_seed must be a nonnegative integer scalar.');
end
flag_names = {'stochastic_synaptic_decay','include_synaptic_diffusion'};
for i = 1:numel(flag_names)
    name = flag_names{i};
    if ~isscalar(params.(name)) || ...
            ~(islogical(params.(name)) || ...
            (isnumeric(params.(name)) && any(params.(name) == [0, 1])))
        error('params.%s must be a logical scalar.', name);
    end
    params.(name) = logical(params.(name));
end
params.ode_solver = lower(char(params.ode_solver));
if ~ismember(params.ode_solver, {'ode15s','ode23tb','ode23s','ode23t'})
    error('params.ode_solver must be ode15s, ode23tb, ode23s, or ode23t.');
end
if ~isscalar(params.use_jpattern) || ...
        ~(islogical(params.use_jpattern) || ...
        (isnumeric(params.use_jpattern) && any(params.use_jpattern == [0, 1])))
    error('params.use_jpattern must be a logical scalar.');
end
params.use_jpattern = logical(params.use_jpattern);
params.time_integrator = lower(char(params.time_integrator));
if ~ismember(params.time_integrator, {'ode','fixed_queue_ssprk3'})
    error('params.time_integrator must be ode or fixed_queue_ssprk3.');
end
if strcmp(params.time_integrator,'fixed_queue_ssprk3') && ...
        abs(round(params.tau_r/params.dt)*params.dt-params.tau_r) > 1e-12
    error('fixed_queue_ssprk3 requires tau_r/dt to be an integer.');
end
end


function p = local_probability(p)
p = double(p);
if ~isscalar(p) || ~isfinite(p)
    error('Selected connection probabilities must be finite scalars.');
end
p = min(max(p, 0), 1);
end


function L = local_layout(K, refractory_stages)
cursor = 0;
L.N_e = cursor + (1:K); cursor = cursor + K;
L.M_e = cursor + (1:K); cursor = cursor + K;
L.R_e = cursor + (1:refractory_stages); cursor = cursor + refractory_stages;
L.N_i = cursor + (1:K); cursor = cursor + K;
L.M_i = cursor + (1:K); cursor = cursor + K;
L.R_i = cursor + (1:refractory_stages); cursor = cursor + refractory_stages;

names = {'H_ee','Q_ee','H_ei','Q_ei','H_ie','Q_ie','H_ii','Q_ii'};
for i = 1:numel(names)
    cursor = cursor + 1;
    L.(names{i}) = cursor;
end
L.n_state = cursor;
L.nonnegative = [L.N_e, L.R_e, L.N_i, L.R_i, ...
    L.H_ee, L.Q_ee, L.H_ei, L.Q_ei, ...
    L.H_ie, L.Q_ie, L.H_ii, L.Q_ii];
end


function pattern = local_jacobian_pattern(L, K, reset_bin)
pattern = sparse(L.n_state, L.n_state);
syn_all = [L.H_ee,L.Q_ee,L.H_ei,L.Q_ei, ...
    L.H_ie,L.Q_ie,L.H_ii,L.Q_ii];
syn_e = [L.H_ee,L.Q_ee,L.H_ie,L.Q_ie];
syn_i = [L.H_ei,L.Q_ei,L.H_ii,L.Q_ii];

for bin = 1:K
    nearby = max(1,bin-1):min(K,bin+1);
    rows_e = [L.N_e(bin),L.M_e(bin)];
    cols_e = [L.N_e(nearby),L.M_e(nearby),syn_e];
    pattern(rows_e,cols_e) = 1;
    rows_i = [L.N_i(bin),L.M_i(bin)];
    cols_i = [L.N_i(nearby),L.M_i(nearby),syn_i];
    pattern(rows_i,cols_i) = 1;
end
pattern([L.N_e(reset_bin),L.M_e(reset_bin)],L.R_e(end)) = 1;
pattern([L.N_i(reset_bin),L.M_i(reset_bin)],L.R_i(end)) = 1;

pattern(L.R_e(1),[L.R_e(1),L.N_e(end),L.M_e(end),syn_e]) = 1;
pattern(L.R_i(1),[L.R_i(1),L.N_i(end),L.M_i(end),syn_i]) = 1;
for stage = 2:numel(L.R_e)
    pattern(L.R_e(stage),L.R_e(stage-1:stage)) = 1;
    pattern(L.R_i(stage),L.R_i(stage-1:stage)) = 1;
end

source_e_rows = [L.H_ee,L.Q_ee,L.H_ei,L.Q_ei];
source_i_rows = [L.H_ie,L.Q_ie,L.H_ii,L.Q_ii];
pattern(source_e_rows,[L.N_e(end),L.M_e(end),syn_all]) = 1;
pattern(source_i_rows,[L.N_i(end),L.M_i(end),syn_all]) = 1;
pattern = spones(pattern);
end


function [y0, info] = local_initial_state(lif_result, lif_time_ms, ...
        p, L, edges, reset_bin)
y0 = zeros(L.n_state, 1);
if isempty(lif_result)
    if ~isempty(lif_time_ms)
        error('lif_time_ms was provided without a LIF result.');
    end
    y0(L.N_e(reset_bin)) = p.ne;
    y0(L.M_e(reset_bin)) = p.ne * p.V_reset;
    y0(L.N_i(reset_bin)) = p.ni;
    y0(L.M_i(reset_bin)) = p.ni * p.V_reset;
    info = local_initial_state_info('reset', NaN, NaN, y0, L, 0, 0);
    return
end

[snapshot, used_time_ms] = local_select_lif_snapshot(lif_result, lif_time_ms);
required = {'V_e','V_i','H_ee','H_ie','H_ei','H_ii'};
for k = 1:numel(required)
    if ~isfield(snapshot, required{k})
        error('The selected LIF state is missing field %s.', required{k});
    end
end

V_e = local_finite_vector(snapshot.V_e, 'V_e');
V_i = local_finite_vector(snapshot.V_i, 'V_i');
if numel(V_e) ~= p.ne || numel(V_i) ~= p.ni
    error(['LIF state population sizes (%d E, %d I) do not match ' ...
        'params.ne/params.ni (%d E, %d I).'], ...
        numel(V_e), numel(V_i), p.ne, p.ni);
end

tau_r_lif = local_lif_tau_r(lif_result, p.tau_r);
ref_e = local_refractory_mask(snapshot, lif_result, 'E', ...
    used_time_ms, tau_r_lif, p.ne);
ref_i = local_refractory_mask(snapshot, lif_result, 'I', ...
    used_time_ms, tau_r_lif, p.ni);
[N_e, M_e, clipped_e] = local_project_voltage(V_e(~ref_e), edges);
[N_i, M_i, clipped_i] = local_project_voltage(V_i(~ref_i), edges);

y0(L.N_e) = N_e;
y0(L.M_e) = M_e;
y0(L.R_e) = local_refractory_stage_counts(snapshot, 'ref_e', ...
    ref_e, p.refractory_stages);
y0(L.N_i) = N_i;
y0(L.M_i) = M_i;
y0(L.R_i) = local_refractory_stage_counts(snapshot, 'ref_i', ...
    ref_i, p.refractory_stages);
[y0(L.H_ee), y0(L.Q_ee)] = local_synaptic_moments( ...
    snapshot.H_ee, p.ne, 'H_ee');
[y0(L.H_ie), y0(L.Q_ie)] = local_synaptic_moments( ...
    snapshot.H_ie, p.ne, 'H_ie');
[y0(L.H_ei), y0(L.Q_ei)] = local_synaptic_moments( ...
    snapshot.H_ei, p.ni, 'H_ei');
[y0(L.H_ii), y0(L.Q_ii)] = local_synaptic_moments( ...
    snapshot.H_ii, p.ni, 'H_ii');

if sum(N_e) + sum(y0(L.R_e)) ~= p.ne || ...
        sum(N_i) + sum(y0(L.R_i)) ~= p.ni
    error('LIF-to-FVM projection did not conserve population mass.');
end
requested_time_ms = lif_time_ms;
if isempty(requested_time_ms)
    requested_time_ms = used_time_ms;
end
info = local_initial_state_info('lif', requested_time_ms, ...
    used_time_ms, y0, L, clipped_e, clipped_i);
info.lif_tau_r_ms = tau_r_lif;
end


function info = local_initial_state_info(mode, requested_time_ms, ...
        used_time_ms, y0, L, clipped_e, clipped_i)
info.mode = mode;
info.requested_time_ms = requested_time_ms;
info.used_time_ms = used_time_ms;
info.clipped_voltage_count_e = clipped_e;
info.clipped_voltage_count_i = clipped_i;
state.n_e = y0(L.N_e).';
state.V_e_all = y0(L.M_e).';
state.ref_e = sum(y0(L.R_e));
state.ref_e_stages = y0(L.R_e).';
state.n_i = y0(L.N_i).';
state.V_i_all = y0(L.M_i).';
state.ref_i = sum(y0(L.R_i));
state.ref_i_stages = y0(L.R_i).';
state.H_ee_mean = y0(L.H_ee); state.H_ee_var = y0(L.Q_ee);
state.H_ei_mean = y0(L.H_ei); state.H_ei_var = y0(L.Q_ei);
state.H_ie_mean = y0(L.H_ie); state.H_ie_var = y0(L.Q_ie);
state.H_ii_mean = y0(L.H_ii); state.H_ii_var = y0(L.Q_ii);
info.state = state;
end


function [snapshot, used_time_ms] = local_select_lif_snapshot(lif_result, requested_time_ms)
required = {'V_e','V_i','H_ee','H_ie','H_ei','H_ii'};
is_snapshot = all(isfield(lif_result, required));
if is_snapshot
    snapshot = lif_result;
    if isfield(snapshot, 't') && isscalar(snapshot.t) && isfinite(snapshot.t)
        used_time_ms = double(snapshot.t);
    elseif ~isempty(requested_time_ms)
        used_time_ms = double(requested_time_ms);
    else
        used_time_ms = 0;
    end
    return
end

if isempty(requested_time_ms) || ~isscalar(requested_time_ms) || ...
        ~isfinite(requested_time_ms) || requested_time_ms < 0
    error('A finite nonnegative lif_time_ms is required for a recorded LIF result.');
end
requested_time_ms = double(requested_time_ms);
if requested_time_ms == 0 && isfield(lif_result, 'initial_V_e') && ...
        isfield(lif_result, 'initial_V_i')
    snapshot = struct('t', 0, 'V_e', lif_result.initial_V_e, ...
        'V_i', lif_result.initial_V_i, ...
        'H_ee', zeros(size(lif_result.initial_V_e)), ...
        'H_ie', zeros(size(lif_result.initial_V_e)), ...
        'H_ei', zeros(size(lif_result.initial_V_i)), ...
        'H_ii', zeros(size(lif_result.initial_V_i)), ...
        'ref_e', false(size(lif_result.initial_V_e)), ...
        'ref_i', false(size(lif_result.initial_V_i)));
    used_time_ms = 0;
    return
end
if ~isfield(lif_result, 'record') || isempty(lif_result.record)
    error(['The LIF result has no recorded full states. Use run_LIF_model_mini ' ...
        'with record_times containing the requested time.']);
end

times = nan(1, numel(lif_result.record));
for k = 1:numel(lif_result.record)
    if ~isempty(lif_result.record(k).t)
        times(k) = double(lif_result.record(k).t);
    end
end
[distance, idx] = min(abs(times - requested_time_ms));
tolerance = max(1e-9, 10*eps(max(1, requested_time_ms)));
if isempty(idx) || ~isfinite(distance) || distance > tolerance
    available = times(isfinite(times));
    if isempty(available)
        range_text = 'none';
    else
        range_text = sprintf('%g to %g ms', min(available), max(available));
    end
    error(['LIF time %g ms was not recorded exactly (available range: %s). ' ...
        'Set record_times when running the LIF model.'], ...
        requested_time_ms, range_text);
end
snapshot = lif_result.record(idx);
used_time_ms = times(idx);
end


function x = local_finite_vector(x, name)
x = double(x(:).');
if isempty(x) || any(~isfinite(x))
    error('LIF state field %s must be a nonempty finite vector.', name);
end
end


function tau_r = local_lif_tau_r(lif_result, fallback)
tau_r = fallback;
if isfield(lif_result, 'meta') && isstruct(lif_result.meta) && ...
        isfield(lif_result.meta, 'params') && ...
        isfield(lif_result.meta.params, 'tau_r')
    tau_r = double(lif_result.meta.params.tau_r);
end
if ~isscalar(tau_r) || ~isfinite(tau_r) || tau_r < 0
    error('The LIF refractory time must be a finite nonnegative scalar.');
end
end


function mask = local_refractory_mask(snapshot, lif_result, population, ...
        time_ms, tau_r, n)
field = ['ref_', lower(population)];
if isfield(snapshot, field) && ~isempty(snapshot.(field))
    ref = local_finite_vector(snapshot.(field), field);
    if numel(ref) ~= n
        error('LIF state field %s has the wrong number of neurons.', field);
    end
    mask = ref > 0;
    return
end
if tau_r == 0
    mask = false(1, n);
    return
end
spike_field = [upper(population), '_sp'];
if ~isfield(lif_result, spike_field)
    error(['The LIF state does not contain %s or %s; refractory neurons ' ...
        'cannot be identified.'], field, spike_field);
end
spikes = local_spike_matrix(lif_result.(spike_field), spike_field);
recent = spikes(:,2) <= time_ms + 1e-9 & spikes(:,2) > time_ms - tau_r + 1e-9;
ids = spikes(recent, 1);
if any(ids < 1 | ids > n | ids ~= round(ids))
    error('%s contains invalid neuron IDs.', spike_field);
end
mask = false(1, n);
mask(unique(ids)) = true;
end


function spikes = local_spike_matrix(spikes, name)
spikes = double(spikes);
if isempty(spikes)
    spikes = zeros(0, 2);
elseif size(spikes, 2) == 2
    % Already [event x (id,time)].
elseif size(spikes, 1) == 2
    spikes = spikes.';
else
    error('%s must be an N-by-2 or 2-by-N spike matrix.', name);
end
if any(~isfinite(spikes(:)))
    error('%s must contain only finite values.', name);
end
end


function counts = local_refractory_stage_counts(snapshot, field, mask, n_stages)
counts = zeros(n_stages, 1);
if ~any(mask)
    return
end
if isfield(snapshot, field) && numel(snapshot.(field)) == numel(mask)
    values = snapshot.(field);
    remaining_fraction = min(max(double(values(mask)), 0), 1);
    stage = min(floor((1-remaining_fraction)*n_stages)+1, n_stages);
    counts = accumarray(stage(:), 1, [n_stages, 1]);
else
    counts(1) = nnz(mask);
end
end


function [N, M1, clipped_count] = local_project_voltage(V, edges)
K = numel(edges) - 1;
V = double(V(:).');
below = V < edges(1);
above = V >= edges(end);
clipped_count = nnz(below) + nnz(above);
upper_inside = edges(end) - max(eps(edges(end)), eps);
V = min(max(V, edges(1)), upper_inside);
N = histcounts(V, edges);
if isempty(V)
    M1 = zeros(1, K);
else
    bins = discretize(V, edges);
    M1 = accumarray(bins(:), V(:), [K, 1], @sum, 0).';
end
end


function [mu, variance] = local_synaptic_moments(H, expected_n, name)
H = local_finite_vector(H, name);
if numel(H) ~= expected_n || any(H < 0)
    error('LIF state field %s must contain %d nonnegative values.', ...
        name, expected_n);
end
mu = mean(H);
variance = mean((H - mu).^2);
end


function dy = local_full_rhs(~, y, p, L, edges, reset_bin)
N_e = max(y(L.N_e).', 0);
M_e = y(L.M_e).';
R_e = max(y(L.R_e), 0);
N_i = max(y(L.N_i).', 0);
M_i = y(L.M_i).';
R_i = max(y(L.R_i), 0);

H_ee = max(y(L.H_ee), 0); Q_ee = max(y(L.Q_ee), 0);
H_ei = max(y(L.H_ei), 0); Q_ei = max(y(L.Q_ei), 0);
H_ie = max(y(L.H_ie), 0); Q_ie = max(y(L.Q_ie), 0);
H_ii = max(y(L.H_ii), 0); Q_ii = max(y(L.Q_ii), 0);

denom = p.M + p.Mr;
synaptic_diffusion = double(p.include_synaptic_diffusion);
inh_e = (p.s_ie / p.tau_i) * H_ie / denom;
a0_e = p.J_ex + (p.s_ee / p.tau_ee) * H_ee - inh_e * p.Mr;
a1_e = -1 / p.tau_m - inh_e;
b0_e = p.J_ex + synaptic_diffusion * ...
    (p.s_ee^2 / p.tau_ee^2) * Q_ee;
b2_e = synaptic_diffusion * ...
    (p.s_ie^2 / p.tau_i^2) * Q_ie / denom^2;

inh_i = (p.s_ii / p.tau_i) * H_ii / denom;
a0_i = p.J_ex + (p.s_ei / p.tau_ei) * H_ei - inh_i * p.Mr;
a1_i = -1 / p.tau_m - inh_i;
b0_i = p.J_ex + synaptic_diffusion * ...
    (p.s_ei^2 / p.tau_ei^2) * Q_ei;
b2_i = synaptic_diffusion * ...
    (p.s_ii^2 / p.tau_i^2) * Q_ii / denom^2;

release_e = local_refractory_release(R_e, p);
release_i = local_refractory_release(R_i, p);
[dN_e, dM_e, fire_e] = local_voltage_rhs(N_e, M_e, release_e, ...
    a0_e, a1_e, b0_e, b2_e, p, edges, reset_bin);
[dN_i, dM_i, fire_i] = local_voltage_rhs(N_i, M_i, release_i, ...
    a0_i, a1_i, b0_i, b2_i, p, edges, reset_bin);

dy = zeros(size(y));
dy(L.N_e) = dN_e(:);
dy(L.M_e) = dM_e(:);
dy(L.R_e) = local_refractory_rhs(R_e, fire_e, p);
dy(L.N_i) = dN_i(:);
dy(L.M_i) = dM_i(:);
dy(L.R_i) = local_refractory_rhs(R_i, fire_i, p);

dy(L.H_ee) = -H_ee / p.tau_ee + fire_e * p.p_ee;
decay_noise = double(p.stochastic_synaptic_decay);
dy(L.Q_ee) = -2*Q_ee / p.tau_ee + ...
    decay_noise * H_ee / p.tau_ee + ...
    fire_e * p.p_ee * (1-p.p_ee);
dy(L.H_ei) = -H_ei / p.tau_ei + fire_e * p.p_ei;
dy(L.Q_ei) = -2*Q_ei / p.tau_ei + ...
    decay_noise * H_ei / p.tau_ei + ...
    fire_e * p.p_ei * (1-p.p_ei);
dy(L.H_ie) = -H_ie / p.tau_i + fire_i * p.p_ie;
dy(L.Q_ie) = -2*Q_ie / p.tau_i + ...
    decay_noise * H_ie / p.tau_i + ...
    fire_i * p.p_ie * (1-p.p_ie);
dy(L.H_ii) = -H_ii / p.tau_i + fire_i * p.p_ii;
dy(L.Q_ii) = -2*Q_ii / p.tau_i + ...
    decay_noise * H_ii / p.tau_i + ...
    fire_i * p.p_ii * (1-p.p_ii);
end


function [dN, dM1, firing] = local_voltage_rhs(N, M1, release, ...
        a0, a1, b0, b2, p, edges, reset_bin)
K = numel(N);
h = p.V_bin;
left = edges(1:end-1);
right = edges(2:end);
centers = (left + right) / 2;
rho = N / h;

% First-moment linear reconstruction with a positivity limiter.
slope = 12 * (M1 - centers .* N) / h^3;
edge_delta = abs(slope) * h / 2;
theta = min(1, rho ./ max(edge_delta, eps));
slope = slope .* theta;
rho_left = max(rho - slope*h/2, 0);
rho_right = max(rho + slope*h/2, 0);

b_center = max(b0 + b2 * (centers + p.Mr).^2, 0);
g_center = b_center .* rho;
J = zeros(1, K+1);
if K > 1
    face_voltage = right(1:end-1);
    face_drift = a0 + a1 * face_voltage;
    advective = max(face_drift, 0) .* rho_right(1:end-1) + ...
        min(face_drift, 0) .* rho_left(2:end);
    diffusive = -0.5 * diff(g_center) / h;
    J(2:K) = advective + diffusive;
end

% Reflecting lower boundary and absorbing threshold boundary.
J(1) = 0;
top_drift = a0 + a1 * right(end);
top_diffusion = max(b0 + b2 * (right(end) + p.Mr)^2, 0);
J(end) = max(top_drift, 0) * rho_right(end) + ...
    top_diffusion * rho(end) / h;
firing = max(J(end), 0);

dN = J(1:end-1) - J(2:end);
dN(reset_bin) = dN(reset_bin) + release;

b_left = max(b0 + b2 * (left + p.Mr).^2, 0);
b_right = max(b0 + b2 * (right + p.Mr).^2, 0);
g_left = b_left .* rho_left;
g_right = b_right .* rho_right;
g_right(end) = 0;
integral_flux = a0*N + a1*M1 - 0.5*(g_right - g_left);
dM1 = left .* J(1:end-1) - right .* J(2:end) + integral_flux;
dM1(reset_bin) = dM1(reset_bin) + p.V_reset * release;
end


function release = local_refractory_release(R, p)
release = p.refractory_stages * R(end) / p.tau_r;
end


function dR = local_refractory_rhs(R, firing, p)
rate = p.refractory_stages / p.tau_r;
dR = rate * ([0; R(1:end-1)] - R);
dR(1) = dR(1) + firing;
end


function res = local_fixed_queue_solver(p, edges, reset_bin)
K = p.V_bin_num;
L = local_layout(K, 1);
y = zeros(L.n_state, 1);
y(L.N_e(reset_bin)) = p.ne;
y(L.M_e(reset_bin)) = p.ne*p.V_reset;
y(L.N_i(reset_bin)) = p.ni;
y(L.M_i(reset_bin)) = p.ni*p.V_reset;

queue_steps = round(p.tau_r/p.dt);
queue_e = zeros(queue_steps,1);
queue_i = zeros(queue_steps,1);
queue_position = 1;
n_steps = round(p.duration_time/p.dt);
if abs(n_steps*p.dt-p.duration_time) > 1e-12
    error('fixed_queue_ssprk3 requires duration_time/dt to be an integer.');
end

res.t = (0:n_steps).'*p.dt;
res.fr_e = zeros(n_steps+1,1);
res.fr_i = zeros(n_steps+1,1);
res.mass_error_e = zeros(n_steps+1,1);
res.mass_error_i = zeros(n_steps+1,1);
res.mass_error_e(1) = sum(y(L.N_e))+sum(queue_e)-p.ne;
res.mass_error_i(1) = sum(y(L.N_i))+sum(queue_i)-p.ni;
total_spikes_e = 0;
total_spikes_i = 0;
total_substeps = 0;
started = tic;

for step = 1:n_steps
    release_e = queue_e(queue_position)/p.dt;
    release_i = queue_i(queue_position)/p.dt;
    y(L.R_e) = 0;
    y(L.R_i) = 0;

    operator_rate = local_fixed_operator_rate(y,p,L,edges);
    n_substeps = max(ceil(p.dt/p.fixed_max_step), ...
        ceil(p.dt*operator_rate/p.fixed_cfl));
    n_substeps = max(n_substeps,1);
    ds = p.dt/n_substeps;
    rhs = @(state) local_fixed_queue_rhs(state,release_e,release_i, ...
        p,L,edges,reset_bin);

    for substep = 1:n_substeps
        y0 = y;
        y1 = local_fixed_project(y0+ds*rhs(y0),p,L,edges);
        y2 = local_fixed_project(0.75*y0+0.25*(y1+ds*rhs(y1)), ...
            p,L,edges);
        y = local_fixed_project((1/3)*y0+(2/3)*(y2+ds*rhs(y2)), ...
            p,L,edges);
    end
    total_substeps = total_substeps+n_substeps;

    spike_count_e = max(y(L.R_e),0);
    spike_count_i = max(y(L.R_i),0);
    queue_e(queue_position) = spike_count_e;
    queue_i(queue_position) = spike_count_i;
    queue_position = mod(queue_position,queue_steps)+1;
    total_spikes_e = total_spikes_e+spike_count_e;
    total_spikes_i = total_spikes_i+spike_count_i;
    res.fr_e(step+1) = spike_count_e/p.dt;
    res.fr_i(step+1) = spike_count_i/p.dt;
    res.mass_error_e(step+1) = sum(y(L.N_e))+sum(queue_e)-p.ne;
    res.mass_error_i(step+1) = sum(y(L.N_i))+sum(queue_i)-p.ni;
end

res.runtime_s = toc(started);
res.cumulative_spikes_e = total_spikes_e;
res.cumulative_spikes_i = total_spikes_i;
res.mean_rate_e_hz = 1000*total_spikes_e/(p.ne*p.duration_time);
res.mean_rate_i_hz = 1000*total_spikes_i/(p.ni*p.duration_time);
res.final_n_e = y(L.N_e).';
res.final_V_e_all = y(L.M_e).';
res.final_n_i = y(L.N_i).';
res.final_V_i_all = y(L.M_i).';
res.final_ref_e_queue = queue_e.';
res.final_ref_i_queue = queue_i.';
res.params = p;
res.meta.model = 'fixed_step_fvm_with_exact_refractory_queue';
res.meta.total_substeps = total_substeps;
res.meta.mean_substeps_per_dt = total_substeps/n_steps;
res.meta.state_count_integrated = L.n_state;
res.meta.refractory_queue_slots = 2*queue_steps;
end


function dy = local_fixed_queue_rhs(y,release_e,release_i,p,L,edges,reset_bin)
N_e = max(y(L.N_e).',0); M_e = y(L.M_e).';
N_i = max(y(L.N_i).',0); M_i = y(L.M_i).';
H_ee = max(y(L.H_ee),0); Q_ee = max(y(L.Q_ee),0);
H_ei = max(y(L.H_ei),0); Q_ei = max(y(L.Q_ei),0);
H_ie = max(y(L.H_ie),0); Q_ie = max(y(L.Q_ie),0);
H_ii = max(y(L.H_ii),0); Q_ii = max(y(L.Q_ii),0);

denom = p.M+p.Mr;
synaptic_diffusion = double(p.include_synaptic_diffusion);
inh_e = (p.s_ie/p.tau_i)*H_ie/denom;
a0_e = p.J_ex+(p.s_ee/p.tau_ee)*H_ee-inh_e*p.Mr;
a1_e = -1/p.tau_m-inh_e;
b0_e = p.J_ex+synaptic_diffusion*(p.s_ee^2/p.tau_ee^2)*Q_ee;
b2_e = synaptic_diffusion*(p.s_ie^2/p.tau_i^2)*Q_ie/denom^2;
inh_i = (p.s_ii/p.tau_i)*H_ii/denom;
a0_i = p.J_ex+(p.s_ei/p.tau_ei)*H_ei-inh_i*p.Mr;
a1_i = -1/p.tau_m-inh_i;
b0_i = p.J_ex+synaptic_diffusion*(p.s_ei^2/p.tau_ei^2)*Q_ei;
b2_i = synaptic_diffusion*(p.s_ii^2/p.tau_i^2)*Q_ii/denom^2;

[dN_e,dM_e,fire_e] = local_voltage_rhs(N_e,M_e,release_e, ...
    a0_e,a1_e,b0_e,b2_e,p,edges,reset_bin);
[dN_i,dM_i,fire_i] = local_voltage_rhs(N_i,M_i,release_i, ...
    a0_i,a1_i,b0_i,b2_i,p,edges,reset_bin);

dy = zeros(size(y));
dy(L.N_e) = dN_e(:); dy(L.M_e) = dM_e(:); dy(L.R_e) = fire_e;
dy(L.N_i) = dN_i(:); dy(L.M_i) = dM_i(:); dy(L.R_i) = fire_i;
decay_noise = double(p.stochastic_synaptic_decay);
dy(L.H_ee) = -H_ee/p.tau_ee+fire_e*p.p_ee;
dy(L.Q_ee) = -2*Q_ee/p.tau_ee+decay_noise*H_ee/p.tau_ee+ ...
    fire_e*p.p_ee*(1-p.p_ee);
dy(L.H_ei) = -H_ei/p.tau_ei+fire_e*p.p_ei;
dy(L.Q_ei) = -2*Q_ei/p.tau_ei+decay_noise*H_ei/p.tau_ei+ ...
    fire_e*p.p_ei*(1-p.p_ei);
dy(L.H_ie) = -H_ie/p.tau_i+fire_i*p.p_ie;
dy(L.Q_ie) = -2*Q_ie/p.tau_i+decay_noise*H_ie/p.tau_i+ ...
    fire_i*p.p_ie*(1-p.p_ie);
dy(L.H_ii) = -H_ii/p.tau_i+fire_i*p.p_ii;
dy(L.Q_ii) = -2*Q_ii/p.tau_i+decay_noise*H_ii/p.tau_i+ ...
    fire_i*p.p_ii*(1-p.p_ii);
end


function y = local_fixed_project(y,p,L,edges)
left = edges(1:end-1).';
right = edges(2:end).';
N_e = max(y(L.N_e),0); N_i = max(y(L.N_i),0);
y(L.N_e) = N_e;
y(L.M_e) = min(max(y(L.M_e),N_e.*left),N_e.*right);
y(L.N_i) = N_i;
y(L.M_i) = min(max(y(L.M_i),N_i.*left),N_i.*right);
y(L.R_e) = max(y(L.R_e),0);
y(L.R_i) = max(y(L.R_i),0);
positive = [L.H_ee,L.Q_ee,L.H_ei,L.Q_ei, ...
    L.H_ie,L.Q_ie,L.H_ii,L.Q_ii];
y(positive) = max(y(positive),0);
end


function rate = local_fixed_operator_rate(y,p,L,edges)
H_ee = max(y(L.H_ee),0); Q_ee = max(y(L.Q_ee),0);
H_ei = max(y(L.H_ei),0); Q_ei = max(y(L.Q_ei),0);
H_ie = max(y(L.H_ie),0); Q_ie = max(y(L.Q_ie),0);
H_ii = max(y(L.H_ii),0); Q_ii = max(y(L.Q_ii),0);
denom = p.M+p.Mr;
synaptic_diffusion = double(p.include_synaptic_diffusion);
inh_e = (p.s_ie/p.tau_i)*H_ie/denom;
a0_e = p.J_ex+(p.s_ee/p.tau_ee)*H_ee-inh_e*p.Mr;
a1_e = -1/p.tau_m-inh_e;
b0_e = p.J_ex+synaptic_diffusion*(p.s_ee^2/p.tau_ee^2)*Q_ee;
b2_e = synaptic_diffusion*(p.s_ie^2/p.tau_i^2)*Q_ie/denom^2;
inh_i = (p.s_ii/p.tau_i)*H_ii/denom;
a0_i = p.J_ex+(p.s_ei/p.tau_ei)*H_ei-inh_i*p.Mr;
a1_i = -1/p.tau_m-inh_i;
b0_i = p.J_ex+synaptic_diffusion*(p.s_ei^2/p.tau_ei^2)*Q_ei;
b2_i = synaptic_diffusion*(p.s_ii^2/p.tau_i^2)*Q_ii/denom^2;
centers = 0.5*(edges(1:end-1)+edges(2:end));
rate_e = max(abs(a0_e+a1_e*edges))/p.V_bin+ ...
    max(b0_e+b2_e*(centers+p.Mr).^2)/p.V_bin^2;
rate_i = max(abs(a0_i+a1_i*edges))/p.V_bin+ ...
    max(b0_i+b2_i*(centers+p.Mr).^2)/p.V_bin^2;
rate = max([rate_e,rate_i,1/p.tau_ee,1/p.tau_ei,1/p.tau_i]);
end


function [E_sp,I_sp] = local_generate_spikes(t,fr_e,fr_i,p,method)
rng_state = rng;
restore_rng = onCleanup(@() rng(rng_state));
rng(p.rng_seed,'twister');
E_sp = local_population_spikes(t,fr_e,p.ne,method);
I_sp = local_population_spikes(t,fr_i,p.ni,method);
end


function spikes = local_population_spikes(t,rate,population_size,method)
dt = diff(t(:));
rate = max(rate(:),0);
if strcmp(method,'interval_endpoint')
    expected = rate(2:end).*dt;
else
    expected = 0.5*(rate(1:end-1)+rate(2:end)).*dt;
end
base = floor(expected);
counts = base+(rand(size(expected)) < expected-base);
spikes = zeros(2,sum(counts));
cursor = 0;
for k = 1:numel(counts)
    count = counts(k);
    if count == 0
        continue
    elseif count <= population_size
        ids = randperm(population_size,count);
    else
        ids = randi(population_size,1,count);
    end
    destination = cursor+(1:count);
    spikes(1,destination) = ids;
    spikes(2,destination) = t(k+1);
    cursor = cursor+count;
end
end


function res = local_pack_result(t, Y, p, L, edges, reset_bin)
T = numel(t);
K = p.V_bin_num;
res.t = t;
res.n_e = Y(:, L.N_e);
res.V_e_all = Y(:, L.M_e);
res.ref_e_stages = Y(:, L.R_e);
res.ref_e = sum(res.ref_e_stages, 2);
res.n_i = Y(:, L.N_i);
res.V_i_all = Y(:, L.M_i);
res.ref_i_stages = Y(:, L.R_i);
res.ref_i = sum(res.ref_i_stages, 2);
res.V_e_mean = res.V_e_all ./ max(res.n_e, eps);
res.V_i_mean = res.V_i_all ./ max(res.n_i, eps);
res.V_e_mean(res.n_e <= eps) = 0;
res.V_i_mean(res.n_i <= eps) = 0;

res.H_ee_mean = Y(:, L.H_ee); res.H_ee_var = Y(:, L.Q_ee);
res.H_ei_mean = Y(:, L.H_ei); res.H_ei_var = Y(:, L.Q_ei);
res.H_ie_mean = Y(:, L.H_ie); res.H_ie_var = Y(:, L.Q_ie);
res.H_ii_mean = Y(:, L.H_ii); res.H_ii_var = Y(:, L.Q_ii);
res.fr_e = zeros(T, 1);
res.fr_i = zeros(T, 1);
res.I_e_mean = zeros(T, K);
res.I_e_var = zeros(T, K);
res.I_i_mean = zeros(T, K);
res.I_i_var = zeros(T, K);

denom = p.M + p.Mr;
synaptic_diffusion = double(p.include_synaptic_diffusion);
for it = 1:T
    H_ee = max(Y(it, L.H_ee), 0); Q_ee = max(Y(it, L.Q_ee), 0);
    H_ei = max(Y(it, L.H_ei), 0); Q_ei = max(Y(it, L.Q_ei), 0);
    H_ie = max(Y(it, L.H_ie), 0); Q_ie = max(Y(it, L.Q_ie), 0);
    H_ii = max(Y(it, L.H_ii), 0); Q_ii = max(Y(it, L.Q_ii), 0);

    res.I_e_mean(it,:) = p.J_ex + (p.s_ee/p.tau_ee)*H_ee - ...
        (p.s_ie/p.tau_i)*H_ie .* ...
        (res.V_e_mean(it,:) + p.Mr) / denom;
    res.I_e_var(it,:) = p.J_ex + synaptic_diffusion * ...
        (p.s_ee^2/p.tau_ee^2)*Q_ee + synaptic_diffusion * ...
        (p.s_ie^2/p.tau_i^2)*Q_ie .* ...
        (res.V_e_mean(it,:) + p.Mr).^2 / denom^2;
    res.I_i_mean(it,:) = p.J_ex + (p.s_ei/p.tau_ei)*H_ei - ...
        (p.s_ii/p.tau_i)*H_ii .* ...
        (res.V_i_mean(it,:) + p.Mr) / denom;
    res.I_i_var(it,:) = p.J_ex + synaptic_diffusion * ...
        (p.s_ei^2/p.tau_ei^2)*Q_ei + synaptic_diffusion * ...
        (p.s_ii^2/p.tau_i^2)*Q_ii .* ...
        (res.V_i_mean(it,:) + p.Mr).^2 / denom^2;

    N_e = max(Y(it, L.N_e), 0);
    M_e = Y(it, L.M_e);
    N_i = max(Y(it, L.N_i), 0);
    M_i = Y(it, L.M_i);
    inh_e = (p.s_ie/p.tau_i)*H_ie/denom;
    inh_i = (p.s_ii/p.tau_i)*H_ii/denom;
    release_e = local_refractory_release(max(Y(it,L.R_e),0).', p);
    release_i = local_refractory_release(max(Y(it,L.R_i),0).', p);
    [~,~,res.fr_e(it)] = local_voltage_rhs(N_e, M_e, release_e, ...
        p.J_ex+(p.s_ee/p.tau_ee)*H_ee-inh_e*p.Mr, ...
        -1/p.tau_m-inh_e, ...
        p.J_ex+synaptic_diffusion*(p.s_ee^2/p.tau_ee^2)*Q_ee, ...
        synaptic_diffusion*(p.s_ie^2/p.tau_i^2)*Q_ie/denom^2, ...
        p, edges, reset_bin);
    [~,~,res.fr_i(it)] = local_voltage_rhs(N_i, M_i, release_i, ...
        p.J_ex+(p.s_ei/p.tau_ei)*H_ei-inh_i*p.Mr, ...
        -1/p.tau_m-inh_i, ...
        p.J_ex+synaptic_diffusion*(p.s_ei^2/p.tau_ei^2)*Q_ei, ...
        synaptic_diffusion*(p.s_ii^2/p.tau_i^2)*Q_ii/denom^2, ...
        p, edges, reset_bin);
end

res.mass_error_e = sum(res.n_e, 2) + res.ref_e - p.ne;
res.mass_error_i = sum(res.n_i, 2) + res.ref_i - p.ni;
res.params = p;
res.meta.model = 'fully_coupled_finite_volume_moment_ODE';
res.meta.state_equation = 'dXdt=F(X)';
res.meta.voltage_edges = edges;
res.meta.connection_direction = ...
    'ie: I-to-E, ei: E-to-I, rows presynaptic';
end
