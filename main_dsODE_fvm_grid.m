function res_fvm_grid = main_dsODE_fvm_grid(sigmaEE,sigmaEI,sigmaIE,sigmaII)
%MAIN_DSODE_FVM_GRID Run the finite-volume model on all 121 cortical blocks.
%
% Usage:
%   res_fvm_grid = main_dsODE_fvm_grid(sigmaEE,sigmaEI,sigmaIE,sigmaII)
% EI means E->I and IE means I->E throughout this package.

total_started = tic;
if nargin ~= 4
    error('Four sigma values are required: EE, EI, IE and II.');
end
sigma = [sigmaEE,sigmaEI,sigmaIE,sigmaII];
if any(~isfinite(sigma)) || any(sigma <= 0)
    error('All sigma values must be finite positive scalars.');
end
cfg = local_cfg_defaults(struct());

project_root = fileparts(mfilename('fullpath'));
connection_dir = fullfile(project_root,'connection_mat');
md = load(fullfile(connection_dir,'network_layout.mat'), ...
    'E_group', 'I_group');
E_full = double(md.E_group(:)');
I_full = double(md.I_group(:)');
n_blocks = numel(E_full);
selected_blocks = 1:n_blocks;
B = numel(selected_blocks);
block_prob = local_load_probability_matrix(connection_dir,sigma);
probability_source = 'connection_mat';
block_prob = min(max(block_prob, 0), 1);

p = local_parameter_defaults(1000);
if isempty(cfg.visualization_interval)
    cfg.visualization_interval = p.dt;
end
if strcmpi(cfg.refractory_mode,'fixed_queue')
    if ~strcmpi(cfg.solver,'ssprk3')
        error('fixed_queue refractory mode requires cfg.solver = ssprk3.');
    end
    if abs(round(p.tau_r/p.dt)*p.dt-p.tau_r) > 1e-12
        error('fixed_queue requires tau_r/dt to be an integer.');
    end
    if abs(round(p.duration_time/p.dt)*p.dt-p.duration_time) > 1e-12
        error('fixed_queue requires duration_time/dt to be an integer.');
    end
end
p.Ne = E_full(selected_blocks);
p.Ni = I_full(selected_blocks);
p.p_ee = block_prob(1:B, 1:B);
p.p_ei = block_prob(1:B, B+1:2*B);       % E -> I
p.p_ie = block_prob(B+1:2*B, 1:B);       % I -> E
p.p_ii = block_prob(B+1:2*B, B+1:2*B);
p.p_ee_var = p.p_ee .* (1-p.p_ee);
p.p_ei_var = p.p_ei .* (1-p.p_ei);
p.p_ie_var = p.p_ie .* (1-p.p_ie);
p.p_ii_var = p.p_ii .* (1-p.p_ii);

K_exact = p.M / p.V_bin - p.V_bin_min;
if abs(K_exact-round(K_exact)) > 1e-12 || K_exact < 2
    error('M/V_bin - V_bin_min must be an integer greater than one.');
end
p.V_bin_num = round(K_exact);
K = p.V_bin_num;
edges = p.V_bin * (p.V_bin_min + (0:K));
reset_bin = find(p.V_reset >= edges(1:end-1) & ...
    p.V_reset < edges(2:end), 1, 'last');
if isempty(reset_bin), error('V_reset must lie inside the voltage domain.'); end

L = local_layout(K, B);
y0 = zeros(L.n_state, 1);
y0(L.N_e(reset_bin, :)) = p.Ne;
y0(L.M_e(reset_bin, :)) = p.Ne * p.V_reset;
y0(L.N_i(reset_bin, :)) = p.Ni;
y0(L.M_i(reset_bin, :)) = p.Ni * p.V_reset;

if strcmpi(cfg.output_mode, 'sampled')
    tout = (0:p.dt:p.duration_time).';
    if tout(end) < p.duration_time, tout = [tout; p.duration_time]; end
elseif strcmpi(cfg.output_mode, 'visualization')
    tout = (0:cfg.visualization_interval:p.duration_time).';
    if abs(tout(end)-p.duration_time) > 1e-12
        tout = [tout; p.duration_time];
    end
else
    tout = [0; p.duration_time/2; p.duration_time];
end

rhs = @(t,y) local_rhs(t, y, p, L, edges, reset_bin);
started = tic;
if strcmpi(cfg.solver, 'ssprk3')
    if strcmpi(cfg.refractory_mode,'fixed_queue')
        [t,Y,solver_steps] = local_ssprk3_fixed_queue( ...
            tout,y0,p,L,edges,reset_bin,cfg);
    else
        [t,Y,solver_steps] = local_ssprk3(rhs,tout,y0,p,L,edges,cfg);
    end
else
    jpattern = local_jpattern(L, K, B);
    options = odeset('RelTol', p.rel_tol, 'AbsTol', p.abs_tol, ...
        'MaxStep', p.max_step, 'NonNegative', L.nonnegative, ...
        'JPattern', jpattern, 'Stats', cfg.solver_stats);
    [t, Y] = ode15s(rhs, tout, y0, options);
    solver_steps = NaN;
end
runtime_s = toc(started);

res_fvm_grid = local_pack(t, Y, p, L, edges, reset_bin);
res_fvm_grid.runtime_s = runtime_s;
[res_fvm_grid.E_sp,res_fvm_grid.I_sp] = local_generate_spikes( ...
    res_fvm_grid.cumulative_spikes_e_group,res_fvm_grid.cumulative_spikes_i_group,t,p.Ne,p.Ni);
res_fvm_grid.params = p;
res_fvm_grid.meta.model = 'multiblock_fully_coupled_finite_volume_moment_ODE';
res_fvm_grid.meta.selected_blocks = selected_blocks;
res_fvm_grid.meta.E_group_selected = p.Ne;
res_fvm_grid.meta.I_group_selected = p.Ni;
res_fvm_grid.meta.block_prob_mini = block_prob;
res_fvm_grid.meta.block_prob_source = probability_source;
res_fvm_grid.meta.sigma = sigma;
res_fvm_grid.meta.interaction_name_convention = 'pre_to_post';
res_fvm_grid.meta.output_mode = cfg.output_mode;
res_fvm_grid.meta.solver = cfg.solver;
res_fvm_grid.meta.refractory_mode = cfg.refractory_mode;
res_fvm_grid.meta.solver_steps = solver_steps;
res_fvm_grid.meta.state_count = L.n_state;
res_fvm_grid.meta.spike_id_convention = struct( ...
    'E_sp','mini_local_E_1_to_ne','I_sp','mini_local_I_1_to_ni');
res_fvm_grid.meta.params = p;
res_fvm_grid.total_runtime_s = toc(total_started);
fprintf(['main_dsODE_fvm_grid finished: solver %.3f s, ', ...
    'total %.3f s.\n'],res_fvm_grid.runtime_s,res_fvm_grid.total_runtime_s);
end


function cfg = local_cfg_defaults(cfg)
if ~isfield(cfg, 'output_mode') || isempty(cfg.output_mode), cfg.output_mode = 'visualization'; end
if ~any(strcmpi(cfg.output_mode, {'compact','sampled','visualization'}))
    error('cfg.output_mode must be compact, sampled, or visualization.');
end
if ~isfield(cfg, 'solver_stats') || isempty(cfg.solver_stats), cfg.solver_stats = 'off'; end
if ~isfield(cfg, 'solver') || isempty(cfg.solver), cfg.solver = 'ssprk3'; end
if ~any(strcmpi(cfg.solver, {'ode15s','ssprk3'}))
    error('cfg.solver must be ode15s or ssprk3.');
end
if ~isfield(cfg, 'cfl') || isempty(cfg.cfl), cfg.cfl = 0.2; end
if ~isfield(cfg, 'explicit_max_step') || isempty(cfg.explicit_max_step)
    cfg.explicit_max_step = 0.1;
end
if ~isfield(cfg, 'refractory_mode') || isempty(cfg.refractory_mode)
    cfg.refractory_mode = 'fixed_queue';
end
if ~any(strcmpi(cfg.refractory_mode, {'exponential','fixed_queue'}))
    error('cfg.refractory_mode must be exponential or fixed_queue.');
end
if ~isfield(cfg, 'visualization_interval') || isempty(cfg.visualization_interval)
    cfg.visualization_interval = [];
end
if ~isempty(cfg.visualization_interval) && cfg.visualization_interval <= 0
    error('cfg.visualization_interval must be positive.');
end
end


function p = local_parameter_defaults(duration_time)
p = struct('J_ex',5, 'M',100, 'Mr',66, 'dt',0.1, ...
    'duration_time',duration_time, 'tau_ee',3, 'tau_ei',3, ...
    'tau_i',10, 'tau_r',2, 'tau_m',20, 's_ee',3, 's_ei',4, ...
    's_ie',8, 's_ii',8, 'V_bin',5, 'V_bin_min',-10, ...
    'V_reset',0, 'rel_tol',1e-6, 'abs_tol',1e-8, 'max_step',0.1);
end


function matrix = local_load_probability_matrix(connection_dir,sigma)
types = {'EE','EI','IE','II'};
parts = cell(1,4);
for k = 1:4
    path = fullfile(connection_dir,sprintf('%s_sig%g_prob_mat.mat',types{k},sigma(k)));
    if ~isfile(path)
        error('Connection probability file not found: %s',path);
    end
    data = load(path,[types{k},'_prob_mat']);
    parts{k} = double(data.([types{k},'_prob_mat']));
end
matrix = [parts{1},parts{2};parts{3},parts{4}];
end


function L = local_layout(K, B)
cursor = 0;
L.N_e = reshape(cursor+(1:K*B), K, B); cursor = cursor+K*B;
L.M_e = reshape(cursor+(1:K*B), K, B); cursor = cursor+K*B;
L.R_e = cursor+(1:B); cursor = cursor+B;
L.N_i = reshape(cursor+(1:K*B), K, B); cursor = cursor+K*B;
L.M_i = reshape(cursor+(1:K*B), K, B); cursor = cursor+K*B;
L.R_i = cursor+(1:B); cursor = cursor+B;
names = {'H_ee','Q_ee','H_ie','Q_ie','H_ei','Q_ei','H_ii','Q_ii', ...
    'C_e','C_i'};
for k = 1:numel(names)
    L.(names{k}) = cursor+(1:B); cursor = cursor+B;
end
L.n_state = cursor;
L.nonnegative = [L.N_e(:); L.R_e(:); L.N_i(:); L.R_i(:); ...
    L.H_ee(:); L.Q_ee(:); L.H_ie(:); L.Q_ie(:); ...
    L.H_ei(:); L.Q_ei(:); L.H_ii(:); L.Q_ii(:); ...
    L.C_e(:); L.C_i(:)].';
end


function pattern = local_jpattern(L, K, B)
pattern = sparse(L.n_state, L.n_state);
for j = 1:B
    e_rows = [L.N_e(:,j); L.M_e(:,j); L.R_e(j); L.C_e(j)];
    e_cols = [L.N_e(:,j); L.M_e(:,j); L.R_e(j); ...
        L.H_ee(j); L.Q_ee(j); L.H_ie(j); L.Q_ie(j)];
    pattern(e_rows, e_cols) = true;
    i_rows = [L.N_i(:,j); L.M_i(:,j); L.R_i(j); L.C_i(j)];
    i_cols = [L.N_i(:,j); L.M_i(:,j); L.R_i(j); ...
        L.H_ei(j); L.Q_ei(j); L.H_ii(j); L.Q_ii(j)];
    pattern(i_rows, i_cols) = true;
end
syn_e_rows = [L.H_ee, L.Q_ee, L.H_ei, L.Q_ei];
syn_i_rows = [L.H_ie, L.Q_ie, L.H_ii, L.Q_ii];
fire_e_cols = [L.N_e(K,:), L.M_e(K,:), L.H_ee, L.Q_ee, L.H_ie, L.Q_ie];
fire_i_cols = [L.N_i(K,:), L.M_i(K,:), L.H_ei, L.Q_ei, L.H_ii, L.Q_ii];
pattern(syn_e_rows, fire_e_cols) = true;
pattern(syn_i_rows, fire_i_cols) = true;
syn_cols = [L.H_ee,L.Q_ee,L.H_ie,L.Q_ie,L.H_ei,L.Q_ei,L.H_ii,L.Q_ii];
pattern([syn_e_rows,syn_i_rows], syn_cols) = true;
end


function [t,Y,n_steps] = local_ssprk3(rhs,tout,y,Lp,L,edges,cfg)
t = tout(:);
Y = zeros(numel(t),numel(y));
Y(1,:) = y.';
current_t = t(1);
n_steps = 0;
for out_id = 2:numel(t)
    target_t = t(out_id);
    while current_t < target_t
        ds = min([cfg.explicit_max_step, target_t-current_t, ...
            local_stable_step(y,Lp,L,edges,cfg.cfl)]);
        y1 = local_nonnegative(y+ds*rhs(current_t,y),L.nonnegative);
        y2 = local_nonnegative(0.75*y+0.25*(y1+ds*rhs(current_t+ds,y1)), ...
            L.nonnegative);
        y = local_nonnegative((1/3)*y+(2/3)*(y2+ds*rhs(current_t+ds/2,y2)), ...
            L.nonnegative);
        current_t = current_t+ds;
        n_steps = n_steps+1;
    end
    Y(out_id,:) = y.';
end
end


function [t,Y,n_steps] = local_ssprk3_fixed_queue( ...
        tout,y,p,L,edges,reset_bin,cfg)
t = tout(:);
Y = zeros(numel(t),numel(y));
Y(1,:) = y.';
queue_steps = round(p.tau_r/p.dt);
queue_e = zeros(queue_steps,numel(p.Ne));
queue_i = zeros(queue_steps,numel(p.Ni));
queue_position = 1;
n_steps = 0;
output_id = 2;
n_outer_steps = round(p.duration_time/p.dt);

for outer_step = 1:n_outer_steps
    release_e = queue_e(queue_position,:)/p.dt;
    release_i = queue_i(queue_position,:)/p.dt;
    count_e_start = y(L.C_e).';
    count_i_start = y(L.C_i).';
    rhs = @(time,state) local_rhs_fixed_queue(time,state,p,L,edges, ...
        reset_bin,release_e,release_i);
    stable_step = local_stable_step(y,p,L,edges,cfg.cfl);
    n_substeps = max(ceil(p.dt/cfg.explicit_max_step), ...
        ceil(p.dt/stable_step-1e-12));
    ds = p.dt/n_substeps;
    for substep = 1:n_substeps
        elapsed = (substep-1)*ds;
        y1 = local_nonnegative(y+ds*rhs(elapsed,y),L.nonnegative);
        y2 = local_nonnegative(0.75*y+0.25*(y1+ ...
            ds*rhs(elapsed+ds,y1)),L.nonnegative);
        y = local_nonnegative((1/3)*y+(2/3)*(y2+ ...
            ds*rhs(elapsed+ds/2,y2)),L.nonnegative);
        n_steps = n_steps+1;
    end
    queue_e(queue_position,:) = max(y(L.C_e).'-count_e_start,0);
    queue_i(queue_position,:) = max(y(L.C_i).'-count_i_start,0);
    queue_position = mod(queue_position,queue_steps)+1;
    y(L.R_e) = sum(queue_e,1);
    y(L.R_i) = sum(queue_i,1);

    current_t = outer_step*p.dt;
    while output_id <= numel(t) && abs(t(output_id)-current_t) <= 1e-10
        Y(output_id,:) = y.';
        output_id = output_id+1;
    end
end
if output_id <= numel(t)
    error('Fixed-queue output times must align with the model dt.');
end
end


function dy = local_rhs_fixed_queue(t,y,p,L,edges,reset_bin,release_e,release_i)
state = y;
state(L.R_e) = release_e*p.tau_r;
state(L.R_i) = release_i*p.tau_r;
dy = local_rhs(t,state,p,L,edges,reset_bin);
dy(L.R_e) = 0;
dy(L.R_i) = 0;
end


function y = local_nonnegative(y,indices)
y(indices) = max(y(indices),0);
end


function ds = local_stable_step(y,p,L,edges,cfl)
H_ee=max(y(L.H_ee).',0); Q_ee=max(y(L.Q_ee).',0);
H_ie=max(y(L.H_ie).',0); Q_ie=max(y(L.Q_ie).',0);
H_ei=max(y(L.H_ei).',0); Q_ei=max(y(L.Q_ei).',0);
H_ii=max(y(L.H_ii).',0); Q_ii=max(y(L.Q_ii).',0);
denom=p.M+p.Mr;
inh_e=(p.s_ie/p.tau_i)*H_ie/denom;
inh_i=(p.s_ii/p.tau_i)*H_ii/denom;
a0_e=p.J_ex+(p.s_ee/p.tau_ee)*H_ee-inh_e*p.Mr;
a1_e=-1/p.tau_m-inh_e;
a0_i=p.J_ex+(p.s_ei/p.tau_ei)*H_ei-inh_i*p.Mr;
a1_i=-1/p.tau_m-inh_i;
b0_e=p.J_ex+(p.s_ee^2/p.tau_ee^2)*Q_ee;
b2_e=(p.s_ie^2/p.tau_i^2)*Q_ie/denom^2;
b0_i=p.J_ex+(p.s_ei^2/p.tau_ei^2)*Q_ei;
b2_i=(p.s_ii^2/p.tau_i^2)*Q_ii/denom^2;
voltage = [edges(1),edges(end)];
max_drift = max([abs(a0_e+a1_e*voltage(1)), ...
    abs(a0_e+a1_e*voltage(2)),abs(a0_i+a1_i*voltage(1)), ...
    abs(a0_i+a1_i*voltage(2))],[],'all');
max_diffusion = max([b0_e+b2_e*(voltage(1)+p.Mr)^2, ...
    b0_e+b2_e*(voltage(2)+p.Mr)^2,b0_i+b2_i*(voltage(1)+p.Mr)^2, ...
    b0_i+b2_i*(voltage(2)+p.Mr)^2],[],'all');
rate = max_drift/p.V_bin+max_diffusion/p.V_bin^2+ ...
    max([2/p.tau_ee,2/p.tau_ei,2/p.tau_i,1/p.tau_r]);
ds = cfl/max(rate,eps);
end


function dy = local_rhs(~, y, p, L, edges, reset_bin)
N_e = max(reshape(y(L.N_e), size(L.N_e)), 0);
M_e = reshape(y(L.M_e), size(L.M_e));
R_e = max(y(L.R_e).', 0);
N_i = max(reshape(y(L.N_i), size(L.N_i)), 0);
M_i = reshape(y(L.M_i), size(L.M_i));
R_i = max(y(L.R_i).', 0);

H_ee = max(y(L.H_ee).',0); Q_ee = max(y(L.Q_ee).',0);
H_ie = max(y(L.H_ie).',0); Q_ie = max(y(L.Q_ie).',0); % I -> E
H_ei = max(y(L.H_ei).',0); Q_ei = max(y(L.Q_ei).',0); % E -> I
H_ii = max(y(L.H_ii).',0); Q_ii = max(y(L.Q_ii).',0);

denom = p.M+p.Mr;
inh_e = (p.s_ie/p.tau_i)*H_ie/denom;
a0_e = p.J_ex+(p.s_ee/p.tau_ee)*H_ee-inh_e*p.Mr;
a1_e = -1/p.tau_m-inh_e;
b0_e = p.J_ex+(p.s_ee^2/p.tau_ee^2)*Q_ee;
b2_e = (p.s_ie^2/p.tau_i^2)*Q_ie/denom^2;

inh_i = (p.s_ii/p.tau_i)*H_ii/denom;
a0_i = p.J_ex+(p.s_ei/p.tau_ei)*H_ei-inh_i*p.Mr;
a1_i = -1/p.tau_m-inh_i;
b0_i = p.J_ex+(p.s_ei^2/p.tau_ei^2)*Q_ei;
b2_i = (p.s_ii^2/p.tau_i^2)*Q_ii/denom^2;

[dN_e,dM_e,fire_e] = local_voltage_rhs(N_e,M_e,R_e,a0_e,a1_e,b0_e,b2_e,p,edges,reset_bin);
[dN_i,dM_i,fire_i] = local_voltage_rhs(N_i,M_i,R_i,a0_i,a1_i,b0_i,b2_i,p,edges,reset_bin);

dy = zeros(size(y));
dy(L.N_e) = dN_e; dy(L.M_e) = dM_e;
dy(L.R_e) = fire_e-R_e/p.tau_r;
dy(L.N_i) = dN_i; dy(L.M_i) = dM_i;
dy(L.R_i) = fire_i-R_i/p.tau_r;

dy(L.H_ee) = -H_ee/p.tau_ee + fire_e*p.p_ee;
dy(L.Q_ee) = -2*Q_ee/p.tau_ee + H_ee/p.tau_ee + fire_e*p.p_ee_var;
dy(L.H_ie) = -H_ie/p.tau_i + fire_i*p.p_ie;
dy(L.Q_ie) = -2*Q_ie/p.tau_i + H_ie/p.tau_i + fire_i*p.p_ie_var;
dy(L.H_ei) = -H_ei/p.tau_ei + fire_e*p.p_ei;
dy(L.Q_ei) = -2*Q_ei/p.tau_ei + H_ei/p.tau_ei + fire_e*p.p_ei_var;
dy(L.H_ii) = -H_ii/p.tau_i + fire_i*p.p_ii;
dy(L.Q_ii) = -2*Q_ii/p.tau_i + H_ii/p.tau_i + fire_i*p.p_ii_var;
dy(L.C_e) = fire_e;
dy(L.C_i) = fire_i;
end


function [dN,dM,firing] = local_voltage_rhs(N,M,R,a0,a1,b0,b2,p,edges,reset_bin)
h = p.V_bin;
left = edges(1:end-1).'; right = edges(2:end).';
centers = (left+right)/2;
rho = N/h;
slope = 12*(M-centers.*N)/h^3;
theta = min(1, rho./max(abs(slope)*h/2, eps));
slope = slope.*theta;
rho_left = max(rho-slope*h/2,0);
rho_right = max(rho+slope*h/2,0);

b_center = max(b0+b2.*(centers+p.Mr).^2,0);
g_center = b_center.*rho;
J = zeros(size(N,1)+1,size(N,2));
face_drift = a0+a1.*right(1:end-1);
J(2:end-1,:) = max(face_drift,0).*rho_right(1:end-1,:) + ...
    min(face_drift,0).*rho_left(2:end,:) - 0.5*diff(g_center,1,1)/h;
top_drift = a0+a1*right(end);
top_diffusion = max(b0+b2*(right(end)+p.Mr)^2,0);
J(end,:) = max(top_drift,0).*rho_right(end,:) + top_diffusion.*rho(end,:)/h;
firing = max(J(end,:),0);

release = R/p.tau_r;
dN = J(1:end-1,:)-J(2:end,:);
dN(reset_bin,:) = dN(reset_bin,:)+release;
b_left = max(b0+b2.*(left+p.Mr).^2,0);
b_right = max(b0+b2.*(right+p.Mr).^2,0);
g_left = b_left.*rho_left;
g_right = b_right.*rho_right;
g_right(end,:) = 0;
dM = left.*J(1:end-1,:)-right.*J(2:end,:) + ...
    a0.*N+a1.*M-0.5*(g_right-g_left);
dM(reset_bin,:) = dM(reset_bin,:)+p.V_reset*release;
end


function res = local_pack(t, Y, p, L, edges, reset_bin)
T = numel(t); B = numel(p.Ne);
res.t = t;
res.fr_e = zeros(T,B); res.fr_i = zeros(T,B);
res.ref_e = Y(:,L.R_e); res.ref_i = Y(:,L.R_i);
res.n_e_group = zeros(T,B); res.n_i_group = zeros(T,B);
res.mass_error_e = zeros(T,B); res.mass_error_i = zeros(T,B);
res.n_e = zeros(T,size(L.N_e,1),B); res.n_i = zeros(T,size(L.N_i,1),B);
res.V_e_all = zeros(T,size(L.M_e,1),B); res.V_i_all = zeros(T,size(L.M_i,1),B);
for it = 1:T
    y = Y(it,:).';
    N_e = max(reshape(y(L.N_e),size(L.N_e)),0);
    M_e = reshape(y(L.M_e),size(L.M_e));
    N_i = max(reshape(y(L.N_i),size(L.N_i)),0);
    M_i = reshape(y(L.M_i),size(L.M_i));
    H_ee=max(y(L.H_ee).',0); Q_ee=max(y(L.Q_ee).',0);
    H_ie=max(y(L.H_ie).',0); Q_ie=max(y(L.Q_ie).',0);
    H_ei=max(y(L.H_ei).',0); Q_ei=max(y(L.Q_ei).',0);
    H_ii=max(y(L.H_ii).',0); Q_ii=max(y(L.Q_ii).',0);
    denom=p.M+p.Mr;
    inh_e=(p.s_ie/p.tau_i)*H_ie/denom;
    inh_i=(p.s_ii/p.tau_i)*H_ii/denom;
    [~,~,res.fr_e(it,:)] = local_voltage_rhs(N_e,M_e,max(y(L.R_e).',0), ...
        p.J_ex+(p.s_ee/p.tau_ee)*H_ee-inh_e*p.Mr,-1/p.tau_m-inh_e, ...
        p.J_ex+(p.s_ee^2/p.tau_ee^2)*Q_ee,(p.s_ie^2/p.tau_i^2)*Q_ie/denom^2,p,edges,reset_bin);
    [~,~,res.fr_i(it,:)] = local_voltage_rhs(N_i,M_i,max(y(L.R_i).',0), ...
        p.J_ex+(p.s_ei/p.tau_ei)*H_ei-inh_i*p.Mr,-1/p.tau_m-inh_i, ...
        p.J_ex+(p.s_ei^2/p.tau_ei^2)*Q_ei,(p.s_ii^2/p.tau_i^2)*Q_ii/denom^2,p,edges,reset_bin);
    res.n_e_group(it,:) = sum(N_e,1);
    res.n_i_group(it,:) = sum(N_i,1);
    res.n_e(it,:,:) = reshape(N_e,[1,size(N_e)]);
    res.n_i(it,:,:) = reshape(N_i,[1,size(N_i)]);
    res.V_e_all(it,:,:) = reshape(M_e,[1,size(M_e)]);
    res.V_i_all(it,:,:) = reshape(M_i,[1,size(M_i)]);
    res.mass_error_e(it,:) = res.n_e_group(it,:)+res.ref_e(it,:)-p.Ne;
    res.mass_error_i(it,:) = res.n_i_group(it,:)+res.ref_i(it,:)-p.Ni;
end
res.spike_count_e = Y(end,L.C_e);
res.spike_count_i = Y(end,L.C_i);
res.cumulative_spikes_e_group = Y(:,L.C_e);
res.cumulative_spikes_i_group = Y(:,L.C_i);
interval_ms = diff(t);
res.interval_time_ms = t(2:end);
res.interval_rate_e_hz_group = diff(res.cumulative_spikes_e_group,1,1) ./ ...
    ((interval_ms/1000).*p.Ne);
res.interval_rate_i_hz_group = diff(res.cumulative_spikes_i_group,1,1) ./ ...
    ((interval_ms/1000).*p.Ni);
res.interval_rate_e_hz = sum(diff(res.cumulative_spikes_e_group,1,1),2) ./ ...
    ((interval_ms/1000)*sum(p.Ne));
res.interval_rate_i_hz = sum(diff(res.cumulative_spikes_i_group,1,1),2) ./ ...
    ((interval_ms/1000)*sum(p.Ni));
res.mean_rate_e_hz_group = 1000*res.spike_count_e./(p.Ne*p.duration_time);
res.mean_rate_i_hz_group = 1000*res.spike_count_i./(p.Ni*p.duration_time);
res.mean_rate_e_hz = 1000*sum(res.spike_count_e)/(sum(p.Ne)*p.duration_time);
res.mean_rate_i_hz = 1000*sum(res.spike_count_i)/(sum(p.Ni)*p.duration_time);
res.n_e_final = reshape(Y(end,L.N_e),size(L.N_e));
res.n_i_final = reshape(Y(end,L.N_i),size(L.N_i));
res.V_e_all_final = reshape(Y(end,L.M_e),size(L.M_e));
res.V_i_all_final = reshape(Y(end,L.M_i),size(L.M_i));
end


function [E_sp,I_sp] = local_generate_spikes(cum_e,cum_i,t,Ne,Ni)
E_sp = local_population_spikes(cum_e,t,Ne);
I_sp = local_population_spikes(cum_i,t,Ni);
end


function spikes = local_population_spikes(cumulative,t,group_sizes)
expected = max(diff(cumulative,1,1),0);
base = floor(expected);
counts = base + (rand(size(expected)) < expected-base);
event_count = sum(counts,'all');
spikes = zeros(2,event_count);
group_start = [0,cumsum(group_sizes(1:end-1))];
cursor = 0;
for it = 1:size(counts,1)
    active_groups = find(counts(it,:)>0);
    for block = active_groups
        count = counts(it,block);
        neuron_count = group_sizes(block);
        if count <= neuron_count
            ids = randperm(neuron_count,count);
        else
            ids = randi(neuron_count,1,count);
        end
        destination = cursor+(1:count);
        spikes(1,destination) = group_start(block)+ids;
        spikes(2,destination) = t(it+1);
        cursor = cursor+count;
    end
end
end
