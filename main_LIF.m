function res_lif = main_LIF()
%MAIN_LIF Run one coupled excitatory/inhibitory LIF population.
%
% Usage:
%   res_lif = main_LIF();
%
% Edit the parameters below to change the model. EI means E->I and IE
% means I->E throughout this package.

params = struct();
params.Ex_Poisson_lambda = 5;
params.M = 100;
params.Mr = 66;
params.ne = 300;
params.ni = 100;
params.dt = 0.1;
params.duration_time = 1000;
params.tau_m = 20;
params.tau_ee = 3;
params.tau_ei = 3;
params.tau_i = 10;
params.tau_r = 2;
params.s_ee = 3;
params.s_ei = 4;
params.s_ie = 8;
params.s_ii = 8;
params.p_ee = 0.2;
params.p_ei = 0.2;
params.p_ie = 0.2;
params.p_ii = 0.2;
params.record_interval = 5;
params.refractory_mode = 'fixed';
params.rng_seed = 1;

rng(params.rng_seed,'twister');
conn_ee = sprand(params.ne,params.ne,params.p_ee) > 0;
conn_ei = sprand(params.ne,params.ni,params.p_ei) > 0;
conn_ie = sprand(params.ni,params.ne,params.p_ie) > 0;
conn_ii = sprand(params.ni,params.ni,params.p_ii) > 0;
conn_ee(1:params.ne+1:end) = 0;
conn_ii(1:params.ni+1:end) = 0;
connection_mat = [conn_ee,conn_ei;conn_ie,conn_ii];

res_lif = run_LIF_model_core(params,connection_mat);
res_lif = rmfield(res_lif,'connection_mat');
res_lif.E_sp = res_lif.E_sp.';
res_lif.I_sp = res_lif.I_sp.';
res_lif.params = params;
res_lif.meta = struct('model','single_population_LIF', ...
    'interaction_name_convention','pre_to_post','params',params);
end
