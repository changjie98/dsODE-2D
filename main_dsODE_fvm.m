function res_fvm = main_dsODE_fvm()
%MAIN_DSODE_FVM Run one coupled excitatory/inhibitory FVM population.
%
% Usage:
%   res_fvm = main_dsODE_fvm();
%
% Edit the parameters below to change the model. EI means E->I and IE
% means I->E throughout this package.

params = struct();
params.J_ex = 5;
params.M = 100;
params.Mr = 66;
params.ne = 300;
params.ni = 100;
params.dt = 0.1;
params.duration_time = 1000;
params.tau_ee = 3;
params.tau_ei = 3;
params.tau_i = 10;
params.tau_r = 2;
params.tau_m = 20;
params.s_ee = 3;
params.s_ei = 4;
params.s_ie = 8;
params.s_ii = 8;
params.p_ee = 0.2;
params.p_ei = 0.2;
params.p_ie = 0.2;
params.p_ii = 0.2;
params.V_bin = 5;
params.V_bin_min = -10;
params.V_reset = 0;
params.rel_tol = 1e-6;
params.abs_tol = 1e-8;
params.max_step = 0.1;
params.stochastic_synaptic_decay = true;
params.include_synaptic_diffusion = true;
params.refractory_stages = 1;
params.ode_solver = 'ode15s';
params.use_jpattern = true;
params.time_integrator = 'fixed_queue_ssprk3';
params.fixed_cfl = 0.2;
params.fixed_max_step = 0.1;
params.rng_seed = 1;

probability = [params.p_ee,params.p_ei;params.p_ie,params.p_ii];
res_fvm = dsODE_fvm_single_core(probability,params,[],[]);
end
