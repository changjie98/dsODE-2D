tic;
params = struct;
params.Ex_Poisson_lambda = 5;
params.M = 100;
params.Mr = 66;
k = 1/100;
params.ne = 300/k;
params.ni = 100/k;
params.dt = 0.1;
params.duration_time = 300;
params.tau_adapt = 0;
params.tau_m = 20;
params.tau_ee = 3;
params.tau_ie = 3;
params.tau_i = 10;
params.tau_r = 2;
params.p_ee = 0.2; 
params.p_ie = 0.2;
params.p_ei  = 0.2; 
params.p_ii = 0.2;
params.sigma_ee = 0.4; 
params.sigma_ie = 0.25;
params.sigma_ei  = 0.1; 
params.sigma_ii = 0.25; 
params.s_ee     = 3;
params.s_ie     = 5;
params.s_ei     = 9;
params.s_ii     = 10;


% res_lif = run_LIF_model(params);
% res_lif2 = run_LIF_model(params,E_group,I_group,block_prob_mat);
res_lif = run_LIF_model_fast(params,connection_mat);

toc
