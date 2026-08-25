function res_lif_grid = main_LIF_grid(sigmaEE,sigmaEI,sigmaIE,sigmaII)
%MAIN_LIF_GRID Run the 40000-neuron LIF model on all 121 cortical blocks.
%
% Usage:
%   res_lif_grid = main_LIF_grid(sigmaEE,sigmaEI,sigmaIE,sigmaII)
% EI means E->I and IE means I->E throughout this package.

started = tic;
if nargin ~= 4
    error('Four sigma values are required: EE, EI, IE and II.');
end
sigma = [sigmaEE,sigmaEI,sigmaIE,sigmaII];
if any(~isfinite(sigma)) || any(sigma <= 0)
    error('All sigma values must be finite positive scalars.');
end

project_root = fileparts(mfilename('fullpath'));
connection_dir = fullfile(project_root,'connection_mat');
params = struct();
params.connection_matrix_type = 'block_conn'; % 'block_conn' or 'conn_mat'
params.connection_matrix_type = lower(char(params.connection_matrix_type));
layout = load(fullfile(connection_dir,'network_layout.mat'), ...
    'E_group','I_group','index_map');
E_group = double(layout.E_group(:)');
I_group = double(layout.I_group(:)');
index_map = double(layout.index_map(:));
ne = sum(E_group);
ni = sum(I_group);

[connection_mat,connection_files] = local_load_connections( ...
    connection_dir,sigma,ne,ni,params.connection_matrix_type);

params.Ex_Poisson_lambda = 5;
params.M = 100;
params.Mr = 66;
params.ne = ne;
params.ni = ni;
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
params.record_interval = 5;
params.refractory_mode = 'fixed';
params.rng_seed = 1;
params.sigmaEE = sigmaEE;
params.sigmaEI = sigmaEI;
params.sigmaIE = sigmaIE;
params.sigmaII = sigmaII;

rng(params.rng_seed,'twister');
res_lif_grid = run_LIF_model_core(params,connection_mat);
res_lif_grid = rmfield(res_lif_grid,'connection_mat');

inverse_map = zeros(size(index_map));
inverse_map(index_map) = 1:numel(index_map);
e_remap_idx = (1:ne)';
i_remap_idx = (ne+(1:ni))';
if strcmp(params.connection_matrix_type,'block_conn')
    lif_e_ids = inverse_map(e_remap_idx);
    lif_i_ids = inverse_map(i_remap_idx);
    spike_id_order = 'block_order';
else
    lif_e_ids = e_remap_idx;
    lif_i_ids = i_remap_idx;
    spike_id_order = 'original_order';
end
if isempty(res_lif_grid.E_sp)
    res_lif_grid.E_sp_global = zeros(2,0);
else
    ids = lif_e_ids(res_lif_grid.E_sp(:,1));
    res_lif_grid.E_sp_global = [ids(:)';res_lif_grid.E_sp(:,2)'];
end
if isempty(res_lif_grid.I_sp)
    res_lif_grid.I_sp_global = zeros(2,0);
else
    ids = lif_i_ids(res_lif_grid.I_sp(:,1));
    res_lif_grid.I_sp_global = [ids(:)';res_lif_grid.I_sp(:,2)'];
end
res_lif_grid.E_sp = res_lif_grid.E_sp.';
res_lif_grid.I_sp = res_lif_grid.I_sp.';
res_lif_grid.params = params;
res_lif_grid.runtime_s = toc(started);
res_lif_grid.meta = struct('model','grid_LIF','selected_blocks',1:121, ...
    'E_group_selected',E_group,'I_group_selected',I_group, ...
    'e_remap_idx',e_remap_idx,'i_remap_idx',i_remap_idx, ...
    'lif_e_ids',lif_e_ids,'lif_i_ids',lif_i_ids,'sigma',sigma, ...
    'connection_files',connection_files, ...
    'connection_matrix_type',params.connection_matrix_type, ...
    'spike_id_order',spike_id_order, ...
    'interaction_name_convention','pre_to_post','params',params);
fprintf('main_LIF_grid finished in %.3f s.\n',res_lif_grid.runtime_s);
end


function [conn,files] = local_load_connections(folder,sigma,ne,ni,matrix_type)
% All stored matrices use rows=Pre and columns=Post.
matrix_type = lower(char(matrix_type));
if ~ismember(matrix_type,{'block_conn','conn_mat'})
    error('params.connection_matrix_type must be block_conn or conn_mat.');
end
if strcmp(matrix_type,'block_conn')
    suffix = 'block_conn_mat';
else
    suffix = 'conn_mat';
end
files = struct( ...
    'EE',fullfile(folder,sprintf('EE_sig%g_%s.mat',sigma(1),suffix)), ...
    'EI',fullfile(folder,sprintf('EI_sig%g_%s.mat',sigma(2),suffix)), ...
    'IE',fullfile(folder,sprintf('IE_sig%g_%s.mat',sigma(3),suffix)), ...
    'II',fullfile(folder,sprintf('II_sig%g_%s.mat',sigma(4),suffix)));
[parts.EE,ok_ee] = local_load_part(files.EE,['EE_',suffix],[ne,ne]);
[parts.EI,ok_ei] = local_load_part(files.EI,['EI_',suffix],[ne,ni]);
[parts.IE,ok_ie] = local_load_part(files.IE,['IE_',suffix],[ni,ne]);
[parts.II,ok_ii] = local_load_part(files.II,['II_',suffix],[ni,ni]);
if ~(ok_ee && ok_ei && ok_ie && ok_ii)
    error('Connection files are missing or have inconsistent Pre x Post dimensions.');
end
conn = [parts.EE,parts.EI;parts.IE,parts.II];
end


function [part,ok] = local_load_part(path,variable,expected_size)
part = [];
ok = false;
if ~isfile(path)
    return
end
info = whos('-file',path);
if ~any(strcmp({info.name},variable))
    return
end
data = load(path,variable);
part = data.(variable);
ok = isequal(size(part),expected_size);
end
