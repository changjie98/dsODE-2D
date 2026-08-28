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
params.connection_matrix_type = 'block_conn'; % 'block_conn', 'conn_mat' or 'prob_mat'
params.connection_matrix_type = lower(char(params.connection_matrix_type));
layout = load(fullfile(connection_dir,'network_layout.mat'), ...
    'E_group','I_group','index_map');
E_group = double(layout.E_group(:)');
I_group = double(layout.I_group(:)');
index_map = double(layout.index_map(:));
ne = sum(E_group);
ni = sum(I_group);

[connection_mat,connection_files,block_prob_mat] = local_load_connections( ...
    connection_dir,sigma,E_group,I_group,params.connection_matrix_type);
if strcmp(params.connection_matrix_type,'prob_mat')
    params.connect = 'prob_mat';
    params.block_prob_mat = block_prob_mat;
    params.E_group = E_group;
    params.I_group = I_group;
else
    params.connect = 'fixed';
end

params.Ex_Poisson_lambda = 5;
params.M = 100;
params.Mr = 66;
params.ne = ne;
params.ni = ni;
params.dt = 0.1;
params.duration_time = 500;
params.tau_m = 20;
params.tau_ee = 3;
params.tau_ei = 3;
params.tau_i = 10;
params.tau_r = 2;
params.s_ee = 3;
params.s_ei = 5;
params.s_ie = 9;
params.s_ii = 10;
params.record_interval = 5;
params.refractory_mode = 'fixed';
params.init_v_min = 0;
params.init_v_max = 0;
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
if ismember(params.connection_matrix_type,{'block_conn','prob_mat'})
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


function [conn,files,block_prob_mat] = local_load_connections( ...
        folder,sigma,E_group,I_group,matrix_type)
% All stored matrices use rows=Pre and columns=Post.
matrix_type = lower(char(matrix_type));
if ~ismember(matrix_type,{'block_conn','conn_mat','prob_mat'})
    error(['params.connection_matrix_type must be block_conn, ' ...
        'conn_mat or prob_mat.']);
end
if strcmp(matrix_type,'block_conn')
    suffix = 'block_conn_mat';
elseif strcmp(matrix_type,'prob_mat')
    suffix = 'prob_mat';
else
    suffix = 'conn_mat';
end
files = struct( ...
    'EE',fullfile(folder,sprintf('EE_sig%g_%s.mat',sigma(1),suffix)), ...
    'EI',fullfile(folder,sprintf('EI_sig%g_%s.mat',sigma(2),suffix)), ...
    'IE',fullfile(folder,sprintf('IE_sig%g_%s.mat',sigma(3),suffix)), ...
    'II',fullfile(folder,sprintf('II_sig%g_%s.mat',sigma(4),suffix)));
block_prob_mat = struct();
if strcmp(matrix_type,'prob_mat')
    [parts.EE,ok_ee] = local_load_probability_part( ...
        files.EE,'EE_prob_mat',E_group,E_group);
    [parts.EI,ok_ei] = local_load_probability_part( ...
        files.EI,'EI_prob_mat',E_group,I_group);
    [parts.IE,ok_ie] = local_load_probability_part( ...
        files.IE,'IE_prob_mat',I_group,E_group);
    [parts.II,ok_ii] = local_load_probability_part( ...
        files.II,'II_prob_mat',I_group,I_group);
else
    ne = sum(E_group);
    ni = sum(I_group);
    [parts.EE,ok_ee] = local_load_part(files.EE,['EE_',suffix],[ne,ne]);
    [parts.EI,ok_ei] = local_load_part(files.EI,['EI_',suffix],[ne,ni]);
    [parts.IE,ok_ie] = local_load_part(files.IE,['IE_',suffix],[ni,ne]);
    [parts.II,ok_ii] = local_load_part(files.II,['II_',suffix],[ni,ni]);
end
if ~(ok_ee && ok_ei && ok_ie && ok_ii)
    error('Connection files are missing or have inconsistent Pre x Post dimensions.');
end
if strcmp(matrix_type,'prob_mat')
    conn = [];
    block_prob_mat = parts;
else
    conn = [parts.EE,parts.EI;parts.IE,parts.II];
end
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


function [part,ok] = local_load_probability_part( ...
        path,variable,expected_rows,expected_cols)
[part,ok] = local_load_part(path,variable, ...
    [numel(expected_rows),numel(expected_cols)]);
if ~ok
    return
end
info = whos('-file',path);
if ~all(ismember({'row_groups','col_groups'},{info.name}))
    ok = false;
    return
end
groups = load(path,'row_groups','col_groups');
ok = isequal(double(groups.row_groups(:).'),double(expected_rows(:).')) && ...
    isequal(double(groups.col_groups(:).'),double(expected_cols(:).')) && ...
    all(isfinite(part(:))) && all(part(:) >= 0 & part(:) <= 1);
end
