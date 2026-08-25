function summary = generate_connection_matrices(output_dir,params)
%GENERATE_CONNECTION_MATRICES Generate spatial and block connection files.
%
% summary = generate_connection_matrices(output_dir)
% summary = generate_connection_matrices(params)
% summary = generate_connection_matrices(output_dir,params)
%
% The generated files use public Pre->Post names (EI means E->I).
% Neuron-level and block-probability matrices both use rows=Pre and
% columns=Post. See connection_mat/README.md for numbering rules.

if nargin < 1
    output_dir = pwd;
    params = struct();
elseif nargin == 1 && isstruct(output_dir)
    params = output_dir;
    output_dir = pwd;
elseif isempty(output_dir)
    output_dir = pwd;
end
if nargin < 2 && ~exist('params','var') || isempty(params)
    params = struct();
end
[params,layout] = generate_network_layout(output_dir,params);
positions = layout.positions;
E_group = layout.E_group;
I_group = layout.I_group;
index_map = layout.index_map;
block_ids = layout.block_ids;
output_dir = char(output_dir);
layout_path = fullfile(output_dir,'network_layout.mat');

rng_state = rng;
restore_rng = onCleanup(@() rng(rng_state));
rng(params.rng_seed,'twister');
types = {'EE','EI','IE','II'};
pre_types = {'E','E','I','I'};
post_types = {'E','I','E','I'};
sigma_fields = {'sigmaEE','sigmaEI','sigmaIE','sigmaII'};
p_values = [params.p_ee,params.p_ei,params.p_ie,params.p_ii];
summary = struct('layout_file',layout_path,'files',{{}},'params',params);

for type_id = 1:numel(types)
    type = types{type_id};
    [pre_positions,pre_blocks,pre_order] = local_population_data( ...
        pre_types{type_id},positions,block_ids,index_map,params.ne);
    [post_positions,post_blocks,post_order] = local_population_data( ...
        post_types{type_id},positions,block_ids,index_map,params.ne);
    sigma_values = params.(sigma_fields{type_id});
    for sigma = sigma_values(:).'
        paths = local_output_paths(output_dir,type,sigma);
        local_check_outputs(paths,params.overwrite);
        fprintf('Generating %s, sigma=%g ...\n',type,sigma);
        original = local_spatial_matrix(pre_positions,post_positions, ...
            p_values(type_id),sigma,params);
        probability = local_block_probability(original,pre_blocks, ...
            post_blocks,numel(E_group));
        block_connection = original(pre_order,post_order);

        original_data = struct();
        original_data.([type,'_conn_mat']) = original;
        save(paths.original,'-struct','original_data');
        probability_data = struct();
        probability_data.([type,'_prob_mat']) = probability;
        probability_data.row_groups = local_group_counts( ...
            pre_types{type_id},E_group,I_group);
        probability_data.col_groups = local_group_counts( ...
            post_types{type_id},E_group,I_group);
        save(paths.probability,'-struct','probability_data');
        block_data = struct();
        block_data.([type,'_block_conn_mat']) = block_connection;
        save(paths.block,'-struct','block_data');
        summary.files = [summary.files; ...
            {paths.original;paths.probability;paths.block}];
    end
end
end


function [population_positions,population_blocks,population_order] = ...
        local_population_data(type,positions,block_ids,index_map,ne)
if type == 'E'
    ids = (1:ne).';
    offset = 0;
else
    ids = (ne+1:size(positions,1)).';
    offset = ne;
end
population_positions = positions(ids,:);
population_blocks = block_ids(ids);
inverse_map = zeros(size(index_map));
inverse_map(index_map) = 1:numel(index_map);
population_order = inverse_map(offset+(1:numel(ids))).'-offset;
end


function matrix = local_spatial_matrix(pre_positions,post_positions,p_base,sigma,p)
n_pre = size(pre_positions,1);
n_post = size(post_positions,1);
n_chunks = ceil(n_pre/p.chunk_size);
rows = cell(n_chunks,1);
cols = cell(n_chunks,1);
normalizer = 2*pi*sigma^2*(1-exp(-p.truncation_factor^2/2));
cutoff_squared = (p.truncation_factor*sigma)^2;
for chunk = 1:n_chunks
    first = (chunk-1)*p.chunk_size+1;
    last = min(chunk*p.chunk_size,n_pre);
    pre = pre_positions(first:last,:);
    dx = abs(pre(:,1)-post_positions(:,1).');
    dy = abs(pre(:,2)-post_positions(:,2).');
    dx = min(dx,p.L-dx);
    dy = min(dy,p.L-dy);
    distance_squared = dx.^2+dy.^2;
    [local_rows,local_cols] = find(distance_squared < cutoff_squared);
    selected = sub2ind(size(distance_squared),local_rows,local_cols);
    probability = min(p_base/normalizer* ...
        exp(-distance_squared(selected)/(2*sigma^2)),1);
    connected = rand(size(probability)) < probability;
    rows{chunk} = local_rows(connected)+first-1;
    cols{chunk} = local_cols(connected);
end
rows = vertcat(rows{:});
cols = vertcat(cols{:});
matrix = logical(sparse(rows,cols,true(size(rows)),n_pre,n_post));
end


function probability = local_block_probability(original,pre_blocks,post_blocks,n_blocks)
pre_ids = accumarray(pre_blocks,(1:numel(pre_blocks)).', ...
    [n_blocks,1],@(x){x},{[]});
post_ids = accumarray(post_blocks,(1:numel(post_blocks)).', ...
    [n_blocks,1],@(x){x},{[]});
probability = zeros(n_blocks,n_blocks);
for pre_block = 1:n_blocks
    for post_block = 1:n_blocks
        count = numel(pre_ids{pre_block})*numel(post_ids{post_block});
        if count > 0
            probability(pre_block,post_block) = nnz(original( ...
                pre_ids{pre_block},post_ids{post_block}))/count;
        end
    end
end
end


function groups = local_group_counts(type,E_group,I_group)
if type == 'E'
    groups = E_group(:);
else
    groups = I_group(:);
end
end


function paths = local_output_paths(folder,type,sigma)
paths.original = fullfile(folder,sprintf('%s_sig%g_conn_mat.mat',type,sigma));
paths.probability = fullfile(folder,sprintf('%s_sig%g_prob_mat.mat',type,sigma));
paths.block = fullfile(folder,sprintf('%s_sig%g_block_conn_mat.mat',type,sigma));
end


function local_check_outputs(paths,overwrite)
if overwrite
    return
end
names = fieldnames(paths);
for k = 1:numel(names)
    if isfile(paths.(names{k}))
        error('Output already exists (set params.overwrite=true to replace it): %s', ...
            paths.(names{k}));
    end
end
end
