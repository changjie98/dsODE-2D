function summary = generate_connection_matrices(output_dir,params)
%GENERATE_CONNECTION_MATRICES Generate spatial and block connection files.
%
% summary = generate_connection_matrices(output_dir)
% summary = generate_connection_matrices(params)
% summary = generate_connection_matrices(output_dir,params)
%
% The generated files use public Pre->Post names (EI means E->I).
% All matrices use rows=Pre and columns=Post. The neuron-level conn_mat uses
% continuous pairwise distances. prob_mat aggregates conn_mat by block, and
% block_conn_mat is independently sampled with one constant probability per
% block pair. params.p_ee/p_ei/p_ie/p_ii are target mean probabilities over
% the corresponding full Pre-by-Post matrices. See connection_mat/README.md.

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
        [pre_positions,pre_blocks] = local_population_data( ...
            pre_types{type_id},positions,block_ids,params.ne);
        [post_positions,post_blocks] = local_population_data( ...
            post_types{type_id},positions,block_ids,params.ne);
        pre_groups = local_group_counts( ...
            pre_types{type_id},E_group,I_group);
        post_groups = local_group_counts( ...
            post_types{type_id},E_group,I_group);
    sigma_values = params.(sigma_fields{type_id});
    for sigma = sigma_values(:).'
        paths = local_output_paths(output_dir,type,sigma);
        local_check_outputs(paths,params.overwrite);
        fprintf('Generating %s, sigma=%g ...\n',type,sigma);
        original = local_spatial_matrix(pre_positions,post_positions, ...
            p_values(type_id),sigma,params);
        empirical_mean = nnz(original)/(size(original,1)*size(original,2));
        fprintf('  target mean=%g, sampled mean=%g\n', ...
            p_values(type_id),empirical_mean);
        probability = local_block_probability(original,pre_blocks, ...
            post_blocks,numel(E_group));
        block_connection = local_block_matrix( ...
            probability,pre_groups,post_groups);
        block_empirical_mean = nnz(block_connection)/numel(block_connection);
        fprintf('  block-resampled mean=%g\n',block_empirical_mean);

        original_data = struct();
        original_data.([type,'_conn_mat']) = original;
        save(paths.original,'-struct','original_data');
        probability_data = struct();
        probability_data.([type,'_prob_mat']) = probability;
        probability_data.row_groups = pre_groups;
        probability_data.col_groups = post_groups;
        save(paths.probability,'-struct','probability_data');
        block_data = struct();
        block_data.([type,'_block_conn_mat']) = block_connection;
        save(paths.block,'-struct','block_data');
        summary.files = [summary.files; ...
            {paths.original;paths.probability;paths.block}];
    end
end
end


function [population_positions,population_blocks] = ...
        local_population_data(type,positions,block_ids,ne)
if type == 'E'
    ids = (1:ne).';
else
    ids = (ne+1:size(positions,1)).';
end
population_positions = positions(ids,:);
population_blocks = block_ids(ids);
end


function matrix = local_spatial_matrix(pre_positions,post_positions,p_mean,sigma,p)
n_pre = size(pre_positions,1);
n_post = size(post_positions,1);
n_chunks = ceil(n_pre/p.chunk_size);
rows = cell(n_chunks,1);
cols = cell(n_chunks,1);
amplitude = local_mean_probability_amplitude(p_mean,sigma,p);
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
    probability = min(amplitude* ...
        exp(-distance_squared(selected)/(2*sigma^2)),1);
    connected = rand(size(probability)) < probability;
    rows{chunk} = local_rows(connected)+first-1;
    cols{chunk} = local_cols(connected);
end
rows = vertcat(rows{:});
cols = vertcat(cols{:});
matrix = logical(sparse(rows,cols,true(size(rows)),n_pre,n_post));
end


function amplitude = local_mean_probability_amplitude(p_mean,sigma,p)
if p_mean == 0
    amplitude = 0;
    return
end

cutoff = min(p.truncation_factor*sigma,p.L/sqrt(2));
area = p.L^2;
support_fraction = local_torus_integral(Inf,sigma,cutoff,p.L)/area;
if p_mean > support_fraction+1e-12
    error(['Target mean probability %g is unreachable for sigma=%g: ', ...
        'the truncated kernel covers at most %g of the periodic domain.'], ...
        p_mean,sigma,support_fraction);
end

low = 0;
high = exp(cutoff^2/(2*sigma^2));
for iteration = 1:60
    amplitude = (low+high)/2;
    current_mean = local_torus_integral(amplitude,sigma,cutoff,p.L)/area;
    if current_mean < p_mean
        low = amplitude;
    else
        high = amplitude;
    end
end
amplitude = (low+high)/2;
end


function value = local_torus_integral(amplitude,sigma,cutoff,L)
half_width = L/2;
max_radius = min(cutoff,L/sqrt(2));
integrand = @(radius) min(amplitude*exp(-radius.^2/(2*sigma^2)),1).* ...
    local_torus_shell(radius,L);
split = min(max_radius,half_width);
value = integral(integrand,0,split,'AbsTol',1e-12,'RelTol',1e-10);
if max_radius > half_width
    value = value+integral(integrand,half_width,max_radius, ...
        'AbsTol',1e-12,'RelTol',1e-10);
end
end


function shell = local_torus_shell(radius,L)
half_width = L/2;
shell = 2*pi*radius;
outside = radius > half_width;
shell(outside) = shell(outside)-8*radius(outside).* ...
    acos(half_width./radius(outside));
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


function matrix = local_block_matrix(probability,pre_groups,post_groups)
pre_groups = pre_groups(:);
post_groups = post_groups(:);
pre_offsets = [0;cumsum(pre_groups)];
post_offsets = [0;cumsum(post_groups)];
[pre_blocks,post_blocks] = find(probability > 0);
rows = cell(numel(pre_blocks),1);
cols = cell(numel(pre_blocks),1);
for pair = 1:numel(pre_blocks)
    pre_block = pre_blocks(pair);
    post_block = post_blocks(pair);
    connected = rand(pre_groups(pre_block),post_groups(post_block)) < ...
        probability(pre_block,post_block);
    [local_rows,local_cols] = find(connected);
    rows{pair} = local_rows+pre_offsets(pre_block);
    cols{pair} = local_cols+post_offsets(post_block);
end
rows = vertcat(rows{:});
cols = vertcat(cols{:});
matrix = logical(sparse(rows,cols,true(size(rows)), ...
    pre_offsets(end),post_offsets(end)));
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
