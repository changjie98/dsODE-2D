function [params,layout] = generate_network_layout(output_dir,params)
%GENERATE_NETWORK_LAYOUT Generate network_layout.mat and reusable params.
%
% params = generate_network_layout(output_dir)
% params = generate_network_layout(params)
% params = generate_network_layout(output_dir,params)
% [params,layout] = generate_network_layout(output_dir,params)
%
% The returned params can be passed directly to:
%   generate_connection_matrices(output_dir,params)

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
params = local_defaults(params);
output_dir = char(output_dir);
if ~isfolder(output_dir)
    mkdir(output_dir);
end

E_positions = local_grid_positions(params.ne,params.L);
I_positions = local_grid_positions(params.ni,params.L);
positions = [E_positions;I_positions];
edges = 0:params.block_step:params.L;
if edges(end) < params.L
    edges = [edges,params.L];
end
E_blocks = local_block_ids(E_positions,edges);
I_blocks = local_block_ids(I_positions,edges);
block_ids = [E_blocks;I_blocks];
n_blocks = (numel(edges)-1)^2;
E_group = accumarray(E_blocks,1,[n_blocks,1]).';
I_group = accumarray(I_blocks,1,[n_blocks,1]).';
E_order = local_block_order(E_positions,E_blocks,n_blocks);
I_order = local_block_order(I_positions,I_blocks,n_blocks)+params.ne;
block_to_original = [E_order;I_order];
block_positions = positions(block_to_original,:);
index_map = zeros(params.ne+params.ni,1);
index_map(block_to_original) = 1:(params.ne+params.ni);
numbering = struct( ...
    'original_neurons','bottom_to_top_then_left_to_right', ...
    'blocks','left_to_right_on_each_row_then_bottom_to_top', ...
    'within_block','bottom_to_top_then_left_to_right', ...
    'index_map','original_id_to_block_order_id');

layout = struct('positions',positions,'block_positions',block_positions, ...
    'E_group',E_group,'I_group',I_group,'index_map',index_map, ...
    'block_ids',block_ids,'numbering',numbering);
layout_path = fullfile(output_dir,'network_layout.mat');
if isfile(layout_path) && ~params.overwrite
    old = load(layout_path,'positions','block_positions','E_group', ...
        'I_group','index_map');
    if ~local_layout_matches(old,layout)
        error('Existing network_layout.mat is incompatible with these parameters.');
    end
else
    save(layout_path,'positions','block_positions','E_group','I_group', ...
        'index_map','block_ids','numbering','params');
end
end


function params = local_defaults(params)
defaults.ne = 30000;
defaults.ni = 10000;
defaults.L = sqrt(10);
defaults.block_step = 0.3;
% Target mean connection probabilities over each full Pre-by-Post matrix.
defaults.p_ee = 0.002;
defaults.p_ei = 0.002;
defaults.p_ie = 0.002;
defaults.p_ii = 0.002;
defaults.sigmaEE = 0.10;
defaults.sigmaEI = 0.25;
defaults.sigmaIE = 0.10;
defaults.sigmaII = 0.50;
defaults.truncation_factor = 3;
defaults.chunk_size = 500;
defaults.rng_seed = 1;
defaults.overwrite = true;
names = fieldnames(defaults);
for k = 1:numel(names)
    if ~isfield(params,names{k}) || isempty(params.(names{k}))
        params.(names{k}) = defaults.(names{k});
    end
end

positive_scalars = {'ne','ni','L','block_step', ...
    'truncation_factor','chunk_size'};
for k = 1:numel(positive_scalars)
    value = params.(positive_scalars{k});
    if ~isscalar(value) || ~isfinite(value) || value <= 0
        error('params.%s must be a positive scalar.',positive_scalars{k});
    end
end
if params.ne ~= round(params.ne) || params.ni ~= round(params.ni) || ...
        params.chunk_size ~= round(params.chunk_size)
    error('Neuron counts and chunk_size must be integers.');
end

probability_names = {'p_ee','p_ei','p_ie','p_ii'};
for k = 1:numel(probability_names)
    value = params.(probability_names{k});
    if ~isscalar(value) || ~isfinite(value) || value < 0 || value > 1
        error('params.%s must be a finite scalar between zero and one.', ...
            probability_names{k});
    end
end
sigma_names = {'sigmaEE','sigmaEI','sigmaIE','sigmaII'};
for k = 1:numel(sigma_names)
    value = params.(sigma_names{k});
    if ~isvector(value) || isempty(value) || ...
            any(~isfinite(value)) || any(value <= 0)
        error('params.%s must contain positive finite values.',sigma_names{k});
    end
end
if ~isscalar(params.rng_seed) || ~isfinite(params.rng_seed) || ...
        params.rng_seed < 0 || params.rng_seed ~= round(params.rng_seed)
    error('params.rng_seed must be a nonnegative integer scalar.');
end
if ~isscalar(params.overwrite)
    error('params.overwrite must be a scalar logical value.');
end
params.overwrite = logical(params.overwrite);
end


function positions = local_grid_positions(count,L)
side = ceil(sqrt(count));
axis_values = linspace(0,L-L/side,side);
[X,Y] = meshgrid(axis_values,axis_values);
positions = [X(:),Y(:)];
positions = positions(1:count,:);
end


function ids = local_block_ids(positions,edges)
n_side = numel(edges)-1;
x_bin = discretize(positions(:,1),edges);
y_bin = discretize(positions(:,2),edges);
ids = x_bin+(y_bin-1)*n_side;
end


function order = local_block_order(positions,block_ids,n_blocks)
parts = cell(n_blocks,1);
for block = 1:n_blocks
    ids = find(block_ids == block);
    [~,within] = sortrows(positions(ids,:),[1 2]);
    parts{block} = ids(within);
end
order = vertcat(parts{:});
end


function tf = local_layout_matches(old,new)
required = {'positions','block_positions','E_group','I_group','index_map'};
tf = all(isfield(old,required));
for k = 1:numel(required)
    tf = tf && isequal(old.(required{k}),new.(required{k}));
end
end
