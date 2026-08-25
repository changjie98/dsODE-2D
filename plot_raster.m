function ax = plot_raster(res)
%PLOT_RASTER Plot spikes from any public LIF or dsODE-FVM result.
%
%   plot_raster(res)
%   ax = plot_raster(res)
%
% E_sp and I_sp may be 2-by-N or N-by-2. For a multiblock result using
% block_conn, IDs are mapped back with connection_mat/network_layout.mat.

if ~isstruct(res) || ~isfield(res,'E_sp') || ~isfield(res,'I_sp')
    error('Input must be a result struct containing E_sp and I_sp.');
end

E_sp = local_spike_matrix(res.E_sp,'E_sp');
I_sp = local_spike_matrix(res.I_sp,'I_sp');
is_grid = local_is_grid(res);
[ne,ni] = local_population_sizes(res,E_sp,I_sp,is_grid);

if is_grid
    local_validate_ids(E_sp,ne,'E_sp');
    local_validate_ids(I_sp,ni,'I_sp');
    if local_uses_original_order(res)
        E_y = E_sp(1,:)+ni;
        I_y = I_sp(1,:);
    else
        root = fileparts(mfilename('fullpath'));
        layout_path = fullfile(root,'connection_mat','network_layout.mat');
        if ~isfile(layout_path)
            error('Network layout file not found: %s',layout_path);
        end
        layout = load(layout_path,'index_map');
        index_map = double(layout.index_map(:));
        if numel(index_map) ~= ne+ni || ...
                ~isequal(sort(index_map),(1:numel(index_map))')
            error('network_layout.mat contains an invalid index_map.');
        end
        inverse_map = zeros(size(index_map));
        inverse_map(index_map) = 1:numel(index_map);
        E_y = inverse_map(E_sp(1,:))+ni;
        I_y = inverse_map(ne+I_sp(1,:))-ne;
    end
else
    local_validate_ids(E_sp,ne,'E_sp');
    local_validate_ids(I_sp,ni,'I_sp');
    E_y = E_sp(1,:)+ni;
    I_y = I_sp(1,:);
end

fig = figure;
ax = axes(fig);
hold(ax,'on');
plot(ax,I_sp(2,:),I_y,'.','Color',[0 0 1], ...
    'MarkerSize',3,'DisplayName','I');
plot(ax,E_sp(2,:),E_y,'.','Color',[1 0 0], ...
    'MarkerSize',3,'DisplayName','E');
hold(ax,'off');
xlabel(ax,'Time (ms)');
ylabel(ax,'Neuron ID');
ylim(ax,[0.5,ne+ni+0.5]);
box(ax,'on');
%legend(ax,'Location','best');
if is_grid
    title(ax,'Multiblock raster');
else
    title(ax,'Single-block raster');
end

if nargout == 0
    clear ax
end
end


function spikes = local_spike_matrix(spikes,name)
spikes = double(spikes);
if isempty(spikes)
    spikes = zeros(2,0);
elseif size(spikes,1) == 2
    % Already 2-by-N.
elseif size(spikes,2) == 2
    spikes = spikes.';
else
    error('%s must be a 2-by-N or N-by-2 matrix.',name);
end
if any(~isfinite(spikes(:)))
    error('%s must contain finite values.',name);
end
end


function tf = local_uses_original_order(res)
tf = isfield(res,'meta') && isstruct(res.meta) && ...
    isfield(res.meta,'connection_matrix_type') && ...
    strcmpi(res.meta.connection_matrix_type,'conn_mat');
end


function tf = local_is_grid(res)
tf = isfield(res,'E_sp_global') || isfield(res,'I_sp_global');
if isfield(res,'meta') && isstruct(res.meta)
    if isfield(res.meta,'selected_blocks') && numel(res.meta.selected_blocks) > 1
        tf = true;
    end
    if isfield(res.meta,'model')
        model = lower(char(res.meta.model));
        tf = tf || contains(model,'grid') || contains(model,'multiblock');
    end
end
end


function [ne,ni] = local_population_sizes(res,E_sp,I_sp,is_grid)
if is_grid
    root = fileparts(mfilename('fullpath'));
    layout = load(fullfile(root,'connection_mat','network_layout.mat'), ...
        'E_group','I_group');
    ne = sum(double(layout.E_group));
    ni = sum(double(layout.I_group));
elseif isfield(res,'params') && isfield(res.params,'ne') && ...
        isfield(res.params,'ni')
    ne = double(res.params.ne);
    ni = double(res.params.ni);
else
    ne = max([E_sp(1,:),0]);
    ni = max([I_sp(1,:),0]);
end
if ~isscalar(ne) || ~isscalar(ni) || ne <= 0 || ni <= 0
    error('Could not determine positive E/I population sizes from the result.');
end
end


function local_validate_ids(spikes,population_size,name)
ids = spikes(1,:);
if any(ids < 1 | ids > population_size | ids ~= round(ids))
    error('%s contains neuron IDs outside 1:%d.',name,population_size);
end
end
