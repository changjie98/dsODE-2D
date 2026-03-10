ni = 10000;
ne = 30000;
% res_lif = res_lif_block;
flag = 4;
figure
if flag == 1
    scatter(res_lif.E_sp(2,:),res_lif.E_sp(1,:)+ni,'r.');
    hold on
    scatter(res_lif.I_sp(2,:),res_lif.I_sp(1,:),'b.');
elseif flag == 2
    inv_index_map = zeros(size(index_map));
    inv_index_map(index_map) = 1:numel(index_map);
    E_sp = res_lif.E_sp(1,:);
    E_sp_remap = inv_index_map(E_sp);
    scatter(res_lif.E_sp(2,:),E_sp_remap+ni,'r.');
    hold on
    I_sp = res_lif.I_sp(1,:)+ne;
    I_sp_remap = inv_index_map(I_sp);
    scatter(res_lif.I_sp(2,:),I_sp_remap-ne,'b.');
else
    inv_index_map = zeros(size(index_map));
    inv_index_map(index_map) = 1:numel(index_map);
    E_sp_dsODE = res_dsODE.E_sp;
    I_sp_dsODE = res_dsODE.I_sp;
    E_sp = E_sp_dsODE(1,:);
    E_sp_remap = inv_index_map(E_sp);
    scatter(E_sp_dsODE(2,:),E_sp_remap+ni,'r.');
    hold on
    I_sp = I_sp_dsODE(1,:);
    I_sp_remap = inv_index_map(I_sp);
    scatter(I_sp_dsODE(2,:),I_sp_remap-ne,'b.');
end
