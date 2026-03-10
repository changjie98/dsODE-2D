load('connect_40000_p0002_sigmaIEII05EEEI025.mat')    
[E_group, I_group, block_prob_mat] = block_spatial_connection_matrix(positions, connection_mat, 0.3);
[block_conn_mat, block_positions] = build_connection_matrix(E_group, I_group, block_prob_mat, positions);
[remap_conn_mat, index_map] = remap_connectivity(block_positions, positions, block_conn_mat, 30000, 10000);
main_dsODE_batch_test2
[E_sp_dsODE, I_sp_dsODE] = generate_spike_times(res_dsODE.fr_e, res_dsODE.fr_i, E_group, I_group);
plot_raster_fast
