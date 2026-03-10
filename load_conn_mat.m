flag = 1;
sigEE = 0.4;
sigEI = 0.1;
sigIE = 0.25;
sigII = 0.25;

if flag == 1
    load(['D:\matlab project\conn_mat\EE_sig', num2str(sigEE), '_conn_mat.mat'])
    load(['D:\matlab project\conn_mat\EI_sig', num2str(sigEI), '_conn_mat.mat'])
    load(['D:\matlab project\conn_mat\IE_sig', num2str(sigIE), '_conn_mat.mat'])
    load(['D:\matlab project\conn_mat\II_sig', num2str(sigII), '_conn_mat.mat'])
    load('D:\matlab project\res_dsODE\positionandmap.mat')
    connection_mat = [EE_conn_mat EI_conn_mat;IE_conn_mat II_conn_mat];
elseif flag == 2
    load(['D:\matlab project\conn_mat\EE_sig', num2str(sigEE), '_block_conn_mat.mat'])
    load(['D:\matlab project\conn_mat\EI_sig', num2str(sigEI), '_block_conn_mat.mat'])
    load(['D:\matlab project\conn_mat\IE_sig', num2str(sigIE), '_block_conn_mat.mat'])
    load(['D:\matlab project\conn_mat\II_sig', num2str(sigII), '_block_conn_mat.mat'])
    load('D:\matlab project\res_dsODE\positionandmap.mat')
    connection_mat = [EE_conn_mat EI_conn_mat;IE_conn_mat II_conn_mat];
else
    load(['D:\matlab project\block_conn_mat\EE_sig', num2str(sigEE), '_prob_mat.mat'])
    load(['D:\matlab project\block_conn_mat\EI_sig', num2str(sigEI), '_prob_mat.mat'])
    load(['D:\matlab project\block_conn_mat\IE_sig', num2str(sigIE), '_prob_mat.mat'])
    load(['D:\matlab project\block_conn_mat\II_sig', num2str(sigII), '_prob_mat.mat'])
    load('D:\matlab project\res_dsODE\positionandmap.mat')
    block_prob_mat = [EE_prob_mat EI_prob_mat;IE_prob_mat II_prob_mat];
end
