function output = dsODE_moudle_2D(state, params)

% --- 参数与尺寸 ---
dt = params.dt;
tau_r = params.tau_r;
V_bin = params.V_bin;
V_bin_min = params.V_bin_min;
V_bin_num = params.V_bin_num;
digit_num = params.digit_num;
tau_m = params.tau_m;

% tolerances / small constants used in original
tiny = 1e-4;

% Ensure shapes
[nRows, G] = size(state.n_n); 
assert(nRows == V_bin_num, 'state.n_n rows must equal params.V_bin_num');

% previous-step quantities
n_prev = state.n_n;              % V_bin_num x G
V_all_prev = state.V_n_all;      % V_bin_num x G
fr_prev = reshape(state.fr_n, 1, []);   % 1 x G
ref_prev = reshape(state.ref_n, 1, []); % 1 x G
I_mean_prev = state.I_n_mean;    % V_bin_num x G
I_var_prev = state.I_n_var;      % V_bin_num x G

% compute previous mean voltages (avoid divide by zero)
V_mean_prev = V_all_prev ./ max(n_prev, 1);  % V_bin_num x G

% --- Build bin edge vector (length L = V_bin_num+1) ---
V_lim = V_bin * (V_bin_min + (0:V_bin_num)); % 1 x L
L = length(V_lim); % V_bin_num + 1

% --- Compute a_vals and b_vals for each source interval and group (V_bin_num x G) ---
% boundaries per interval (vectors length V_bin_num)
V_lowlim = V_bin * ( (1:V_bin_num) + V_bin_min - 1 ); % 1 x V_bin_num
V_highlim = V_bin * ( (1:V_bin_num) + V_bin_min ); % 1 x V_bin_num

% reshape to broadcast: make (V_bin_num x G)
V_lowlim_mat = repmat(V_lowlim(:), 1, G);   % V_bin_num x G
V_highlim_mat = repmat(V_highlim(:), 1, G); % V_bin_num x G

vmean = V_mean_prev; % V_bin_num x G

% masks
mask_low = vmean < (V_lowlim_mat + tiny);
mask_mid = (vmean >= (V_lowlim_mat + tiny)) & (vmean <= (V_lowlim_mat + V_highlim_mat)/2);
mask_high = vmean >= V_highlim_mat;
mask_else = ~(mask_low | mask_mid | mask_high);

% allocate
a_vals = zeros(V_bin_num, G);
b_vals = zeros(V_bin_num, G);

% apply vectorized rules
a_vals(mask_low) = V_lowlim_mat(mask_low);
b_vals(mask_low) = V_lowlim_mat(mask_low) + tiny;

a_vals(mask_mid) = V_lowlim_mat(mask_mid);
b_vals(mask_mid) = 2 .* vmean(mask_mid) - V_lowlim_mat(mask_mid);

a_vals(mask_high) = V_highlim_mat(mask_high) - tiny;
b_vals(mask_high) = V_highlim_mat(mask_high);

a_vals(mask_else) = 2 .* vmean(mask_else) - V_highlim_mat(mask_else);
b_vals(mask_else) = V_highlim_mat(mask_else);

% apply membrane decay if tau_m nonzero
if tau_m ~= 0
    decay_factor = 1 - dt / tau_m;
    a_vals = a_vals * decay_factor;
    b_vals = b_vals * decay_factor;
end

% --- Prepare mu, sigma (V_bin_num x G) ---
mu = I_mean_prev;                 % same shape
sigma = sqrt(max(0, I_var_prev)); % ensure non-negative

% For numerical stability, where sigma == 0 we'll set a small positive to avoid division by zero
% but keep original behavior by handling inv_sigma carefully
sigma_safe = sigma;
sigma_safe(sigma_safe == 0) = 1e-12;

% --- Expand to 3D for source x group x destination-edge computations ---
% Shapes:
%   src x grp x edge  => (V_bin_num x G x L)
% reshape matrices to 3D (add third singleton dim)
a3 = reshape(a_vals, [V_bin_num, G, 1]);
b3 = reshape(b_vals, [V_bin_num, G, 1]);
mu3 = reshape(mu, [V_bin_num, G, 1]);
sigma3 = reshape(sigma_safe, [V_bin_num, G, 1]);
Vlim3 = reshape(V_lim, [1, 1, L]);  % 1 x 1 x L

% Precompute constants
sqrt2 = sqrt(2);
inv_sqrt2 = 1 / sqrt2;
sqrt2pi = sqrt(2 / pi);

% --- Compute n_probability_accumulation (source x group x edges) ---
delta_b = b3 + mu3 - Vlim3;  % V_bin_num x G x L
delta_a = a3 + mu3 - Vlim3;

inv_sigma3 = 1 ./ sigma3;

delta_b_norm = delta_b .* inv_sigma3 * inv_sqrt2;
delta_a_norm = delta_a .* inv_sigma3 * inv_sqrt2;

exp_term_b = exp(-delta_b_norm .^ 2);
exp_term_a = exp(-delta_a_norm .^ 2);
erf_term_b = erf(delta_b_norm);
erf_term_a = erf(delta_a_norm);

term1 = sqrt2pi .* sigma3 .* (exp_term_b - exp_term_a);
term2 = - delta_a .* erf_term_a;
term3 = delta_b .* erf_term_b;

numerator = term1 + term2 + term3;
denominator = 2 .* (a3 - b3);

% avoid division by zero in denominator (shouldn't normally happen)
denominator(denominator == 0) = 1e-12;

n_prob_accum = numerator ./ denominator; % src x grp x L

% replicate the trick from original: diff([n_prob_accum repmat(0.5,[len,1])],1,2)
% to create per-destination bin probabilities we need an array to diff along 3rd dim.
% Append constant column of 0.5 along edge dimension for each src x grp.
append_const = 0.5;
append3 = append_const * ones(V_bin_num, G, 1);
n_prob_with_end = cat(3, n_prob_accum, append3); % V_bin_num x G x (L+1) but original appended 0.5 shaped len x1 so after diff length becomes L
% Now take diff along 3rd dim to get per-destination probabilities length L
n_prob = diff(n_prob_with_end, 1, 3); % result: V_bin_num x G x L

% clean tiny values and NaNs
n_prob(abs(n_prob) < 10.^(-digit_num)) = 0;
n_prob(isnan(n_prob)) = 0;

% --- Compute V_probability_accumulation similarly (source x grp x edges) ---
delta_a_pos = a3 + mu3 - Vlim3;
delta_a_neg = -a3 - mu3 + Vlim3;
delta_b_pos = b3 + mu3 - Vlim3;
delta_b_neg = -b3 - mu3 + Vlim3;

delta_a_pos_norm = delta_a_pos .* inv_sigma3 * inv_sqrt2;
delta_a_neg_norm = delta_a_neg .* inv_sigma3 * inv_sqrt2;
delta_b_pos_norm = delta_b_pos .* inv_sigma3 * inv_sqrt2;
delta_b_neg_norm = delta_b_neg .* inv_sigma3 * inv_sqrt2;

exp_a_pos = exp(-delta_a_pos_norm .^ 2);
exp_b_pos = exp(-delta_b_pos_norm .^ 2);
erf_a_pos = erf(delta_a_pos_norm);
erf_a_neg = erf(delta_a_neg_norm);
erf_b_pos = erf(delta_b_pos_norm);
erf_b_neg = erf(delta_b_neg_norm);

term1_a = a3 .^ 2 .* erf_a_neg;
term2_a = mu3 .^ 2 .* erf_a_neg;
term3_a = sigma3 .^ 2 .* erf_a_neg;
term4_a = Vlim3 .^ 2 .* erf_a_pos;
term5_a = 2 .* a3 .* mu3 .* erf_a_neg;
group_a = term1_a + term2_a + term3_a + term4_a + term5_a;

term1_b = b3 .^ 2 .* erf_b_neg;
term2_b = mu3 .^ 2 .* erf_b_neg;
term3_b = sigma3 .^ 2 .* erf_b_neg;
term4_b = Vlim3 .^ 2 .* erf_b_pos;
term5_b = 2 .* b3 .* mu3 .* erf_b_neg;
group_b = - (term1_b + term2_b + term3_b + term4_b + term5_b);

group_exp_a = - sqrt2pi .* sigma3 .* (a3 + mu3 + Vlim3) .* exp_a_pos;
group_exp_b = sqrt2pi .* sigma3 .* (b3 + mu3 + Vlim3) .* exp_b_pos;

numeratorV = group_a + group_b + group_exp_a + group_exp_b;
denominatorV = 4 .* (a3 - b3);
denominatorV(denominatorV == 0) = 1e-12;

V_prob_accum = numeratorV ./ denominatorV; % src x grp x L

% perform same diff trick as original (append a scalar based on mu_vals(...))
% create appended slice matching original code: (mu_vals(:,1)+(a_vals(:,1)+b_vals(:,1))/2)/2
% compute that scalar per src x grp
append_scalar = ( mu + (a_vals + b_vals)/2 ) / 2; % V_bin_num x G
append3V = reshape(append_scalar, [V_bin_num, G, 1]);
V_prob_with_end = cat(3, V_prob_accum, append3V);
V_prob = diff(V_prob_with_end, 1, 3); % src x grp x L

V_prob(abs(V_prob) < 10.^(-digit_num)) = 0;

% compute V_n_num = V_prob ./ n_prob (elementwise), keep safe
V_n_num = V_prob ./ (n_prob + (n_prob==0)); % avoids NaN by adding mask
V_n_num(n_prob == 0) = 0;
V_n_num(isnan(V_n_num)) = 0;
V_n_num(isinf(V_n_num)) = 0;

% --- Now compute integer neuron counts moved: n_n_num = round(n_neurons * n_prob) ---
% n_prev is counts per source interval (V_bin_num x G) -> expand to 3D and multiply
n_prev3 = reshape(n_prev, [V_bin_num, G, 1]); % src x grp x 1
n_n_num = round(n_prev3 .* n_prob, digit_num); % src x grp x L

% zero out tiny negatives
n_n_num(n_n_num < 0) = 0;

% --- Destination bins counts (exclude last column which is 'nf' in original) ---
% sum over source intervals (dim 1)
% n_n_num: src x grp x L  -> sum over src gives 1 x grp x L
sum_over_src = squeeze(sum(n_n_num, 1)); % grp x L

% Destination bins (1..L-1) correspond to actual voltage bins
n_next_mat = sum_over_src(:, 1:(L-1)); % grp x (L-1)
n_next = n_next_mat.'; % (L-1) x G  => V_bin_num x G

% nf (neurons fired beyond top) is column L
nf_next = sum_over_src(:, L).'; % 1 x G

% --- Compute V_mean_next per destination bin: weighted average over source intervals ---
% V_n_num: src x grp x L
% We need weighted sum for columns 1..L-1: for each dst j, sum_src V_n_num(src,grp,dst)*n_n_num(src,grp,dst)
Vn_times_n = V_n_num .* n_n_num; % src x grp x L
weighted_sum = squeeze(sum(Vn_times_n(:,:,1:(L-1)), 1)); % grp x (L-1)
weighted_sum = weighted_sum.'; % (L-1) x G

% Avoid dividing by zero
n_next_nonzero = n_next;
n_next_nonzero(n_next_nonzero == 0) = 1; % to prevent division by 0
V_mean_next = weighted_sum ./ n_next_nonzero; % V_bin_num x G
V_mean_next(n_next == 0) = 0;

% --- Compute nf, fr, ref and refractory leave (vectorized over groups) ---
fr_next = nf_next / dt; % 1 x G
ref_next = ref_prev + nf_next; % 1 x G

if tau_r ~= 0
    dref = - ref_next ./ tau_r + nf_next;
    leave = abs(round(dref * dt, digit_num)); % 1 x G
    ref_next = ref_next - leave;
else
    leave = ref_next;
    ref_next = zeros(1, G);
end

% The 0-interval index j0
j0 = 1 - V_bin_min; % scalar index (should be integer)
if j0 < 1 || j0 > V_bin_num
    error('Computed j0 out of bounds. Check V_bin_min and V_bin_num.');
end

% Add leaving neurons back to bin j0
n_next(j0, :) = n_next(j0, :) + leave;
% Update its mean voltage: weighted combination of existing and returning neurons.
% For returning neurons, original code added 0*leave in numerator (so no voltage contribution)
% thus V_mean_next(j0) = sum(V_n_num(:,j0).*n_n_num(:,j0)) ./ n_next(j0)
% replicate that:
% Extract numerator for dst=j0: sum over src V_n_num(src,grp,j0)*n_n_num(src,grp,j0)
numerator_j0 = squeeze(sum(Vn_times_n(:,:,j0), 1)).'; % 1 x G -> transpose to row? careful
% But numerator_j0 is (1 x G) as squeeze(sum(...,1)) gives grp x1 then transpose yields 1xG
% convert shapes:
numerator_j0 = reshape(numerator_j0, [1, G]);
V_mean_next(j0, :) = numerator_j0 ./ max(n_next(j0, :), 1);
V_mean_next(j0, n_next(j0,:) == 0) = 0;

% --- finalize V_all_next ---
V_all_next = V_mean_next .* n_next;
V_all_next(isnan(V_all_next)) = 0;

% ensure non-negative variances or other cleanups not needed here; pass through input I's
I_n_mean_out = I_mean_prev;
I_n_var_out = I_var_prev;

% --- prepare output struct (shapes consistent) ---
output = struct();
output.n_n = n_next;               % V_bin_num x G
output.V_n_mean = V_mean_next;     % V_bin_num x G
output.V_n_all = V_all_next;       % V_bin_num x G
output.ref_n = ref_next;           % 1 x G
output.nf_n = nf_next;             % 1 x G
output.fr_n = fr_next;             % 1 x G
output.I_n_mean = I_n_mean_out;    % V_bin_num x G
output.I_n_var = I_n_var_out;      % V_bin_num x G

end
