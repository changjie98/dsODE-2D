# Connection files

本目录包含：

- 项目根目录的 `generate_network_layout.m`：生成网络布局，保存并返回可复用的 `params`；
- 项目根目录的 `generate_connection_matrices.m`：使用同一份 `params` 生成下列三类连接文件；
- `*_conn_mat.mat`：按原始空间编号保存的神经元连接矩阵，也可由 `main_LIF_grid` 直接使用；
- `*_prob_mat.mat`：dsODE-FVM 使用的 121×121 区块连接概率；
- `*_block_conn_mat.mat`：按区块连续重映射的神经元连接矩阵，是 `main_LIF_grid` 的默认选择；
- `network_layout.mat`：各区块 E/I 神经元数量、编号映射和空间位置。

## 三类矩阵的生成关系

当前 `generate_connection_matrices.m` 对每一种连接类型和 sigma 先生成连续空间神经元矩阵 `original`，再执行：

```matlab
probability = local_block_probability(original,...);
block_connection = local_block_matrix(probability,...);
```

因此新生成的三个文件满足：

- `conn_mat` 是随机抽样得到的原始编号连接矩阵；
- `prob_mat(a,b)` 等于 `conn_mat` 中“突触前区块 a、突触后区块 b”子矩阵的非零元素数除以该子矩阵元素总数；
- `block_conn_mat` 按区块连续编号；对每个区块对 `(a,b)`，其中所有神经元对都使用同一个 `prob_mat(a,b)`，并独立重新进行 Bernoulli 抽样。

因此 `conn_mat` 和 `block_conn_mat` 的逐元素连接以及非零元素总数通常不同，但后者在期望上保持前者的区块概率和整体平均概率。这样会消除区块内部的连续距离梯度，使 LIF 的区块内均匀连接假设与多区块 dsODE-FVM 一致。生成器不会自动更新目录中已有的 MAT 文件；修改代码后必须重新生成，才能得到这一逻辑的文件。

## 命名与矩阵方向

公开函数和新生成器统一采用突触前到突触后的命名：

- `EE`：E → E
- `EI`：E → I
- `IE`：I → E
- `II`：I → I

三类矩阵全部采用 `Pre × Post`：行号是突触前神经元或区块，列号是突触后神经元或区块。因此：

- `EE`：`ne × ne`
- `EI`（E → I）：`ne × ni`
- `IE`（I → E）：`ni × ne`
- `II`：`ni × ni`

`main_LIF_grid.m` 可通过 `params.connection_matrix_type` 选择 `'block_conn'` 或 `'conn_mat'`，并按这一方向直接拼接 `[EE, EI; IE, II]`，不会交换 EI/IE，也不会转置。

## 编号规则

- 原始神经元编号：在规则网格的同一列内从下到上递增，完成一列后移到右侧一列。E 和 I 各自遵循这一规则。
- 区块编号：最底行从左到右递增，随后逐行向上，每行仍从左到右。
- 区块重映射编号：不同区块依照上述区块顺序连续排列；每个区块内部仍按同一列从下到上、完成一列后向右排列。
- 全局编号先放全部 E 神经元，再放全部 I 神经元。
- `index_map(original_id) = block_order_id`；逆映射可用：

```matlab
inverse_map = zeros(size(index_map));
inverse_map(index_map) = 1:numel(index_map);
```

三类矩阵具体使用的行列编号如下：

| 文件 | 行编号 | 列编号 | 数值含义 |
|---|---|---|---|
| `*_conn_mat.mat` | 突触前群体的原始编号 | 突触后群体的原始编号 | 单条神经元连接，逻辑稀疏矩阵 |
| `*_prob_mat.mat` | 突触前区块编号 1:121 | 突触后区块编号 1:121 | 对应区块对中 `conn_mat` 的经验连接概率 |
| `*_block_conn_mat.mat` | 突触前群体的区块重映射编号 | 突触后群体的区块重映射编号 | 按 `prob_mat` 的区块常概率独立重新抽样 |

各文件中的 E 和 I 行列都是群体内局部编号。例如 `EI_conn_mat` 的行是 E 原始编号 `1:ne`，列是 I 原始编号 `1:ni`；把四块拼成全网矩阵 `[EE,EI;IE,II]` 后，才形成 E 在前、I 在后的全局编号。

## 生成示例

在项目根目录运行：

```matlab
addpath(fullfile(pwd,'connection_mat'))

p = struct();
p.sigmaEE = 0.10;
p.sigmaEI = 0.25;
p.sigmaIE = 0.10;
p.sigmaII = 0.50;
p.rng_seed = 1;

params = generate_network_layout('my_connection_mat',p);
summary = generate_connection_matrices('my_connection_mat',params);
```

省略输出目录时，默认写入 MATLAB 当前工作目录 `pwd`：

```matlab
params = generate_network_layout(p);
summary = generate_connection_matrices(params);
```

生成的 `network_layout.mat` 包含 `positions`、`block_positions`、`E_group`、`I_group`、`index_map`、`block_ids`、`numbering` 和补全后的 `params`。`generate_connection_matrices` 内部也会调用 `generate_network_layout`，因此两者不会维护不同的默认值。

主要可选参数及默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `ne`, `ni` | 30000, 10000 | E/I 神经元数 |
| `L` | `sqrt(10)` | 周期性正方形区域边长 |
| `block_step` | 0.3 | 正方形区块边长；边缘不足一个完整步长时保留为较窄区块 |
| `p_ee`, `p_ei`, `p_ie`, `p_ii` | 0.002 | 对应整张 Pre×Post 矩阵的目标平均连接概率 |
| `sigmaEE`, `sigmaEI`, `sigmaIE`, `sigmaII` | 0.10, 0.25, 0.10, 0.50 | 四类高斯空间尺度；均可使用向量 |
| `truncation_factor` | 3 | 空间核截断半径，单位为 sigma |
| `chunk_size` | 500 | 每个距离计算分块中的突触前神经元数 |
| `rng_seed` | 1 | 随机种子 |
| `overwrite` | `true` | 是否允许覆盖已有输出 |

### `p_*` 如何控制平均连接概率

代码使用截断半径 `R=3*sigma`，并令

```text
p(r) = min(amplitude*exp(-r^2/(2*sigma^2)), 1),  r < R
```

生成器在周期正方形区域上数值积分该截断核，并用二分法反求 `amplitude`，使

```text
mean_over_periodic_domain(p(r)) = p_*.
```

所以当前默认 `p_*=0.002` 对应的生成矩阵平均连接概率约为 0.002；若设置为 0.02，平均概率就约为 0.02。有限神经元网格和 Bernoulli 抽样会造成小幅偏差。概率上限 1 已包含在反求过程中；如果截断半径覆盖的最大神经元对比例仍小于目标 `p_*`，生成器会报告目标不可达。

本目录现有 MAT 文件的平均连接概率约为 0.002，但不会因生成器代码修改而自动更新。要使用当前的区块常概率重采样逻辑，必须重新生成整组三类文件。

完整规模生成需要较长时间和较多磁盘空间。可以先用较小的 `ne/ni`、较少的区块和单个 sigma 验证自定义参数。
