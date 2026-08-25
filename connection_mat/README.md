# Connection files

本目录包含：

- 项目根目录的 `generate_network_layout.m`：生成网络布局，保存并返回可复用的 `params`；
- 项目根目录的 `generate_connection_matrices.m`：使用同一份 `params` 生成下列三类连接文件；
- `*_conn_mat.mat`：按原始空间编号保存的神经元连接矩阵，也可由 `main_LIF_grid` 直接使用；
- `*_prob_mat.mat`：dsODE-FVM 使用的 121×121 区块连接概率；
- `*_block_conn_mat.mat`：按区块连续重映射的神经元连接矩阵，是 `main_LIF_grid` 的默认选择；
- `network_layout.mat`：各区块 E/I 神经元数量、编号映射和空间位置。

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
| `p_ee`, `p_ei`, `p_ie`, `p_ii` | 0.002 | 四类空间核的总连接尺度 |
| `sigmaEE`, `sigmaEI`, `sigmaIE`, `sigmaII` | 0.10, 0.25, 0.10, 0.50 | 四类高斯空间尺度；均可使用向量 |
| `truncation_factor` | 3 | 空间核截断半径，单位为 sigma |
| `chunk_size` | 500 | 每个距离计算分块中的突触后神经元数 |
| `rng_seed` | 1 | 随机种子 |
| `overwrite` | `false` | 是否允许覆盖已有输出 |

完整规模生成需要较长时间和较多磁盘空间。可以先用较小的 `ne/ni`、较少的区块和单个 sigma 验证自定义参数。
