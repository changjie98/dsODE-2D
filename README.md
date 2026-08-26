# dsODE-FVM and LIF models

本目录提供四个可以直接运行的 MATLAB 入口，用于比较单区块及 121 区块网络中的 LIF 模型和有限体积密度模型（dsODE-FVM）。

## 模型与命名约定

- `EE`：E → E
- `EI`：E → I
- `IE`：I → E
- `II`：I → I

dsODE-FVM 将膜电压概率密度划分为有限体积区间。每个区间保存神经元数量和电压一阶矩，并同时演化突触输入的均值、方差以及固定不应期队列。默认积分方法为 SSP-RK3。

## 文件结构

```text
dsODE_fvm_release/
├── main_dsODE_fvm.m          # 单区块 FVM
├── main_dsODE_fvm_grid.m     # 121 区块 FVM
├── main_LIF.m                # 单区块 LIF
├── main_LIF_grid.m           # 121 区块 LIF
├── plot_raster.m             # 四类结果通用的 raster 绘图函数
├── generate_network_layout.m # 生成网络布局与共享参数
├── generate_connection_matrices.m # 生成三类连接矩阵
├── connection_mat/           # 连接数据和网络布局
└── private/                  # 内部求解器，不需要直接调用
```

## 环境要求

- MATLAB R2021a 或更新版本；
- LIF 模型使用 `normrnd` 和 `random`，需要 Statistics and Machine Learning Toolbox；
- 建议至少 8 GB 内存。运行 40,000 神经元的 `main_LIF_grid` 时建议 16 GB 或更多内存。

首先在 MATLAB 中进入本目录：

```matlab
cd('path/to/dsODE_fvm_release')
```

## 单区块 FVM

```matlab
res_fvm = main_dsODE_fvm();
```

所有参数都在 `main_dsODE_fvm.m` 开头设置，包括神经元数量、突触强度、连接概率、时间常数、模拟时长、时间步长和电压区间宽度。结果中的参数可通过以下方式查看：

```matlab
res_fvm.params
```

`E_sp/I_sp` 是根据 FVM 的连续阈值通量抽样得到的伪神经元脉冲事件，主要用于 raster 可视化。默认 `rng_seed = 1`，所以重复运行会得到相同的抽样结果；生成事件后会恢复 MATLAB 原来的随机数状态。

## 121 区块 FVM

```matlab
res_fvm_grid = main_dsODE_fvm_grid(0.10,0.25,0.10,0.50);
```

四个输入参数依次为：

```text
sigmaEE, sigmaEI, sigmaIE, sigmaII
```

函数会从本目录的 `connection_mat` 文件夹加载对应的 121×121 区块概率矩阵。

如果 `connection_mat` 文件夹暂未有连接矩阵，需要先运行

```matlab
params = generate_network_layout('connection_mat');
summary = generate_connection_matrices('connection_mat',params);
```

连接矩阵的各种参数在 `generate_network_layout.m ` 函数中设定。详细内容及编号规则参见 `connection_mat` 文件夹下的文档。

## 单区块 LIF

```matlab
res_lif = main_LIF();
```

所有参数都在 `main_LIF.m` 开头设置。默认随机种子为 1，因此重复运行可以得到相同结果。结果参数位于：

```matlab
res_lif.params
```

## 121 区块 LIF

```matlab
res_lif_grid = main_LIF_grid(0.10,0.25,0.10,0.50);
```

函数固定运行全部 121 个区块，并从 `connection_mat` 加载预先生成的 40,000 神经元连接矩阵，不会重新随机生成全网连接。`main_LIF_grid.m` 中的以下参数决定使用哪类矩阵：

```matlab
params.connection_matrix_type = 'block_conn'; % 'block_conn' 或 'conn_mat'
```

默认使用按区块连续重映射的 `*_block_conn_mat.mat`。改为 `'conn_mat'` 后使用原始空间编号的 `*_conn_mat.mat`。两种模式表示同一套连接关系且都采用 `Pre × Post`，区别是内部神经元排列和脉冲编号顺序；结果中的 `meta.spike_id_order` 会记录实际顺序。

## Raster 绘图

同一个函数可以直接绘制四类公开入口的结果：

```matlab
plot_raster(res_fvm)
plot_raster(res_lif)
plot_raster(res_fvm_grid)
plot_raster(res_lif_grid)
```

函数根据结果中的 `meta` 自动判断单区块或多区块。单区块直接使用局部神经元编号；多区块使用 `block_conn` 时自动加载 `connection_mat/network_layout.mat`，将区块顺序编号映射回空间顺序；使用 `conn_mat` 时编号已经是原始空间顺序，不再重复映射。图中 I 神经元位于下方、E 神经元位于上方。

## 自行生成连接矩阵

`generate_network_layout.m` 先生成网络位置、分组和编号映射，并返回完整的 `params` 结构体。该结构体可以直接交给 `generate_connection_matrices.m`，生成完整的三阶段数据：

1. 按神经元原始空间编号保存的 `*_conn_mat.mat`；
2. FVM 使用的 `*_prob_mat.mat`；
3. LIF 使用的、按区块连续重映射的 `*_block_conn_mat.mat`；
4. 记录位置、分组和编号映射的 `network_layout.mat`。

建议先输出到一个新目录：

```matlab
addpath(fullfile(pwd,'connection_mat'))

p = struct();
p.ne = 30000;
p.ni = 10000;
p.L = sqrt(10);
p.block_step = 0.3;
p.p_ee = 0.002;
p.p_ei = 0.002;
p.p_ie = 0.002;
p.p_ii = 0.002;
p.sigmaEE = 0.10;
p.sigmaEI = 0.25;
p.sigmaIE = 0.10;
p.sigmaII = 0.50;
p.rng_seed = 1;

params = generate_network_layout('my_connection_mat',p);
summary = generate_connection_matrices('my_connection_mat',params);
```

`network_layout.mat` 会同时保存补全后的 `params`。如果全部采用默认参数，可以简写为：

```matlab
params = generate_network_layout('my_connection_mat');
summary = generate_connection_matrices('my_connection_mat',params);
```

不提供目录时，两个函数默认在 MATLAB 当前工作目录 `pwd` 中生成：

```matlab
params = generate_network_layout();
summary = generate_connection_matrices(params);
```

每个 sigma 参数也可以是向量。例如：

```matlab
p.sigmaEE = 0.05:0.01:0.50;
```

生成器按空间块分批计算距离，`p.chunk_size` 控制一次处理的突触后神经元数量，默认是 500。值越大通常越快，但占用内存越多。已有文件默认不会被覆盖；确实需要替换时必须显式设置：

```matlab
p.overwrite = true;
```

### 神经元与区块编号规则

以下规则同时用于生成器和发布目录中的网络布局：

- 原始神经元编号：先沿同一列从下到上递增，再从左列移动到右列；E 和 I 分别编号，合并的全局编号中 E 在前、I 在后。
- 区块编号：底行从左到右递增，然后移动到上一行，仍从左到右，直至最上方。
- 重映射编号：先把同一区块的神经元放在连续区间内；区块之间采用上述区块顺序；每个区块内部仍先从下到上、再从左到右递增。
- `network_layout.mat` 中的 `index_map(i)` 表示“原始全局神经元 `i` 在区块重映射顺序中的编号”。

公开命名始终采用 `EI = E→I`、`IE = I→E`。所有连接矩阵统一采用 `Pre × Post`：行表示突触前神经元或区块，列表示突触后神经元或区块。因此 `EI_conn_mat` 和 `EI_block_conn_mat` 为 `ne × ni`，`IE` 为 `ni × ne`。详细格式见 `connection_mat/README.md`。

## 默认参数

四个入口采用一致的主要动力学参数：

| 参数 | 默认值 |
|---|---:|
| 模拟时长 | FVM 与两个网格入口 500 ms；单区块 LIF 1000 ms |
| 时间步长 | 0.1 ms |
| `tau_m` | 20 ms |
| `tau_ee`, `tau_ei` | 3 ms |
| `tau_i` | 10 ms |
| 固定不应期 `tau_r` | 2 ms |
| `S_EE`, `S_EI`, `S_IE`, `S_II` | 3, 5, 9, 10 |
| FVM 电压区间宽度 | 5 mV |

单区块模型默认使用 300 个 E 神经元、100 个 I 神经元以及四类均为 0.2 的连接概率。修改参数时，应同时修改 `main_dsODE_fvm.m` 和 `main_LIF.m`，以保证两个模型可比。

## 可用的 sigma

发布目录中四类连接均提供从 0.05 到 0.50、间隔 0.01 的预计算文件。例如 `0.10`、`0.25` 和 `0.47` 都可以直接使用。输入值必须与已有文件名精确对应。

## 主要输出

- `E_sp`, `I_sp`：E/I 神经元脉冲事件；
- `fr_e`, `fr_i`：E/I 放电率或阈值通量；
- `params`：本次运行使用的完整参数；
- `meta`：模型、连接和编号约定等元数据；
- FVM 网格结果还包括每个区块的区间放电率与电压密度；
- LIF 网格结果还包括映射到原始神经元位置编号的 `E_sp_global/I_sp_global`。

所有 `E_sp/I_sp` 都采用 `2×N` 格式：第一行为神经元编号，第二行为脉冲时间（ms）。LIF 的事件来自逐神经元模拟；FVM 的事件由连续阈值通量抽样生成，仅用于把群体密度结果显示成 raster。

## GitHub 文件大小

`connection_mat` 总大小约 323 MB，但单个文件均小于 5 MB，未超过 GitHub 的 100 MB 单文件限制。为了减小普通 Git 仓库体积，也可以使用 Git LFS 管理 `.mat` 文件。

## 引用与许可

发布前请根据项目用途补充作者、许可证和相关论文引用信息。
