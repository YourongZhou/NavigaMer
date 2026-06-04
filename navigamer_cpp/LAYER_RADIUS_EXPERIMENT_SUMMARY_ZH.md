# Layer / Radius 实验总结

## 实验目标

这个实验的目标，是在**固定 finest layer 半径**的前提下，研究 NavigaMer 的**主层数**和**半径衰减 schedule**对**搜索成本**的影响。

最核心的控制变量是：

- `r_leaf` 在同一组比较中保持不变

这样可以保证最终搜索分辨率一致，使得成本变化主要归因于：

- 主层数 `L`
- 半径衰减系数 `alpha`

而不是因为 finest layer 本身变粗或变细。

## 实验设计

### 搜索参数组合

实验扫描下面这些组合：

- `L ∈ {2, 3, 4, 5}`
- `r_leaf ∈ {4, 8, 12}`
- `alpha ∈ {0.5, 0.7}`

对每一个组合，主层半径按几何方式生成：

`radius[layer_idx] = round(r_leaf / alpha^(L - 1 - layer_idx))`

并强制最细层满足：

`radius[L - 1] = r_leaf`

例如：

- `r_leaf = 8, alpha = 0.5, L = 4` 时，得到 `64|32|16|8`
- `r_leaf = 8, alpha = 0.7, L = 4` 时，得到 `23|16|11|8`

辅助层仍然按照相邻主层自动生成，并在查询前被折叠进 beacon / MBB 结构中。

### 固定输入

在一次实验运行里，所有 `(L, r_leaf, alpha)` 组合共用同一组：

- reference sequence
- query 集合
- query length
- tolerance
- random seed

这样可以保证不同参数组合之间的比较是公平的。

### 记录指标

当前实现只记录**搜索成本类指标**，按 query 输出：

- `query_time_ms`
- `world_access_count`
- `node_access_count`
- `edge_access_count`
- `anchor_distance_count`
- `bound_check_count`
- `candidate_count`
- `candidate_verify_count`

这些指标的定义如下：

- `world_access_count`：当前 query 过程中访问到的 world / DAG 节点数
- `node_access_count`：当前 query 过程中检查到的 finest-layer 序列节点数
- `edge_access_count`：当前 query 过程中检查到的 parent-child world 边数
- `anchor_distance_count`：计算 `d(query, beacon)` 的次数
- `bound_check_count`：基于 MBB 或 leaf-beacon 的剪枝判断次数
- `candidate_count`：进入 exact verification 的候选数
- `candidate_verify_count`：实际执行 exact verification 的次数

在当前实现里，`candidate_count` 和 `candidate_verify_count` 基本处于同一个阶段，因为所有 surviving leaf candidate 都会被精确验证。

## 一个关键修正：使用更密的 stride

### 为什么 sparse window 不合适

在原来的 sparse window 设置下，finest layer 很容易退化成：

- 一个 window 对应一个 finest node

原因是：

- query/reference window 之间的相似性太低
- `r_leaf = 4, 8, 12` 这样的半径不足以把大量不重叠窗口合并到同一个 finest node

这样一来，实验测到的主要是：

- traversal overhead

而不是：

- finest layer 上真正有意义的聚类和剪枝效果

### 修正方式

因此实验命令后来扩展支持了：

- 显式 `--stride`

对于 layer/radius 实验，建议优先使用 `--stride`，而不是只依赖 `--stride-mode`。

本组正式结果全部采用：

- `--length 250`
- `--stride 1`

这样会生成高度重叠的窗口，更有利于多个 window 合并到同一个 finest-layer node，从而使层数和半径 schedule 的影响真正体现出来。

## 实际运行设置

原始计划是直接对 `data/human/chr1_subset` 全量 `100001 bp` 做完整 sweep。但在当前 builder 实现下，`length=250, stride=1` 会生成大约 `99752` 个窗口，索引构建时间过长，不适合快速得到第一版结论。

因此本次正式汇总采用的是一个**缩小版但结构完全一致**的设置：

- reference：`chr1_subset` 前 `5000 bp`
- `length = 250`
- `stride = 1`
- 窗口数：`4751`
- `queries-per-cell = 20`
- 参数网格：完整 `24` 组 `(L, r_leaf, alpha)`

也就是说，这次结果可以看作：

- 方法学上有效
- 足够支持比较不同层数和不同 `alpha` 的趋势
- 但仍然不是完整 `chr1_subset` 全量结论

## 实验结果

### 结果文件

本次完整 sweep 的聚合结果保存在：

- [layer_sweep_5k_summary.csv](/home/luting/projects/AnchorMapping/NavigaMer/.tmp_experiments/layer_sweep_5k_summary.csv)

该文件按 `(L, r_leaf, alpha)` 聚合了 `20` 条 query 的平均值。

对应的图像文件为：

- [layer_radius_query_time_vs_L.png](/home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/figures/layer_radius_query_time_vs_L.png)
- [layer_radius_access_breakdown_vs_L.png](/home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/figures/layer_radius_access_breakdown_vs_L.png)

### 图 1：查询时间随层数变化

![查询时间随层数变化](/home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/figures/layer_radius_query_time_vs_L.png)

这张图展示了固定 `r_leaf` 时，不同 `L` 和不同 `alpha` 对平均查询时间的影响。可以直接看到：

- `alpha = 0.5` 基本都优于 `alpha = 0.7`
- 最优层数会随着 `r_leaf` 变化，而不是固定不变

### 图 2：访问成本分解

![访问成本分解](/home/luting/projects/AnchorMapping/NavigaMer/navigamer_cpp/figures/layer_radius_access_breakdown_vs_L.png)

这张图固定观察 `alpha = 0.5`，展示 `world_access`、`edge_access` 和 `anchor_distance` 随 `L` 的变化。它说明：

- `L` 增加时，`world_access` 往往下降
- 但 `edge_access` 和 `anchor_distance` 可能迅速上升
- 总时间最优点来自这几项成本之间的平衡

### 每个 `r_leaf` 下的最快配置

| `r_leaf` | 最快配置 | `radius_schedule` | 平均查询时间 `query_time_ms` |
| --- | --- | --- | ---: |
| `4` | `L=5, alpha=0.5` | `64|32|16|8|4` | `11.103` |
| `8` | `L=4, alpha=0.5` | `64|32|16|8` | `10.161` |
| `12` | `L=3, alpha=0.5` | `48|24|12` | `11.702` |

这说明：

- 层数并不是越多越快
- 对不同的 `r_leaf`，最优的主层数 `L` 可能不同
- 在本次实验里，`alpha = 0.5` 的配置整体更有优势

### 代表性结果表

下面给出每个 `r_leaf` 下若干代表性组合的平均值：

| `L` | `r_leaf` | `alpha` | `radius_schedule` | 平均时间(ms) | 平均 `world_access` | 平均 `edge_access` | 平均 `anchor_distance` | 平均 `candidate_count` |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 0.5 | `8|4` | 48.518 | 481.20 | 9.35 | 5.80 | 5.40 |
| 3 | 4 | 0.5 | `16|8|4` | 27.206 | 251.30 | 23.90 | 12.55 | 6.55 |
| 4 | 4 | 0.5 | `32|16|8|4` | 17.152 | 141.60 | 39.25 | 19.60 | 6.40 |
| 5 | 4 | 0.5 | `64|32|16|8|4` | 11.103 | 72.70 | 62.40 | 29.60 | 6.05 |
| 2 | 8 | 0.5 | `16|8` | 26.352 | 249.05 | 11.50 | 5.85 | 6.15 |
| 3 | 8 | 0.5 | `32|16|8` | 15.875 | 139.10 | 26.45 | 12.45 | 5.90 |
| 4 | 8 | 0.5 | `64|32|16|8` | 10.161 | 70.20 | 48.35 | 21.90 | 6.00 |
| 5 | 8 | 0.5 | `128|64|32|16|8` | 35.543 | 17.55 | 452.70 | 328.25 | 5.85 |
| 2 | 12 | 0.5 | `24|12` | 27.900 | 174.90 | 12.15 | 6.20 | 6.00 |
| 3 | 12 | 0.5 | `48|24|12` | 11.702 | 94.00 | 29.45 | 13.50 | 5.95 |
| 4 | 12 | 0.7 | `35|24|17|12` | 16.299 | 134.55 | 41.50 | 20.50 | 6.00 |
| 5 | 12 | 0.7 | `50|35|24|17|12` | 13.439 | 92.80 | 63.65 | 32.20 | 5.95 |

### 最明显的趋势

#### 1. `alpha = 0.5` 普遍优于 `alpha = 0.7`

这一点在三个 `r_leaf` 上都比较明显：

- `r_leaf = 4` 时，`alpha=0.5` 的最好时间为 `11.103 ms`
- `r_leaf = 8` 时，`alpha=0.5` 的最好时间为 `10.161 ms`
- `r_leaf = 12` 时，`alpha=0.5` 的最好时间为 `11.702 ms`

对应的 `alpha = 0.7` 虽然也有表现不错的组合，但整体上更慢。

这说明更“陡”的半径衰减 schedule，在这组实验里更有利于降低搜索成本。

#### 2. 层数增加先带来收益，但不是越多越好

例如：

- `r_leaf = 4` 时，时间随着 `L=2 -> 3 -> 4 -> 5` 基本持续下降
- `r_leaf = 8` 时，最佳点出现在 `L=4`
- `r_leaf = 12` 时，最佳点出现在 `L=3`

也就是说：

- 增加主层数可以减少 `world_access_count`
- 但过多的层数会增加 `edge_access_count`、`anchor_distance_count` 和 `bound_check_count`
- 当这些额外代价超过前面节省掉的 traversal 成本时，总时间就会反弹

#### 3. 更深的层级可能让 `world_access` 很低，但并不一定更快

最典型的例子是：

- `L=5, r_leaf=8, alpha=0.5`

它的统计是：

- `world_access = 17.55`
- `edge_access = 452.70`
- `anchor_distance = 328.25`
- `query_time_ms = 35.543`

说明它虽然访问的 world 节点很少，但由于边遍历和 anchor 计算爆炸，反而比 `L=4` 差很多。

这说明单看 `world_access_count` 不够，必须结合：

- `edge_access_count`
- `anchor_distance_count`
- `bound_check_count`

一起看。

## 结果解释

本次实验支持下面几条较稳定的结论：

1. `stride = 1` 是合理且必要的。否则 finest layer 容易退化成 one-window-one-node，层数和半径 schedule 的比较没有信息量。
2. `alpha = 0.5` 在这组实验里整体优于 `alpha = 0.7`。
3. 主层数 `L` 存在最优点，不是越多越快。
4. 层数增加带来的收益主要来自更少的 `world_access`，但代价是更高的 `edge_access` 和 `anchor_distance`。
5. 对不同 `r_leaf`，最优层数不同：
   - `r_leaf = 4` 时倾向 `L = 5`
   - `r_leaf = 8` 时倾向 `L = 4`
   - `r_leaf = 12` 时倾向 `L = 3`

## 当前结论的边界

这次汇总已经不是 smoke test，而是一版完整的参数 sweep。

但它仍然有两个边界条件：

1. reference 只用了 `chr1_subset` 的前 `5000 bp`
2. 当前 builder 在全量 `chr1_subset + stride=1` 下仍然太慢，所以还没有得到 10 万窗口规模的完整表

因此更准确的表述应该是：

- 这次结果已经足够支持**方法学和趋势判断**
- 但如果要写成更强的最终结论，仍然建议后续在更大 reference 子集上复现

## 推荐的后续动作

下一步最值得做的有两件事：

1. 对 builder 做并行化或加速，再把同样的实验迁移到更大的 `chr1_subset` 子集
2. 把当前结果进一步画成折线图或热图，例如：
   - `r_leaf` 固定时，`L` 对 `query_time_ms` 的影响
   - `L` 固定时，`alpha` 对 `world_access` / `edge_access` / `anchor_distance` 的影响

## 总结

这次 `5 kb + stride=1` 的完整 sweep 已经给出了第一版可用结论：

- **主层数不是越多越好，而是存在与 `r_leaf` 相关的最优值**
- **`alpha = 0.5` 在这组实验中整体更优**
- **真正决定查询时间的，不只是 world 节点访问数，还包括 edge 遍历和 anchor 距离计算开销**

如果只看这次结果，最简洁的经验结论是：

- `r_leaf = 4` 时优先试 `L = 5`
- `r_leaf = 8` 时优先试 `L = 4`
- `r_leaf = 12` 时优先试 `L = 3`

这是当前最值得带去更大规模实验验证的工作假设。
