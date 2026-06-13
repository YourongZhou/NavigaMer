# NavigaMer 层级参数、数据规模与序列分布实验报告

## 摘要

本报告研究 NavigaMer 广义层级 DAG 中主层数 `L`、几何半径比例
`alpha`、序列总数 `N` 和序列分布之间的关系，目标是回答：

1. 最佳 `L` 和 `alpha` 是否可以仅通过序列总数 `N` 预测；
2. ecoli 与 human 数据是否具有不同的最佳参数；
3. 如果存在差异，差异是否主要由物种类别、GC 含量或局部重复结构导致；
4. 在当前构建算法下，应如何选择层级参数。

实验结果表明：

- 最佳 `L` 和 `alpha` 不随 `N` 线性或单调增长，不能仅通过 `N` 可靠预测；
- 当前有效候选主要集中在 `L=4, alpha=0.5`、`L=5, alpha=0.6`
  和 `L=6, alpha=0.6`；
- human 与 ecoli 的差异主要来自具体区域的重复性和局部距离结构，而不是
  “真核生物”标签或 GC 含量本身；
- 极高重复区域会让折叠后的 DAG 边数和平均出度爆炸，并显著增加构建时间；
- 当前实现的 Phase 2 对相邻 expanded layers 做全量两两距离计算，是主要构建瓶颈；
- 对生产默认配置，`L=4, alpha=0.5` 最稳健；查询延迟优先且索引可长期复用时，
  应额外评估 `L=6, alpha=0.6`。

截至 **2026-06-10 15:14 CST**，低、中、高重复度区域实验已完成 `16/18`。
human 高重复区域的 `L=5, alpha=0.6` 正在 Phase 3，`L=6, alpha=0.6` 尚未开始。

---

## 1. 实验目的

### 1.1 参数规模关系

研究最佳参数是否存在简单规律：

```text
N -> (L, alpha)
```

例如：

- `L` 是否随 `N` 线性增长；
- `L` 是否近似随 `log(N)` 增长；
- `alpha` 是否随数据规模增大而增加；
- 是否可以根据 `N` 直接给出默认参数。

### 1.2 序列分布影响

研究相同 `N` 下，不同序列来源是否需要不同参数，并判断潜在原因：

- GC 含量；
- k-mer 重复率；
- 低复杂度区域；
- 局部序列距离分布；
- 连续滑动窗口产生的强局部相关性。

### 1.3 构建算法影响

分析当前构建方式如何约束参数选择，特别关注：

- top-down greedy cover；
- inter-tier rebinding；
- 辅助层折叠；
- beacon / MBB 预计算；
- 最细层叶子挂接；
- 查询阶段的顶层扫描、边遍历和 beacon 距离计算。

---

## 2. 算法背景与工作假设

## 2.1 半径序列

固定最细层半径 `r_leaf=5` 时，主层半径按几何方式生成：

```text
radius[i] = round(r_leaf / alpha^(L - 1 - i))
```

本报告重点比较三个候选：

| 配置 | 主层半径序列 |
| --- | --- |
| `L=4, alpha=0.5` | `40,20,10,5` |
| `L=5, alpha=0.6` | `39,23,14,8,5` |
| `L=6, alpha=0.6` | `64,39,23,14,8,5` |

`L` 和 `alpha` 并不是独立影响性能。它们共同决定：

- 顶层半径；
- 相邻层半径跨度；
- 每层 world 数量；
- world 之间的重叠程度；
- 折叠后 DAG 的边数和出度。

因此，更合理的选择对象是完整半径 schedule，而不是分别预测 `L` 和 `alpha`。

## 2.2 当前构建流程

当前构建流程为：

```text
deduplicate
-> extended hierarchy sketch
-> inter-tier rebinding
-> collapse auxiliary tiers + MBB
-> leaf attachment
```

### Phase 1：Extended hierarchy sketch

构建采用顺序相关的 top-down greedy cover。输入顺序会影响 world center 和最终结构。

### Phase 2：Inter-tier rebinding

对每对相邻 expanded layers 做全量两两距离计算：

```text
C_rebind ≈ Σ |E_i| * |E_(i+1)|
```

这是当前最主要的构建瓶颈。

### Phase 3：辅助层折叠与 MBB

辅助层被折叠后，一个 primary parent 会连接所有二跳可达的 primary children。
同时为每个 `child × beacon` 组合计算 MBB。

重复结构或过大的半径跨度会导致：

- direct children 数量增加；
- DAG 出度增加；
- MBB 数量增加；
- 查询 edge access 和 anchor distance 增加；
- 构建时间和内存快速上升。

## 2.3 参数选择的工作假设

参数选择应平衡三个成本：

1. **顶层入口扫描成本**：顶层节点太多会增加 `world_access_count`；
2. **DAG 稠密度成本**：顶层半径过大或重叠过强会增加边数和 beacon 数；
3. **层级深度成本**：层数越多，查询和构建经过的层越多。

因此，预期最优参数不会简单随 `N` 增长，而会受到序列距离分布和重复结构影响。

---

## 3. 实验方法

## 3.1 公共设置

除特别说明外，实验使用：

| 设置 | 值 |
| --- | --- |
| 窗口长度 | `250 bp` |
| stride | `1` |
| `r_leaf` | `5` |
| query edits | `2` |
| tolerance | `2` |
| 每个配置查询数 | `100` |

`stride=1` 会生成高度重叠的连续窗口。相邻窗口共享 249 个位置，因此实验研究的是
连续参考片段形成的局部度量结构，而不是独立随机序列集合。

## 3.2 记录指标

查询指标：

- `query_time_ms`
- `world_access_count`
- `node_access_count`
- `edge_access_count`
- `anchor_distance_count`
- `bound_check_count`
- `candidate_count`
- `candidate_verify_count`

构建日志指标：

- 每层节点数；
- finest-layer compression；
- 相邻 primary layers 平均出度；
- 估算总 DAG 边数；
- 构建耗时；
- 构建停止阶段。

本批 layer/radius CSV **不直接记录 recall**。因此，本报告结论主要针对构建成本和查询成本，
不能单独用于证明召回安全性。

## 3.3 实验组

### 实验 A：`N × L × alpha` 扩展实验

数据来源：

- ecoli 基因组随机连续片段；
- human `chr1_subset` 随机连续片段。

规模：

```text
N = 2,000 / 5,000 / 10,000 / 20,000
```

主要网格：

```text
L = 3 / 4 / 5 / 6
alpha = 0.4 / 0.5 / 0.6
```

### 实验 B：单碱基随机化对照

对 ecoli 和 human 片段进行单碱基随机打乱：

- 保持片段长度；
- 保持每种碱基数量；
- 保持 GC 含量；
- 破坏原始局部重复结构和长程结构。

比较随机化前后：

- 层节点数；
- DAG 总边数；
- 最大平均出度；
- 查询访问计数；
- 最佳候选配置。

### 实验 C：低、中、高重复度区域实验

从 ecoli 全基因组和 human `chr1_subset` 中分别选择低、中、高重复度区域。
每个参考片段长度为 `5249 bp`，在 `stride=1` 下产生 `5000` 个窗口。

重复度使用重复 `20-mer` 占比衡量：

| 数据集 | 区域 | 起点 | 重复 20-mer 占比 | GC |
| --- | --- | ---: | ---: | ---: |
| ecoli | low | 0 | 0.0000 | 0.5233 |
| ecoli | mid | 2,535,000 | 0.0000 | 0.4995 |
| ecoli | high | 730,000 | 0.2654 | 0.5691 |
| human | low | 20,000 | 0.0000 | 0.6717 |
| human | mid | 55,000 | 0.0390 | 0.6266 |
| human | high | 75,000 | 0.6346 | 0.6952 |

每个区域串行运行：

```text
L4, alpha=0.5
L5, alpha=0.6
L6, alpha=0.6
```

串行执行用于降低不同配置之间的 CPU 资源竞争，使 wall-clock 时间更可比较。

---

## 4. 实验结果

## 4.1 实验 A：序列总数与最佳参数

### 各数据集当前最佳配置

| 数据集 | N=2k | N=5k | N=10k | N=20k |
| --- | --- | --- | --- | --- |
| ecoli | `L4,a0.5` 7.92 ms | `L4,a0.5` 19.91 ms | `L6,a0.6` 24.76 ms | `L6,a0.6` 63.45 ms |
| human | `L6,a0.6` 9.52 ms | `L6,a0.6` 15.16 ms | `L6,a0.6` 22.91 ms | `L6,a0.6` 38.72 ms |

### 合并 ecoli 与 human 后的当前最佳配置

| N | 最佳配置 | 平均查询时间 | 次优配置 | 次优平均时间 |
| ---: | --- | ---: | --- | ---: |
| 2,000 | `L4,a0.5` | 10.05 ms | `L3,a0.4` | 12.04 ms |
| 5,000 | `L4,a0.5` | 18.30 ms | `L6,a0.6` | 18.98 ms |
| 10,000 | `L6,a0.6` | 23.83 ms | `L4,a0.5` | 27.48 ms |
| 20,000 | `L6,a0.6` | 51.09 ms | `L4,a0.5` | 83.32 ms |

### 结果解释

最佳参数没有表现出简单的线性或单调增长：

- `L` 从 4 跳到 6，但不存在稳定的逐级增长；
- `alpha` 主要集中在 0.5 和 0.6，没有随 `N` 单调变化；
- 相同 `N` 下，ecoli 与 human 可能选择不同参数；
- 重复实验之间的最快配置也存在变化。

因此，当前数据不支持直接拟合：

```text
L = aN + b
alpha = cN + d
```

也不足以证明稳定的：

```text
L ∝ log(N)
```

### 构建失败模式

`N=20,000, L=6, alpha=0.5` 的六个任务均未产出 CSV，并停留在 Phase 2 或 Phase 3。

该配置的半径为：

```text
160,80,40,20,10,5
```

较大的顶层半径和层间重叠会产生稠密 DAG。该结果说明，小 `alpha` 与大 `L`
组合虽然可能降低顶层节点数，但会显著放大构建成本。

---

## 4.2 实验 B：ecoli / human 差异与随机化对照

### N=2000 随机化后的结构

随机化后，ecoli 与 human 的层结构几乎一致：

| 配置 | 数据 | 顶层节点 | 最细层节点 | 总 DAG 边数 | 最大平均出度 |
| --- | --- | ---: | ---: | ---: | ---: |
| `L4,a0.5` | ecoli | 80.0 | 1015.5 | 12,026 | 14.5 |
| `L4,a0.5` | human | 82.5 | 1009.0 | 11,999 | 14.5 |
| `L5,a0.6` | ecoli | 84.0 | 1090.0 | 19,428 | 14.8 |
| `L5,a0.6` | human | 87.0 | 1116.0 | 20,466 | 15.1 |
| `L6,a0.6` | ecoli | 52.5 | 1151.0 | 27,617 | 19.0 |
| `L6,a0.6` | human | 52.5 | 1179.0 | 29,753 | 19.7 |

这说明在局部结构被破坏后，仅保留 GC 和碱基组成不足以产生明显的
ecoli/human DAG 差异。

### 高重复 human 区域随机化前后

`human N=5000 rep1` 约有 15% 的 20-mer 重复出现。随机化后：

| 指标 | 原始 | 随机化 | 变化 |
| --- | ---: | ---: | ---: |
| 顶层节点 | 179 | 208 | +16% |
| 总 DAG 边数 | 48,602 | 32,886 | -32% |
| 最大平均出度 | 38.81 | 15.33 | -61% |
| 查询 edge access | 82.51 | 42.95 | -48% |

该结果支持：

> 局部重复结构会减少粗层节点数量，但同时显著增加 world 重叠、DAG 边数和查询边遍历。

因此，human 与 ecoli 的差异更合理的解释是：

- human 某些区域具有更多重复和低复杂度结构；
- human 不同区域之间的差异可能大于 human 与 ecoli 的平均差异；
- “真核复杂度”不是足够精确的解释变量。

---

## 4.3 实验 C：重复度分层区域实验

### 已完成状态

截至报告时间：

```text
已完成 16 / 18
```

缺失项：

- human high-repeat：`L5,a0.6` 正在 Phase 3；
- human high-repeat：`L6,a0.6` 尚未开始。

### 已完成区域的最佳查询时间

| 数据集 | 区域 | repeat20 | 当前最快配置 | 查询时间 |
| --- | --- | ---: | --- | ---: |
| ecoli | low | 0.0000 | `L5,a0.6` | 16.52 ms |
| ecoli | mid | 0.0000 | `L6,a0.6` | 14.26 ms |
| ecoli | high | 0.2654 | `L6,a0.6` | 10.56 ms |
| human | low | 0.0000 | `L4,a0.5` | 16.04 ms |
| human | mid | 0.0390 | `L6,a0.6` | 11.97 ms |
| human | high | 0.6346 | `L4,a0.5`，深层配置未完成 | 39.85 ms |

低重复区域中，三个配置的 wall-clock 差距通常较小。当前结果不支持通过
repeat20 单独、确定性地预测最快查询配置。

### 已完成普通区域的结构趋势

对于 ecoli low/mid 和 human low：

| 配置 | 顶层节点典型范围 | 总 DAG 边数典型范围 | 最大平均出度 |
| --- | ---: | ---: | ---: |
| `L4,a0.5` | 202–209 | 31k–33k | 14.9–15.2 |
| `L5,a0.6` | 215–221 | 53k–54k | 15.4–15.6 |
| `L6,a0.6` | 131–135 | 75k–76k | 19.9–20.4 |

增加层数的效果是：

- 顶层节点数下降，降低顶层扫描；
- 总 DAG 边数增加；
- query edge access 和 anchor distance 增加；
- 构建时间增加。

### human 高重复区域

human high-repeat 区域的 `L4,a0.5` 结果：

| 指标 | human high | 普通区域典型值 |
| --- | ---: | ---: |
| 原始窗口数 | 5,000 | 5,000 |
| 去重后 unique | 4,610 | 约 5,000 |
| 顶层节点 | 72 | 约 200 |
| 总 DAG 边数 | 816,163 | 约 31k–34k |
| 最大平均出度 | 545.9 | 约 15–18 |
| query edge access | 1,492.37 | 约 41–50 |
| anchor distance | 373.62 | 约 21–24 |
| 查询时间 | 39.85 ms | 约 11–21 ms |
| 构建时间 | 257.2 min | 约 19–26 min |

human high-repeat 区域即使使用相同 `N` 和相同参数，也产生了约普通区域
**24–26 倍**的 DAG 边数。

这说明重复度和局部距离结构对当前构建算法的影响远大于序列总数的细微变化。

### 构建时间

普通区域的构建时间：

| 配置 | 普通区域构建时间范围 |
| --- | ---: |
| `L4,a0.5` | 18.6–25.8 min |
| `L5,a0.6` | 31.4–40.5 min |
| `L6,a0.6` | 37.4–48.5 min |

human high-repeat 的 `L4,a0.5` 构建耗时为 **257.2 min**。
`L5,a0.6` 在报告生成时仍停留在 Phase 2。

---

## 5. 综合分析

## 5.1 `L` 和 `alpha` 是否可以通过 `N` 估计

不能只通过 `N` 准确估计。

原因是相同 `N` 下，不同区域可能产生完全不同的：

- 顶层节点数；
- 相邻层节点乘积；
- world 重叠程度；
- DAG 总边数；
- beacon 数量；
- 查询边访问量。

在当前实现中，结构成本更接近以下函数：

```text
cost = f(
  per-layer node counts,
  adjacent-layer overlap,
  DAG degree,
  local distance distribution,
  repeat structure
)
```

而不是：

```text
cost = f(N)
```

## 5.2 ecoli 和 human 是否具有不同最佳参数

不能简单认为 ecoli 与 human 各自拥有固定最佳参数。

更准确的结论是：

- 低重复 human 区域与低重复 ecoli 区域具有相近 DAG 结构；
- 高重复 human 区域会产生极稠密 DAG；
- ecoli 中也能找到较高重复区域，并表现出结构变化；
- 序列来源标签只是间接变量，重复度和局部距离结构才是直接变量。

## 5.3 为什么更深层级有时更快

`L6,a0.6` 通常显著减少顶层节点扫描：

```text
world_access_count ↓
```

但同时增加：

```text
edge_access_count ↑
anchor_distance_count ↑
bound_check_count ↑
build time ↑
```

当顶层扫描节省超过新增边和 beacon 成本时，深层级更快；反之则更慢。

## 5.4 为什么小 alpha 与大 L 危险

例如 `L6,a0.5` 的半径序列：

```text
160,80,40,20,10,5
```

粗层半径过大可能造成：

- 顶层只有极少节点；
- 下层大量 world 同时落入粗层覆盖；
- rebinding 后出度爆炸；
- Phase 2/3 时间和内存不可接受。

因此，不能把“顶层节点更少”视为单独优化目标。

---

## 6. 结论

### 6.1 可确认结论

1. **最佳 `L` 和 `alpha` 不随 `N` 线性或单调增长。**
2. **不能仅通过序列总数 `N` 可靠选择参数。**
3. **序列重复性和局部距离结构显著影响 DAG 密度。**
4. **human 与 ecoli 的差异主要来自具体区域，而不是物种标签本身。**
5. **GC 含量不是当前观察差异的主要原因。**
6. **Phase 2 全量 inter-tier rebinding 是当前主要构建瓶颈。**
7. **层数增加会降低顶层扫描，但增加 DAG 边数、anchor 计算和构建成本。**
8. **极高重复区域会让当前 DAG 构建出现数量级恶化。**

### 6.2 当前推荐参数

#### 平衡构建成本、内存和查询

```text
L=4, alpha=0.5
schedule=40,20,10,5
```

适合作为生产默认值。构建成本较低，普通区域 DAG 出度稳定。

#### 查询延迟优先，索引长期复用

```text
L=6, alpha=0.6
schedule=64,39,23,14,8,5
```

应在抽样数据上验证后使用。它经常降低查询时间，但构建和内存成本更高。

#### 中间方案

```text
L=5, alpha=0.6
schedule=39,23,14,8,5
```

层间变化平滑，但在当前已完成区域中很少成为明确最快配置。

### 6.3 推荐自动选择策略

对目标数据抽样 `2k–5k` 条序列，分别构建三个候选 schedule。

优先拒绝满足以下任一条件的配置：

- 最大平均出度超过 `64–100`；
- 总 DAG 边数显著超过 `20 × N`；
- 出现连续多个仅包含一个节点的粗层；
- `Σ |E_i||E_(i+1)|` 超出构建预算；
- Phase 2 或 Phase 3 构建时间异常。

对剩余配置，再使用固定查询集比较：

- P50 / P95 查询时间；
- `world_access_count`；
- `edge_access_count`；
- `anchor_distance_count`；
- recall。

---

## 7. 局限性

1. 当前查询时间是 wall-clock 时间，仍可能受到机器负载影响；
2. 部分早期实验采用并发运行，不适合精确比较微小时间差；
3. 当前 CSV 不直接记录 recall；
4. human 数据只来自一个 `chr1_subset`，不能代表完整人类基因组；
5. 当前重复度只使用简单的重复 20-mer 占比描述；
6. Phase 1 greedy cover 对输入顺序敏感；
7. human high-repeat 的两个深层配置尚未完成；
8. 当前没有完整记录峰值内存和每个构建阶段的独立耗时。

---

## 8. 后续实验建议

### 8.1 完成 human high-repeat 深层配置

优先观察：

- `L5,a0.6` 和 `L6,a0.6` 是否能降低查询成本；
- 是否会进一步增加 DAG 边数；
- 构建是否能够在可接受时间和内存内完成。

### 8.2 增加结构预测指标

建议为每个参考片段记录：

- k-mer 重复率，`k=8/12/16/20/31`；
- 最长同聚物；
- sampled pairwise edit-distance 分位数；
- 每层节点数；
- expanded-layer 相邻节点乘积；
- primary DAG 总边数和出度分位数；
- 去重率；
- 构建各阶段时间和峰值内存。

### 8.3 验证构建顺序敏感性

对同一序列集合比较：

- 原始窗口顺序；
- 随机窗口顺序；
- 多个随机 seed。

这能够区分序列分布影响与 greedy center 选择影响。

### 8.4 优化 Phase 2

当前 Phase 2 的全量相邻层比较限制了更大规模实验。建议优先研究：

- 使用父子候选关系限制 rebinding 搜索范围；
- 基于距离下界或 sketch 预过滤；
- 对相邻层建立临时 metric index；
- 并行化 parent-child 距离计算；
- 在构建中提前监测并拒绝稠密配置。

---

## 9. 结果文件

主要结果：

- `.tmp_experiments/scaling_runs/`
- `.tmp_experiments/scaling_runs_20k/`
- `.tmp_experiments/distribution_control_runs/`
- `.tmp_experiments/overnight_repeat/runs/`

实验元数据与任务状态：

- `.tmp_experiments/overnight_repeat/metadata.tsv`
- `.tmp_experiments/overnight_repeat/jobs.tsv`
- `.tmp_experiments/overnight_repeat/status.tsv`
- `.tmp_experiments/overnight_repeat/plot_summary.csv`

历史 layer/radius 实验总结：

- `navigamer_cpp/LAYER_RADIUS_EXPERIMENT_SUMMARY_ZH.md`
