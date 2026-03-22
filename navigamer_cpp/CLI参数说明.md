# NavigaMer C++ 命令行参考

本文档对应仓库内 `navigamer_cpp` 可执行文件 `navigamer` 的当前实现（`src/main.cpp` 等）。用于快速查阅子命令、参数含义与典型用法。

## 编译与依赖

- **C++17**、**OpenMP**（CMake 或 Makefile 均会链接）。
- 在项目目录下任选其一：

```bash
cd navigamer_cpp
make -j
# 或
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# CMake 生成物通常在 build/navigamer
```

可执行文件名为 **`navigamer`**（Makefile 直接生成在 `navigamer_cpp/` 下）。

## 公共参数（多数子命令可用）

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--r-sw` | int | `5` | 小世界层（SW）半径，对应 `structure.hpp` 中 `R_SW` |
| `--r-mw` | int | `15` | 中世界层（MW）半径 `R_MW` |
| `--r-lw` | int | `30` | 大世界层（LW）半径 `R_LW` |

三层半径共同决定几何索引的结构与剪枝行为；与 Python 版设计对齐。

**输入约定（`io_utils`）：**

- **`--ref`**：若为**已存在文件路径**，按 **FASTA** 解析（首行 `>` 后为序列 ID）；否则整段字符串视为**一条参考序列**（ID 固定为 `ref`）。
- **`--reads`**：若为**已存在文件路径**，按 **FASTQ** 解析（`@` 行 ID，下一行为碱基，再两行 `+` 与质量值）；否则整段字符串视为**单条 read**（ID 为 `query_0`）。

## 子命令一览

```
navigamer demo [选项]
navigamer build --ref ... --reads ... [选项]
navigamer query --reads ... --query ... [--ref ...] [选项]
navigamer run --ref ... --reads ... [选项]
navigamer benchmark --ref ... --reads ... [选项]
```

---

### `demo`

内置随机参考与 reads，对比 **adaptive / exhaustive / brute_force** 在小样本上的召回（与 BF 一致则计为命中），用于快速自检。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--size` | `500` | 生成的 read 条数 |
| `--r-sw` / `--r-mw` / `--r-lw` | 5 / 15 / 30 | 索引半径 |

**说明：** Demo 中参考长度固定约 50kb，read 长度 20，突变率为 0；仅对前 50 条 read 与 BF 对比。

---

### `build`

从参考与 reads **构建索引**并打印规模信息。

| 必需 | 说明 |
|------|------|
| `--ref` | 参考 FASTA 或序列字符串 |
| `--reads` | reads FASTQ 或单条序列字符串 |

当前实现**不将索引落盘**；日志提示需用 `run` 跑完整流程。参数 `--r-sw` 等同上。

---

### `query`

在**已加载的 reads 集合上建索引**，再对 **`--query` 给定的一条查询序列**做搜索。

| 必需 | 说明 |
|------|------|
| `--reads` | 索引来源（文件或单条序列） |
| `--query` | 查询 DNA 字符串 |

| 可选 | 默认 | 说明 |
|------|------|------|
| `--ref` | 可省略 | 未使用于核心逻辑；占位 |
| `--tolerance` | `2` | 允许的最大编辑距离（或算法使用的距离阈值，见引擎实现） |
| `--mode` | `adaptive` | `adaptive` \| `greedy` \| `exhaustive` |

输出为 stdout：命中条数、各 hit 的 ID 与距离等。

---

### `run`

**构建索引 + 对全部 reads 用 adaptive 搜索**，可选写出 TSV。

| 必需 | 说明 |
|------|------|
| `--ref` | 参考 |
| `--reads` | reads |

| 可选 | 默认 | 说明 |
|------|------|------|
| `--tolerance` | `2` | 搜索容差 |
| `--out` | 空 | 若指定路径，写入 TSV；不指定则仅 stderr 汇总行数 |

内部使用 OpenMP 并行处理各 read。

---

### `benchmark`

面向**参考滑窗建索引、reads 作查询**的评测流程：将参考切为长度为 `--window`、步长为 `--stride` 的窗口，每条窗口序列为索引中一条序列并带参考坐标；再对每个 query read 做 **adaptive** 搜索，TSV 中附带 `dist_calcs` 等统计列。

| 必需 | 说明 |
|------|------|
| `--ref` | 参考 FASTA/序列 |
| `--reads` | 查询 FASTQ/序列（此处语义为 **query reads**） |

| 可选 | 默认 | 说明 |
|------|------|------|
| `--tolerance` | `2`（程序内变量初值；帮助信息中曾写 5，以代码为准） | 搜索容差 |
| `--window` | `200` | 参考上每条索引序列长度 |
| `--stride` | `1` | 窗口起点步长 |
| `--out` | 空 | 输出 TSV 路径 |

若某条 query 无命中，仍会输出一行占位，并填统计字段。

---

## TSV 列（`run` / `benchmark`）

`run` 输出列包括：`query_id`, `hit_id`, `distance`, `ref_positions`, `read_id`, `read_len`, `ref_id`, `strand`, `query_start`, `reference_start`, `aligned_length`, `score`, `edit_distance`, `query_fragment`, `reference_fragment`, `bwt_start`, `bwt_end`。

`benchmark` 在上述基础上增加：`dist_calcs`, `leaf_verify_count`, `candidate_count_for_prune`, `beacon_prune_count`。

---

## 其它测试二进制

- **`test_recall`**：随机数据上验证 `search_adaptive` 相对 brute-force 的召回（无独立丰富 CLI，直接运行即可）。
- **`test_distance_bound`**：距离界相关单元测试。

与主程序 `navigamer` 的参数实验无关，一般不通过主 CLI 传参。

---

## 与 notebook 联调

仓库内 `navigamer_cpp/params_test.ipynb` 用 Python 封装子进程调用，可在表格/变量中批量修改 `demo`、`query`、`run`、`benchmark` 的参数并查看 stdout/stderr。请将 `NAVIGAMER` 指向你本机编译出的 `navigamer` 可执行文件路径。
