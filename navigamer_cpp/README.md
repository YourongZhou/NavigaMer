# NavigaMer v7 - C++ 实现

基于 C++17 的 NavigaMer v7 (Multilateration-Enhanced) 实现，与 Python 版 `world_demo/src` 算法一致。

## 构建

```bash
# 使用 Makefile（仅需 g++）
make

# 或使用 CMake（若已安装）
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make
```

## 用法

```bash
# 内置演示（默认 500 条 reads，可指定 --size N）
./navigamer demo [--size 200]

# 从 FASTA/FASTQ 构建索引
./navigamer build --ref <ref.fa|或序列字符串> --reads <reads.fq|或序列字符串>

# 单条查询（先根据 reads 构建索引，再查 query）
./navigamer query --reads <reads.fq> --query <序列> [--tolerance 2] [--mode adaptive|greedy|exhaustive]

# 完整流程：构建 + 对所有 reads 查询并输出 TSV
./navigamer run --ref <ref.fa> --reads <reads.fq> [--tolerance 2] [--out navigamer_out.tsv]
```

## 模块说明

| 模块 | 说明 |
|------|------|
| `include/structure.hpp` | 核心数据结构：`BioSequence`、`WorldNode`、半径常量 R_SW/R_MW/R_LW |
| `include/tools.hpp` | 编辑距离 `compute_distance`、FPS `farthest_point_sampling`、`shuffle_indices` |
| `include/index_builder.hpp` | 索引构建：Phase 0 去重 → Phase 1 骨架 → Phase 2 稠密连网 → Phase 3 信标 → Phase 4 叶子挂载 |
| `include/search_engine.hpp` | 搜索：`search_adaptive` / `search_greedy` / `search_exhaustive` / `search_brute_force` |
| `include/io_utils.hpp` | I/O：`load_reference`、`load_reads`、`write_tsv`、`search_results_to_tsv_rows` |

## 与 Python 版对应关系

- **structure.py** → `structure.hpp/cpp`
- **tools.py** → `tools.hpp/cpp`（Levenshtein、FPS）
- **index_builder.py** → `index_builder.hpp/cpp`（五阶段流水线）
- **search_engine.py** → `search_engine.hpp/cpp`（Beacon Pruning、Sibling Pruning）
- **io_utils.py** → `io_utils.hpp/cpp`（FASTA/FASTQ/TSV）

Phase 5（FM-Index 定位 ref_positions）未实现，输出 TSV 时 `ref_positions` 可为空；若需基因组坐标，可后续对接 C++ BWT/SA 库。
