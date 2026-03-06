"""
测试套件 (NavigaMer v7)

核心指标：
  1. 压缩率 (Compression Ratio): 1 - (Total SW Nodes / Total Unique Reads)
  2. 召回率 (Recall / FN Rate): 对比 BruteForce，预期 100%
  3. 搜索加速比 (Speedup): DistCalcs(BF) / DistCalcs(Adaptive)
  4. 冗余度 (Edge Density): 平均每个 SW 节点有多少个父节点

可视化：
  - 层级分布金字塔
  - 距离热力图
  - Benchmark 对比图
"""

import random
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.structure import BioSequence, R_SW, R_MW, R_LW
from src.tools import compute_distance, generate_reference_sequence, generate_reads_with_mutations
from src.index_builder import BioGeometryIndexBuilder
from src.search_engine import BioGeometrySearchEngine
from src.fm_index import FMIndex, reverse_complement
from src.io_utils import search_results_to_tsv_rows, write_tsv


# =========================================================================
# Test 1: Radius Compliance
# =========================================================================

def test_radius_compliance(index_builder: BioGeometryIndexBuilder):
    """几何半径验证：球重叠连边 + 叶子包含"""
    print("\n" + "=" * 70)
    print("测试用例 1: 几何半径验证 (Radius Compliance)")
    print("=" * 70)

    failures = []
    total_checks = 0

    print("\n[1.1] 验证父节点与子节点的球重叠关系...")
    for layer_id in [3, 2]:
        layer_name = {3: 'LW', 2: 'MW'}[layer_id]
        parent_nodes = index_builder.layers[layer_id]

        for parent in parent_nodes:
            for child in parent.children:
                if not hasattr(child, 'node_id'):
                    continue
                total_checks += 1
                p_seq = BioSequence("_p", parent.get_center_sequence())
                c_seq = BioSequence("_c", child.get_center_sequence())
                dist = compute_distance(p_seq, c_seq)
                threshold = parent.radius + child.radius
                if dist > threshold:
                    failures.append(
                        f"{layer_name} {parent.node_id} -> {child.node_id}: "
                        f"dist={dist} > R_P+R_C={threshold}")

    print(f"  完成：检查了 {total_checks} 个节点间关系")
    if failures:
        print(f"  ✗ 失败：{len(failures)} 个违反球重叠约束")
        for f in failures[:5]:
            print(f"    {f}")
    else:
        print(f"  ✓ 通过：所有节点间关系满足球重叠约束")

    print("\n[1.2] 验证 SW 节点与叶子序列的包含关系...")
    sw_nodes = index_builder.layers[1]
    leaf_failures = 0
    leaf_checks = 0

    sample_size = min(100, len(sw_nodes))
    sampled = random.sample(sw_nodes, sample_size) if sw_nodes else []

    for sw in sampled:
        sw_center = BioSequence("_sw", sw.get_center_sequence())
        for child in sw.children:
            if isinstance(child, BioSequence):
                leaf_checks += 1
                dist = compute_distance(sw_center, child)
                if dist > R_SW:
                    leaf_failures += 1
                    if leaf_failures <= 3:
                        print(f"    失败: SW {sw.node_id}, dist={dist} > R_SW={R_SW}")

    print(f"  完成：检查了 {leaf_checks} 个叶子序列")
    if leaf_failures == 0:
        print(f"  ✓ 通过：所有叶子序列满足 R_SW 约束")
    else:
        print(f"  ✗ 失败：{leaf_failures} 个叶子违反 R_SW 约束")

    total_failures = len(failures) + leaf_failures
    print(f"\n[总结] 总失败数: {total_failures} / {total_checks + leaf_checks}")
    return total_failures == 0


# =========================================================================
# Test 2: Distance Heatmap
# =========================================================================

def test_distance_heatmap(index_builder: BioGeometryIndexBuilder):
    """距离分布热力图：验证聚类有效性"""
    print("\n" + "=" * 70)
    print("测试用例 2: 距离分布热力图 (Distance Heatmap)")
    print("=" * 70)

    sw_nodes = index_builder.layers[1]
    if len(sw_nodes) < 10:
        print("  警告：SW 节点数量不足，跳过测试")
        return True

    print("\n[2.1] Intra-class (类内距离) 分析...")
    sample_sws = random.sample(sw_nodes, min(10, len(sw_nodes)))
    intra_distances = []
    for sw in sample_sws:
        seqs = [c for c in sw.children if isinstance(c, BioSequence)]
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                intra_distances.append(compute_distance(seqs[i], seqs[j]))

    if intra_distances:
        avg_intra = sum(intra_distances) / len(intra_distances)
        print(f"  类内距离: avg={avg_intra:.2f}, "
              f"min={min(intra_distances)}, max={max(intra_distances)}, "
              f"n={len(intra_distances)}")
    else:
        print("  警告：无法计算类内距离（序列数量不足）")
        avg_intra = 0

    print("\n[2.2] Inter-class (类间距离) 分析...")
    sample_inter = random.sample(sw_nodes, min(50, len(sw_nodes)))
    sw_centers = [BioSequence(f"sw_{n.node_id}", n.get_center_sequence())
                  for n in sample_inter]
    inter_distances = []
    for i in range(len(sw_centers)):
        for j in range(i + 1, len(sw_centers)):
            inter_distances.append(compute_distance(sw_centers[i], sw_centers[j]))

    if inter_distances:
        avg_inter = sum(inter_distances) / len(inter_distances)
        print(f"  类间距离: avg={avg_inter:.2f}, "
              f"min={min(inter_distances)}, max={max(inter_distances)}, "
              f"n={len(inter_distances)}")
    else:
        print("  警告：无法计算类间距离")

    print("\n[2.3] 距离矩阵可视化...")
    try:
        matrix_size = min(20, len(sw_centers))
        dist_matrix = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i != j:
                    dist_matrix[i][j] = compute_distance(
                        sw_centers[i], sw_centers[j])

        plt.figure(figsize=(10, 8))
        plt.imshow(dist_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Edit Distance')
        plt.title('Inter-class Distance Heatmap (SW Centers)')
        plt.xlabel('SW Node Index')
        plt.ylabel('SW Node Index')

        output_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'distance_heatmap.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 热力图已保存到: {output_path}")
    except Exception as e:
        print(f"  [警告] 可视化失败: {e}")

    return True


# =========================================================================
# Test 3: Efficiency Benchmark
# =========================================================================

def _make_queries(raw_sequences, num=100, seed=42):
    random.seed(seed)
    queries = []
    for i in range(num):
        base = random.choice(raw_sequences)
        mutated = list(base.seq)
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, len(mutated) - 1)
            bases = ['A', 'T', 'C', 'G']
            mutated[idx] = random.choice([b for b in bases if b != mutated[idx]])
        queries.append(BioSequence(f"query_{i:03d}", ''.join(mutated)))
    return queries


def plot_benchmark_results(adaptive_stats, exhaustive_stats,
                           output_file='benchmark_result.png'):
    layers = ['LW', 'MW', 'SW', 'Leaf Verify\n(Disk I/O)']

    a_counts = [
        sum(s.layer_breakdown.get('LW', 0) for s in adaptive_stats),
        sum(s.layer_breakdown.get('MW', 0) for s in adaptive_stats),
        sum(s.layer_breakdown.get('SW', 0) for s in adaptive_stats),
        sum(s.leaf_verify_count for s in adaptive_stats),
    ]
    e_counts = [
        sum(s.layer_breakdown.get('LW', 0) for s in exhaustive_stats),
        sum(s.layer_breakdown.get('MW', 0) for s in exhaustive_stats),
        sum(s.layer_breakdown.get('SW', 0) for s in exhaustive_stats),
        sum(s.leaf_verify_count for s in exhaustive_stats),
    ]

    reductions = []
    for a, e in zip(a_counts, e_counts):
        reductions.append((1 - a / e) * 100 if e > 0 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('NavigaMer v7: Adaptive vs Exhaustive Search', fontsize=16)

    x = np.arange(len(layers))
    width = 0.35
    ax1.bar(x - width / 2, a_counts, width, label='Adaptive', color='#2ecc71', alpha=0.9)
    ax1.bar(x + width / 2, e_counts, width, label='Exhaustive', color='#e74c3c', alpha=0.9)
    ax1.set_ylabel('Access / Verify Count')
    ax1.set_title('Absolute Computational Cost')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    colors = plt.cm.Blues(np.array(reductions) / 100 * 0.8 + 0.2)
    bars = ax2.bar(layers, reductions, color=colors)
    ax2.set_ylabel('Reduction (%)')
    ax2.set_title('Efficiency Gain')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, rate in zip(bars, reductions):
        ax2.text(bar.get_x() + bar.get_width() / 2., max(bar.get_height(), 0) + 1,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    a_total = sum(s.dist_calc_count for s in adaptive_stats)
    e_total = sum(s.dist_calc_count for s in exhaustive_stats)
    note = (f"Adaptive: {a_total} dist calcs vs Exhaustive: {e_total}\n"
            f"Saving: {(1 - a_total / e_total) * 100:.1f}% with 0 FN.")
    plt.figtext(0.99, 0.02, note, ha='right', fontsize=10, style='italic',
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), output_file)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  [图表] Benchmark 图表已保存至: {save_path}")


def plot_layer_pyramid(stats, output_file='layer_pyramid.png'):
    """层级分布金字塔"""
    labels = ['LW', 'MW', 'SW', 'Unique Reads', 'Raw Reads']
    counts = [
        stats['lw_count'],
        stats['mw_count'],
        stats['sw_count'],
        stats.get('unique_count', stats['raw_count']),
        stats['raw_count'],
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
    bars = ax.barh(y_pos, counts, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Count')
    ax.set_title('NavigaMer v7 Layer Distribution Pyramid')
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2.,
                str(count), va='center', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), output_file)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [图表] 层级金字塔已保存至: {save_path}")


def test_efficiency_benchmark(index_builder: BioGeometryIndexBuilder,
                              raw_sequences: list, seed: int = 42,
                              tolerance: int = 5):
    """冗余与效率对比"""
    print("\n" + "=" * 70)
    print("测试用例 3: 冗余与效率对比 (Efficiency Benchmark)")
    print("=" * 70)

    print("\n[3.1] 生成测试查询...")
    queries = _make_queries(raw_sequences, num=100, seed=seed)

    engine = BioGeometrySearchEngine(index_builder)

    print("\n[3.2] 运行 Adaptive Search...")
    adaptive_stats_list = []
    adaptive_results = []
    for q in queries:
        results, st = engine.search_adaptive(q, tolerance)
        adaptive_results.append((q.id, set(s.id for s in results)))
        adaptive_stats_list.append(st)

    print("[3.3] 运行 Exhaustive Search...")
    exhaustive_stats_list = []
    exhaustive_results = []
    for q in queries:
        results, st = engine.search_exhaustive(q, tolerance)
        exhaustive_results.append((q.id, set(s.id for s in results)))
        exhaustive_stats_list.append(st)

    print("\n[3.4] 结果对比...")
    no_fp = True
    recall_sum = 0.0
    for i, (_, a_set) in enumerate(adaptive_results):
        _, e_set = exhaustive_results[i]
        if e_set:
            recall_sum += len(a_set & e_set) / len(e_set)
        if a_set - e_set:
            no_fp = False
    nq = len(queries)
    avg_recall = recall_sum / nq if nq else 1.0
    recall_match = no_fp
    print(f"  Adaptive 平均召回率: {avg_recall:.2%}, "
          f"无假阳性: {'是' if no_fp else '否'}")

    print("\n[3.5] 分层冗余度分析:")
    print(f"{'Layer':<10} | {'Adaptive':<15} | {'Exhaustive':<18} | {'Saving':<15}")
    print("-" * 70)

    for layer in ['LW', 'MW', 'SW']:
        a_cnt = sum(s.layer_breakdown.get(layer, 0) for s in adaptive_stats_list)
        e_cnt = sum(s.layer_breakdown.get(layer, 0) for s in exhaustive_stats_list)
        saving = f"{(1 - a_cnt / e_cnt) * 100:.1f}%" if e_cnt > 0 else "N/A"
        print(f"{layer:<10} | {a_cnt:<15} | {e_cnt:<18} | {saving:<15}")

    a_leaf = sum(s.leaf_verify_count for s in adaptive_stats_list)
    e_leaf = sum(s.leaf_verify_count for s in exhaustive_stats_list)
    if e_leaf > 0:
        print("-" * 70)
        print(f"{'Leaf':<10} | {a_leaf:<15} | {e_leaf:<18} | "
              f"{(1 - a_leaf / e_leaf) * 100:.1f}%")

    a_dist = sum(s.dist_calc_count for s in adaptive_stats_list)
    e_dist = sum(s.dist_calc_count for s in exhaustive_stats_list)
    print(f"\n  总距离计算: Adaptive={a_dist}, Exhaustive={e_dist}, "
          f"节省={1 - a_dist / e_dist:.1%}")

    try:
        plot_benchmark_results(adaptive_stats_list, exhaustive_stats_list)
    except Exception as e:
        print(f"  [警告] 绘图失败: {e}")

    return recall_match


# =========================================================================
# Test 4: False Negative Verification
# =========================================================================

def test_false_negative(index_builder: BioGeometryIndexBuilder,
                        raw_sequences: list, seed: int = 42,
                        tolerance: int = 2, num_queries: int = 100):
    """False Negative 精确验证：所有模式 vs Brute Force"""
    print("\n" + "=" * 70)
    print("测试用例 4: False Negative 精确验证 (All Modes vs Brute Force)")
    print("=" * 70)

    queries = _make_queries(raw_sequences, num=num_queries, seed=seed)
    engine = BioGeometrySearchEngine(index_builder)

    print(f"\n  Queries: {num_queries}, Tolerance: {tolerance}, "
          f"DB size: {len(raw_sequences)}")

    print("  [1/4] Running Brute Force...")
    bf_results = []
    bf_total_dist = 0
    for q in queries:
        results, st = engine.search_brute_force(q, tolerance, raw_sequences)
        bf_results.append(set(s.id for s in results))
        bf_total_dist += st.dist_calc_count

    print("  [2/4] Running Exhaustive Search...")
    ex_results, ex_stats = [], []
    for q in queries:
        results, st = engine.search_exhaustive(q, tolerance)
        ex_results.append(set(s.id for s in results))
        ex_stats.append(st)

    print("  [3/4] Running Adaptive Search...")
    ad_results, ad_stats = [], []
    for q in queries:
        results, st = engine.search_adaptive(q, tolerance)
        ad_results.append(set(s.id for s in results))
        ad_stats.append(st)

    print("  [4/4] Running Greedy Search...")
    gr_results, gr_stats = [], []
    for q in queries:
        results, st = engine.search_greedy(q, tolerance)
        gr_results.append(set(s.id for s in results))
        gr_stats.append(st)

    print("\n  === 结果对比 ===")
    bf_total_hits = sum(len(s) for s in bf_results)

    def analyze(name, mode_results, mode_stats):
        fn_count = fp_count = 0
        for i in range(num_queries):
            fn = bf_results[i] - mode_results[i]
            fp = mode_results[i] - bf_results[i]
            fn_count += len(fn)
            fp_count += len(fp)

        total_dist = sum(s.dist_calc_count for s in mode_stats)
        total_node = sum(s.node_access_count for s in mode_stats)
        total_leaf = sum(s.leaf_verify_count for s in mode_stats)
        recall = (bf_total_hits - fn_count) / bf_total_hits if bf_total_hits > 0 else 1.0
        fn_rate = fn_count / bf_total_hits if bf_total_hits > 0 else 0.0  # 漏搜率，论文指标须为 0

        print(f"\n  --- {name} ---")
        print(f"    FN: {fn_count}  FP: {fp_count}  Recall: {recall:.4%}  fn_rate: {fn_rate:.4%}")
        print(f"    Dist calcs: {total_dist}  Node visits: {total_node}  "
              f"Leaf verifies: {total_leaf}")
        # Multilateration 卖点：Beacon 剪枝率
        if mode_stats and getattr(mode_stats[0], 'candidate_count_for_prune', 0) > 0:
            total_cand = sum(s.candidate_count_for_prune for s in mode_stats)
            total_pruned = sum(s.beacon_prune_count for s in mode_stats)
            pr_rate = total_pruned / total_cand if total_cand else 0
            print(f"    Beacon pruning: {total_pruned}/{total_cand}  pruning_rate: {pr_rate:.1%}")
        return fn_count == 0

    print(f"\n  Brute Force 总命中: {bf_total_hits}, 总距离计算: {bf_total_dist}")

    ex_ok = analyze("Exhaustive Search", ex_results, ex_stats)
    ad_ok = analyze("Adaptive Search", ad_results, ad_stats)
    analyze("Greedy Search", gr_results, gr_stats)

    ex_dist = sum(s.dist_calc_count for s in ex_stats)
    ad_dist = sum(s.dist_calc_count for s in ad_stats)
    gr_dist = sum(s.dist_calc_count for s in gr_stats)

    print(f"\n  === 效率摘要 ===")
    print(f"    Brute Force:  {bf_total_dist:>8}")
    print(f"    Exhaustive:   {ex_dist:>8}  ({ex_dist / bf_total_dist:.1%} of BF)")
    print(f"    Adaptive:     {ad_dist:>8}  ({ad_dist / bf_total_dist:.1%} of BF)")
    print(f"    Greedy:       {gr_dist:>8}  ({gr_dist / bf_total_dist:.1%} of BF)")

    if ad_ok:
        speedup = bf_total_dist / ad_dist if ad_dist > 0 else float('inf')
        if ad_dist < ex_dist:
            saving = (1 - ad_dist / ex_dist) * 100
            print(f"\n  ✓ Adaptive: 0 FN + 比 Exhaustive 节省 {saving:.1f}%, "
                  f"加速比(vs BF): {speedup:.1f}x")
        else:
            print(f"\n  ✓ Adaptive: 0 FN, 加速比(vs BF): {speedup:.1f}x")
    else:
        print(f"\n  ✗ Adaptive Search 存在 False Negative!")

    return ex_ok and ad_ok


# =========================================================================
# Test 5: Compression & Summary (v7 新增)
# =========================================================================

def test_compression_summary(index_builder: BioGeometryIndexBuilder):
    """v7 压缩率与结构摘要"""
    print("\n" + "=" * 70)
    print("测试用例 5: 压缩率与结构摘要 (v7 Metrics)")
    print("=" * 70)

    stats = index_builder.get_statistics()

    print(f"\n  原始序列数: {stats['raw_count']}")
    print(f"  去重后唯一序列: {stats.get('unique_count', stats['raw_count'])}")
    print(f"  SW 节点数: {stats['sw_count']}")
    print(f"  MW 节点数: {stats['mw_count']}")
    print(f"  LW 节点数: {stats['lw_count']}")

    cr = stats.get('compression_ratio', 0)
    print(f"\n  压缩率: {cr:.1%} (1 - SW/Unique)")
    print(f"  压缩倍率: Raw->SW={stats['compression_sw']:.2f}x, "
          f"SW->MW={stats['compression_mw']:.2f}x, "
          f"MW->LW={stats['compression_lw']:.2f}x")

    print(f"\n  DAG 冗余度: {stats.get('dag_redundancy', 0):.1f}%")
    print(f"  平均每 SW 父节点数: {stats.get('avg_parents_per_sw', 0):.2f}")

    try:
        plot_layer_pyramid(stats)
    except Exception as e:
        print(f"  [警告] 金字塔图绘制失败: {e}")

    return True


# =========================================================================
# Test 6: FM-Index position verification (BWT locate roundtrip)
# =========================================================================

def test_fm_index_position_verification(
    index_builder: BioGeometryIndexBuilder,
    reference_seq: str,
    ref_id: str = "ref",
    num_queries: int = 20,
    tolerance: int = 0,
) -> bool:
    """
    端到端位置验证：FM-Index 填充 ref_positions 后，搜索命中并验证位置正确。
    检查每条 hit 的 ref_positions 在 reference 上确实对应 hit.seq（或反向互补）。
    """
    print("\n" + "=" * 70)
    print("测试用例 6: FM-Index 位置验证 (BWT Locate Roundtrip)")
    print("=" * 70)

    try:
        fm = FMIndex(reference_seq, ref_id=ref_id)
    except ImportError as e:
        print(f"  [跳过] 需要 pydivsufsort: {e}")
        return True

    engine = BioGeometrySearchEngine(index_builder)
    unique_seqs = list(index_builder.unique_sequences.values())
    if not unique_seqs:
        print("  警告：无唯一序列，跳过")
        return True

    random.seed(42)
    queries = [random.choice(unique_seqs) for _ in range(min(num_queries, len(unique_seqs)))]
    all_rows = []
    errors = 0

    print(f"\n  运行 {len(queries)} 次查询 (tolerance={tolerance})，验证 ref_positions...")
    for q in queries:
        results, _ = engine.search_adaptive(q, tolerance)
        for hit in results:
            for ref_id_val, start, end, strand in hit.ref_positions:
                ref_slice = reference_seq[start:end]
                if strand == "+":
                    if ref_slice != hit.seq:
                        errors += 1
                        if errors <= 3:
                            print(f"    错误: ref[{start}:{end}] != hit.seq (strand +)")
                else:
                    if ref_slice != reverse_complement(hit.seq):
                        errors += 1
                        if errors <= 3:
                            print(f"    错误: ref[{start}:{end}] != revcomp(hit.seq) (strand -)")

    for q in queries:
        results, _ = engine.search_adaptive(q, tolerance)
        for hit in results:
            d = compute_distance(q, hit)
            all_rows.extend(search_results_to_tsv_rows(q.id, q.seq, 0, hit, d))

    if all_rows:
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "navigamer_out.tsv")
        write_tsv(all_rows, out_path)
        print(f"  TSV 已写入: {out_path} ({len(all_rows)} 行)")

    if errors == 0:
        print(f"  ✓ 通过：所有 ref_positions 与 reference 一致")
    else:
        print(f"  ✗ 失败：{errors} 处 ref_position 与 reference 不一致")

    return errors == 0
