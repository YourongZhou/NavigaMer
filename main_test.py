#!/usr/bin/env python3
"""
NavigaMer v7 (Multilateration-Enhanced Edition) - 主测试入口
整合所有测试用例：0 FN 验证、Beacon 剪枝率、压缩率与效率对比
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.structure import BioSequence
from src.tools import generate_reference_sequence, generate_reads_with_mutations
from src.index_builder import BioGeometryIndexBuilder
from src.fm_index import FMIndex
from tests.benchmark import (
    test_radius_compliance,
    test_distance_heatmap,
    test_efficiency_benchmark,
    test_false_negative,
    test_compression_summary,
    test_fm_index_position_verification,
)


def main():
    print("=" * 70)
    print("NavigaMer v7 (Multilateration-Enhanced) - 测试验证套件")
    print("=" * 70)

    # ===== 1. 测试数据生成 =====
    print("\n[阶段 1] 生成测试数据...")
    random.seed(42)

    print("  生成参考序列 (50,000 bp)...")
    reference_seq = generate_reference_sequence(length=50000, seed=42)
    print(f"  参考序列长度: {len(reference_seq)} bp")

    num_reads = 500
    read_length = 20
    mutation_rate = 0.0

    print(f"  生成 {num_reads} 条 Reads (len={read_length}, "
          f"mutation={mutation_rate})...")
    t_start = time.perf_counter()
    raw_sequences = generate_reads_with_mutations(
        reference_seq=reference_seq,
        num_reads=num_reads,
        read_length=read_length,
        mutation_rate=mutation_rate,
        seed=42,
    )
    t_gen = time.perf_counter() - t_start
    print(f"  生成了 {len(raw_sequences)} 条 Reads，耗时 {t_gen:.4f}s")
    print(f"  示例: {raw_sequences[0]}")

    # ===== 2. 构建索引 =====
    print("\n[阶段 2] 构建索引 (v7 Multilateration-Enhanced)...")
    try:
        fm_index = FMIndex(reference_seq, ref_id="ref")
    except ImportError as e:
        fm_index = None
        print(f"  [跳过] FM-Index 不可用 ({e})，Phase 5 不执行，ref_positions 将为空")
    builder = BioGeometryIndexBuilder()

    t_start = time.perf_counter()
    builder.build(raw_sequences, fm_index=fm_index)
    t_build = time.perf_counter() - t_start
    print(f"  索引构建完成，耗时 {t_build:.4f}s")

    stats = builder.get_statistics()
    print("\n  索引统计:")
    print(f"    Raw: {stats['raw_count']} -> Unique: {stats.get('unique_count', stats['raw_count'])}")
    print(f"    SW: {stats['sw_count']} (R={builder.radius_config[1]})")
    print(f"    MW: {stats['mw_count']} (R={builder.radius_config[2]})")
    print(f"    LW: {stats['lw_count']} (R={builder.radius_config[3]})")
    print(f"    压缩率: {stats.get('compression_ratio', 0):.1%}")
    print(f"    DAG 冗余度: {stats.get('dag_redundancy', 0):.1f}%")

    # ===== 3. 运行测试用例 =====
    print("\n[阶段 3] 运行测试用例...")

    test_results = {}

    # 获取去重后的唯一序列列表用于 brute force 对比
    unique_seqs = list(builder.unique_sequences.values())

    tests = [
        ("test_1", "几何半径验证",
         lambda: test_radius_compliance(builder)),
        ("test_2", "距离分布热力图",
         lambda: test_distance_heatmap(builder)),
        ("test_3", "冗余与效率对比",
         lambda: test_efficiency_benchmark(builder, unique_seqs, seed=42, tolerance=2)),
        ("test_4", "False Negative",
         lambda: test_false_negative(builder, unique_seqs, seed=42, tolerance=2)),
        ("test_5", "压缩率与结构摘要",
         lambda: test_compression_summary(builder)),
        ("test_6", "FM-Index 位置验证与 TSV 输出",
         lambda: test_fm_index_position_verification(
             builder, reference_seq, ref_id="ref", num_queries=20, tolerance=0)),
    ]

    for test_id, test_name, test_fn in tests:
        try:
            print(f"\n>>> 开始 {test_id}: {test_name}...")
            result = test_fn()
            test_results[test_id] = result
            print(f">>> {test_id} {'通过' if result else '失败'}")
        except Exception as e:
            print(f">>> {test_id} 执行出错: {e}")
            import traceback
            traceback.print_exc()
            test_results[test_id] = False

    # ===== 4. 测试总结 =====
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    for test_id, test_name, _ in tests:
        passed = test_results.get(test_id, False)
        mark = '✓ 通过' if passed else '✗ 失败'
        print(f"  {test_id} ({test_name}): {mark}")

    all_passed = all(test_results.values())
    print(f"\n{'✓ 所有测试用例通过！' if all_passed else '✗ 部分测试用例失败'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
