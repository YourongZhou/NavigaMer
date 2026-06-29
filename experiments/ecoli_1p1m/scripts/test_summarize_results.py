#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import summarize_results


SUMMARY_HEADER = (
    "dataset\tmethod\tvariant\tread_count\traw_candidate_count_mean\t"
    "raw_candidate_count_median\traw_candidate_count_p95\t"
    "raw_candidate_count_p99\taccepted_candidate_count_mean\t"
    "accepted_candidate_count_median\taccepted_candidate_count_p95\t"
    "accepted_candidate_count_p99\tretrieval_milliseconds_mean\t"
    "retrieval_milliseconds_median\tretrieval_milliseconds_p95\t"
    "retrieval_milliseconds_p99\tverification_milliseconds_mean\t"
    "verification_milliseconds_median\tverification_milliseconds_p95\t"
    "verification_milliseconds_p99\ttotal_milliseconds_mean\t"
    "total_milliseconds_median\ttotal_milliseconds_p95\t"
    "total_milliseconds_p99\toracle_read_count\ttrue_neighbor_count_total\t"
    "false_negative_count_total\tmean_recall\tmean_raw_candidate_blowup\t"
    "mean_accepted_candidate_blowup\n"
)


def summary_row(dataset: str, method: str, variant: str, fn: int,
                recall: str = "1") -> str:
    return (
        f"{dataset}\t{method}\t{variant}\t10\t12\t10\t20\t25\t2\t1\t3\t4\t"
        "1.5\t1.0\t2.0\t3.0\t0.5\t0.4\t0.8\t0.9\t2.0\t1.5\t3.0\t4.0\t"
        f"10\t20\t{fn}\t{recall}\t6\t1\n"
    )


class SummarizeResultsTest(unittest.TestCase):
    def test_summarize_results_writes_result_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "ecoli_1p1m_formal"
            out = root / "summary"
            (base / "collected").mkdir(parents=True)
            (base / "candidates_main_shared").mkdir()
            (base / "navigamer_main_rebuilt").mkdir()
            (base / "path_trace_v1").mkdir()
            (base / "locality_minimal_v1" / "runs").mkdir(parents=True)
            (base / "prefetch_ab_v1").mkdir()
            (base / "corner_v1").mkdir()
            (base / "corner_v2").mkdir()
            (base / "layer_ablation_v1").mkdir()
            (base / "anchor_ablation_v1").mkdir()
            (base / "system_ab_v1").mkdir()
            (base / "perf_stat_minghao_v1").mkdir()
            (base / "mapper_baselines_v1" / "indexes").mkdir(parents=True)
            (base / "mapper_baselines_v1" / "runs").mkdir()
            (base / "navigamer_main_rerun_v1" / "toy_work").mkdir(parents=True)
            (base / "reads").mkdir()
            (base / "results_summary_v1" / "validation_logs").mkdir(parents=True)

            (base / "collected" / "oracle_summary_all.tsv").write_text(
                SUMMARY_HEADER
                + summary_row("prefix100000_tau1", "NavigaMer", "adaptive", 0)
                + summary_row("prefix100000_tau1", "TensorSketch", "d64", 2, "0.9"),
                encoding="utf-8",
            )
            (base / "collected" / "main_summary_all.tsv").write_text(
                SUMMARY_HEADER
                + summary_row("main_exact_10000", "NavigaMer", "adaptive", 0, "NA")
                + summary_row("main_exact_10000", "qgram-safe", "q5", 0, "NA"),
                encoding="utf-8",
            )
            (base / "candidates_main_shared" / "build_summary.tsv").write_text(
                "method\tvariant\treused\twall_seconds\tindex_bytes\n"
                "qgram-safe\tq5\ttrue\t4.5\t1024\n",
                encoding="utf-8",
            )
            (base / "navigamer_main_rebuilt" / "build_scale.csv").write_text(
                "prefix_len,total_build_ms,world_node_count,finest_world_count\n"
                "1100000,9000,700,400\n",
                encoding="utf-8",
            )
            (base / "navigamer_main_rebuilt" / "ecoli_1p1m_w150_s1.navidx").write_bytes(
                b"navidx"
            )
            (base / "path_trace_v1" / "roundrobin.path_trace.tsv").write_text(
                "query_id\tquery_ordinal\tworld_visit_count\tleaf_visit_count\t"
                "world_unique_count\tleaf_unique_count\tprev_world_jaccard\t"
                "prev_leaf_jaccard\tworld_path\n"
                "q0\t0\t10\t1\t10\t1\tNA\tNA\t1,2\n"
                "q1\t1\t12\t1\t12\t1\t0.5\t0\t" + ("1," * 70_000) + "\n"
                "q2\t2\t14\t1\t14\t1\t1.0\t0\t3,4\n",
                encoding="utf-8",
            )
            (base / "prefetch_ab_v1" / "roundrobin_prefetch_off.tsv").write_text(
                "query_id\tquery_time_ms\tsearch_prefetch_enabled\n"
                "q0\t10\tfalse\n"
                "q1\t30\tfalse\n",
                encoding="utf-8",
            )
            (base / "locality_minimal_v1" / "runs" / "sorted.path_trace.tsv").write_text(
                "query_id\tquery_ordinal\tworld_visit_count\tleaf_visit_count\t"
                "world_unique_count\tleaf_unique_count\tprev_world_jaccard\t"
                "prev_leaf_jaccard\tprev_world_overlap_count\tprev_leaf_overlap_count\t"
                "world_path\tleaf_path\n"
                "q0\t0\t8\t1\t8\t1\tNA\tNA\t0\t0\t1,2\t7\n"
                "q1\t1\t12\t1\t12\t1\t0.75\t1\t6\t1\t1,2,3\t7\n",
                encoding="utf-8",
            )
            (base / "locality_minimal_v1" / "runs" / "sorted.tsv").write_text(
                "query_id\tquery_time_ms\tsearch_prefetch_enabled\n"
                "q0\t40\tfalse\n"
                "q1\t60\tfalse\n"
                "q1\t60\tfalse\n",
                encoding="utf-8",
            )
            (base / "locality_minimal_v1" / "runs" / "sorted.stderr_time.log").write_text(
                "\tPercent of CPU this job got: 99%\n"
                "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:02.50\n"
                "\tMaximum resident set size (kbytes): 12345\n",
                encoding="utf-8",
            )
            (base / "corner_v1" / "sample.tsv").write_text(
                "query_id\thit_id\tquery_path_class\tpath_contained_step_count\t"
                "path_overlap_step_count\tpath_uncovered_step_count\t"
                "mbb_rect_fallback_count\tleaf_verify_count\tresult_count\t"
                "query_time_ms\n"
                "q0\th0\tcontained\t2\t0\t0\t0\t1\t1\t10\n"
                "q1\th1\toverlap\t1\t1\t0\t0\t3\t2\t20\n"
                "q1\th2\toverlap\t1\t1\t0\t0\t3\t2\t20\n"
                "q2\t\tuncovered\t0\t0\t1\t1\t5\t0\t30\n",
                encoding="utf-8",
            )
            (base / "corner_v2" / "sample.tsv").write_text(
                "query_id\thit_id\tquery_path_class\tpath_contained_step_count\t"
                "path_overlap_step_count\tpath_uncovered_step_count\t"
                "mbb_rect_fallback_count\tleaf_verify_count\tresult_count\t"
                "query_time_ms\n"
                "q3\th3\toverlap\t0\t2\t0\t0\t7\t1\t40\n",
                encoding="utf-8",
            )
            (base / "corner_v2" / "run_status.tsv").write_text(
                "dataset\ttau\tfastq\tout\tstarted_at\tfinished_at\texit_code\n"
                "sample\t1\treads.fq\tsample.tsv\tstart\tfinish\t0\n",
                encoding="utf-8",
            )
            (base / "layer_ablation_v1" / "layers.csv").write_text(
                "dataset,query_id,source_id,query_length,L,r_leaf,alpha,"
                "radius_schedule,query_time_ms,world_access_count,"
                "node_access_count,edge_access_count,anchor_distance_count,"
                "bound_check_count,candidate_count,candidate_verify_count,"
                "result_count,source_recovered,no_fn\n"
                "ref,0,ref_0,8,2,2,0.5,4|2,10,5,1,2,3,4,1,1,1,1,1\n"
                "ref,1,ref_4,8,2,2,0.5,4|2,30,7,1,4,5,6,1,1,1,1,1\n"
                "ref,0,ref_0,8,3,2,0.5,8|4|2,20,4,1,3,4,5,1,1,1,1,1\n",
                encoding="utf-8",
            )
            (base / "anchor_ablation_v1" / "anchors.tsv").write_text(
                "strategy\tanchor_count\tquery_count\twindow_count\t"
                "mean_envelope_size\tmean_exact_calls\t"
                "mean_true_neighbor_count\tfalse_negative_count_total\t"
                "source_recovery_rate\tbound_check_pass_rate\tpruning_ratio\t"
                "reference_path\twindow_length\tstride\tquery_edits\ttolerance\tseed\n"
                "random\t2\t4\t100\t20\t20\t1\t0\t1\t0.2\t0.8\tref.fa\t40\t20\t2\t2\t11\n"
                "proximal\t2\t4\t100\t3\t3\t1\t0\t1\t0.03\t0.97\tref.fa\t40\t20\t2\t2\t11\n",
                encoding="utf-8",
            )
            (base / "system_ab_v1" / "scan_prefetch_off.tsv").write_text(
                "query_id\tquery_time_ms\tdist_calcs\tleaf_verify_count\t"
                "mbb_scan_child_checks\tmbb_rect_index_queries\t"
                "mbb_rect_candidate_children\tcenter_distance_calls_after_qgram\t"
                "search_prefetch_enabled\tsearch_qgram_prefilter_enabled\t"
                "qgram_prune_ratio\tresult_count\n"
                "q0\t10\t100\t2\t30\t0\t0\t5\tfalse\tfalse\t0\t1\n"
                "q1\t30\t200\t4\t50\t0\t0\t7\tfalse\tfalse\t0\t1\n",
                encoding="utf-8",
            )
            (base / "system_ab_v1" / "scan_prefetch_off.time.log").write_text(
                "\tUser time (seconds): 12.0\n"
                "\tSystem time (seconds): 1.0\n"
                "\tPercent of CPU this job got: 200%\n"
                "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:06.50\n"
                "\tMaximum resident set size (kbytes): 123456\n"
                "\tMajor (requiring I/O) page faults: 2\n"
                "\tMinor (reclaiming a frame) page faults: 100\n",
                encoding="utf-8",
            )
            (base / "perf_stat_minghao_v1" / "scan_prefetch_off.tsv").write_text(
                "query_id\tquery_time_ms\n"
                "q0\t10\n"
                "q1\t30\n",
                encoding="utf-8",
            )
            (base / "perf_stat_minghao_v1" / "scan_prefetch_off.perf.tsv").write_text(
                "# started on Mon Jun 29 15:09:48 2026\n"
                "\n"
                "1000\t\tcycles\t2000\t100.00\t\t\n"
                "500\t\tinstructions\t2000\t100.00\t0.50\tinsn per cycle\n"
                "80\t\tcache-references\t2000\t100.00\t\t\n"
                "20\t\tcache-misses\t2000\t100.00\t25.00\t% of all cache refs\n"
                "300\t\tbranches\t2000\t100.00\t\t\n"
                "30\t\tbranch-misses\t2000\t100.00\t10.00\t% of all branches\n"
                "4\t\tpage-faults\t2000\t100.00\t\t\n"
                "3\t\tminor-faults\t2000\t100.00\t\t\n"
                "1\t\tmajor-faults\t2000\t100.00\t\t\n",
                encoding="utf-8",
            )
            (base / "perf_stat_minghao_v1" / "scan_prefetch_off.time.log").write_text(
                "\tPercent of CPU this job got: 200%\n"
                "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:06.50\n"
                "\tMaximum resident set size (kbytes): 123456\n",
                encoding="utf-8",
            )
            (base / "perf_stat_minghao_v1" / "scan_prefetch_on.tsv").write_text(
                "query_id\tquery_time_ms\n"
                "q0\t10\n"
                "q1\t30\n",
                encoding="utf-8",
            )
            (base / "perf_stat_minghao_v1" / "scan_prefetch_on.perf.tsv").write_text(
                "1000\t\tcycles\t2000\t100.00\t\t\n"
                "0\t\tinstructions\t2000\t100.00\t0.00\tinsn per cycle\n"
                "0\t\tcache-references\t2000\t100.00\t\t\n"
                "0\t\tcache-misses\t2000\t100.00\t\t\n",
                encoding="utf-8",
            )
            (base / "mapper_baselines_v1" / "indexes" / "build_summary.tsv").write_text(
                "method\tbuild_wall_seconds\tmax_rss_kb\tindex_bytes\n"
                "minimap2\t0.08\t15652\t3953048\n",
                encoding="utf-8",
            )
            (base / "mapper_baselines_v1" / "runs" / "toy.truth.tsv").write_text(
                "read_id\tfamily\ttau\tsource_start\tsource_end\tseed\tsequence_length\tedit_script_json\n"
                "r0\texact\t0\t10\t160\t1\t150\t[]\n"
                "r1\texact\t0\t50\t200\t1\t150\t[]\n"
                "r2\texact\t0\t90\t240\t1\t150\t[]\n",
                encoding="utf-8",
            )
            (base / "mapper_baselines_v1" / "runs" / "toy.minimap2.sam").write_text(
                "@SQ\tSN:ref\tLN:1000\n"
                "r0\t0\tref\t11\t60\t150M\t*\t0\t0\t*\t*\n"
                "r1\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*\n"
                "r2\t0\tref\t300\t20\t150M\t*\t0\t0\t*\t*\n",
                encoding="utf-8",
            )
            (base / "mapper_baselines_v1" / "runs" / "toy.minimap2.time.log").write_text(
                "\tPercent of CPU this job got: 200%\n"
                "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:02.00\n"
                "\tMaximum resident set size (kbytes): 20000\n",
                encoding="utf-8",
            )
            (base / "reads" / "toy.truth.tsv").write_text(
                "read_id\tfamily\ttau\tsource_start\tsource_end\tseed\tsequence_length\tedit_script_json\n"
                "r0\texact\t0\t10\t160\t1\t150\t[]\n"
                "r1\texact\t0\t50\t200\t1\t150\t[]\n"
                "r2\texact\t0\t90\t240\t1\t150\t[]\n",
                encoding="utf-8",
            )
            (base / "navigamer_main_rerun_v1" / "toy_work" / "run_meta.json").write_text(
                '{"dataset":"toy","elapsed_seconds":12.5,'
                '"command":["navigamer","query-index-batch","--index","toy.navidx"]}',
                encoding="utf-8",
            )
            (base / "navigamer_main_rerun_v1" / "toy_work" / "navigamer_benchmark.tsv").write_text(
                "query_id\thit_id\tread_id\treference_start\tref_positions\t"
                "dist_calcs\tleaf_verify_count\tcandidate_count_for_prune\t"
                "result_count\tquery_time_ms\n"
                "r0\tref_10\tr0\t10\t[[\"ref\",10,160,\"+\"]]\t100\t2\t4\t1\t5\n"
                "r2\tref_300\tr2\t300\t[[\"ref\",300,450,\"+\"]]\t200\t3\t6\t1\t7\n",
                encoding="utf-8",
            )
            logs = base / "results_summary_v1" / "validation_logs"
            (logs / "test_recall.log").write_text(
                "=== Summary: 11 passed, 0 failed ===\nALL PASSED\n",
                encoding="utf-8",
            )
            (logs / "test_mbb_filter_equivalence.log").write_text(
                "MBB filter equivalence tests passed\n",
                encoding="utf-8",
            )

            summarize_results.summarize(base, out)

            reliability = (out / "result1_no_fn_oracle.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(
                reliability[0],
                "dataset\tmethod\tvariant\tread_count\ttrue_neighbor_count_total\t"
                "false_negative_count_total\tmean_recall\tstatus",
            )
            self.assertIn(
                "prefix100000_tau1\tNavigaMer\tadaptive\t10\t20\t0\t1\tPASS",
                reliability,
            )
            self.assertIn(
                "prefix100000_tau1\tTensorSketch\td64\t10\t20\t2\t0.9\tFAIL",
                reliability,
            )

            candidate = (out / "result5_candidate_retrieval.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn("main_exact_10000\tqgram-safe\tq5\t10\t12", candidate)
            self.assertIn("main_exact_10000\tNavigaMer\tadaptive\t10\t12", candidate)

            mapper = (out / "result5_mapper_end_to_end.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "mapper_baselines_v1\ttoy\tminimap2\t3\t3\t2\t0.666667\t1\t0.333333\t"
                "2\t20000\t200\t0.08\t15652\t3953048",
                mapper,
            )
            navigamer_persisted = (
                out / "result5_navigamer_persisted_retrieval.tsv"
            ).read_text(encoding="utf-8")
            self.assertIn(
                "navigamer_main_rerun_v1\ttoy\tNavigaMer\tpersisted-query-index\t"
                "3\t2\t2\t0.666667\t1\t0.333333\t12.5\t6\t5\t5\t150\t2.5\t5\t1\ttoy.navidx",
                navigamer_persisted,
            )

            build = (out / "index_build_summary.tsv").read_text(encoding="utf-8")
            self.assertIn("baseline\tqgram-safe\tq5\t4.5\t1024", build)
            self.assertIn("NavigaMer\tNavigaMer\tadaptive\t9\t6", build)

            locality = (out / "result6_locality_prefetch.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn("path_trace\troundrobin\t3\t\t\t\t\t\t\t0.75\t0.5\t12\t1", locality)
            self.assertIn("timing\troundrobin_prefetch_off\t2\t20", locality)
            self.assertIn("path_trace\tlocality_minimal_v1/sorted\t2\t\t\t\t\t\t\t0.75\t0.75\t10\t1", locality)
            self.assertIn("timing\tsorted\t2\t50\t40\t40\t2.5\t12345\t99", locality)

            status = (out / "results_status.md").read_text(encoding="utf-8")
            self.assertIn("Result 1", status)
            self.assertIn("zero false negatives", status)
            self.assertIn("Mapper end-to-end rows: 1", status)

            evidence = (out / "result0_evidence_matrix.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "Result 1\tcorrectness/no false negatives\tsupported\t"
                "oracle_rows=2; navigamer_zero_fn=True",
                evidence,
            )
            self.assertIn(
                "Result 5\tcandidate retrieval and mapper baselines\tsupported\t"
                "candidate_rows=2; mapper_rows=1; navigamer_persisted_rows=1; candidate_methods=2",
                evidence,
            )
            claims = (out / "results_claims_1p1m.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "Result 1\tsupported\tNavigaMer returns zero false negatives",
                claims,
            )
            self.assertIn("Result 5b\tmixed\tCurrent q-gram/pigeonhole", claims)
            claims_md = (out / "results_claims_1p1m.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("# E. coli 1.1M Claim-Level Results", claims_md)
            self.assertIn("## Result 6 - preliminary", claims_md)
            draft = (out / "results_manuscript_draft_1p1m.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("# Draft 1.1M Results Section", draft)
            self.assertIn("zero false negatives", draft)
            self.assertIn("should not be interpreted as a throughput advantage", draft)

            modules = (out / "result1_module_equivalence.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn("test_recall\tPASS", modules)
            self.assertIn("test_mbb_filter_equivalence\tPASS", modules)

            corner = (out / "result4_corner_paths.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn("corner_v1/sample\t3\t1\t1\t1", corner)
            self.assertIn("corner_v2/sample\t1\t0\t1\t0", corner)
            self.assertNotIn("corner_v2/run_status", corner)
            self.assertIn("\t20\t", corner)

            corner_by_class = (
                out / "result4_corner_paths_by_class.tsv"
            ).read_text(encoding="utf-8")
            self.assertIn("corner_v1/sample\tcontained\t1\t0.333333", corner_by_class)
            self.assertIn("corner_v1/sample\toverlap\t1\t0.333333", corner_by_class)
            self.assertIn("corner_v1/sample\tuncovered\t1\t0.333333", corner_by_class)
            self.assertIn("corner_v2/sample\toverlap\t1\t1", corner_by_class)

            hierarchy = (out / "result3_hierarchy_ablation.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "layer_ablation_v1/layers\t2\t4|2\t2\t2\t1\t20\t10\t10\t6\t5\t3\t4\t1\t1",
                hierarchy,
            )
            self.assertIn(
                "layer_ablation_v1/layers\t3\t8|4|2\t1\t1\t1\t20",
                hierarchy,
            )

            anchors = (out / "result2_anchor_selection.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "anchor_ablation_v1/anchors\tproximal\t2\t4\t100\t3\t0\t1\t0.97",
                anchors,
            )

            system_ab = (out / "result6_system_ab.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "system_ab_v1/scan_prefetch_off\t2\t20\t10\t10\t6.5\t123456\t200",
                system_ab,
            )
            perf_stat = (out / "result6_perf_stat.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "perf_stat_minghao_v1/scan_prefetch_off\t2\t6.5\t123456\t200\t"
                "1000\t500\t0.5\t80\t20\t0.25\t300\t30\t0.1\t4\t3\t1\t500\t10",
                perf_stat,
            )
            self.assertIn(
                "perf_stat_minghao_v1/scan_prefetch_on\t2\t\t\t\t1000\t0\t\t0\t0\t\t"
                "\t\t\t\t\t\t500\t",
                perf_stat,
            )
            self.assertIn(
                "anchor_ablation_v1/anchors\trandom\t2\t4\t100\t20\t0\t1\t0.8",
                anchors,
            )


if __name__ == "__main__":
    unittest.main()
