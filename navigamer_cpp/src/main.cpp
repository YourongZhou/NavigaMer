/**
 * NavigaMer v7 (Multilateration-Enhanced) - C++ 主程序
 * 用法:
 *   navigamer build --ref <fasta|序列> --reads <fastq|序列> [--out-index <path>]
 *   navigamer query --index <dir> --query <序列> [--tolerance 2] [--mode adaptive|greedy|exhaustive]
 *   navigamer demo  [--size 500]  # 内置小规模演示
 */

#include "structure.hpp"
#include "index_builder.hpp"
#include "search_engine.hpp"
#include "io_utils.hpp"
#include "tools.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>

namespace {

void usage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " demo [--size N]           # 内置演示 (默认 N=500 reads)\n"
            << "  " << prog << " build --ref <path|seq> --reads <path|seq>  # 构建索引\n"
            << "  " << prog << " query --ref <path|seq> --reads <path|seq> --query <seq> [--tolerance 2] [--mode adaptive]\n"
            << "  " << prog << " run  --ref <path|seq> --reads <path|seq> [--tolerance 2] [--out <tsv>]  # 构建+查询全部 reads，输出 TSV\n"
            << "  " << prog << " benchmark --ref <fasta> --reads <fastq> [--tolerance 5] [--window 200] [--stride 1] [--out <tsv>]  # Ref 窗口建索引 + query 搜索 + stats\n";
}

// 生成随机 DNA 序列
std::string generate_reference(size_t length, unsigned seed) {
  static const char bases[] = "ATCG";
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(0, 3);
  std::string s;
  s.reserve(length);
  for (size_t i = 0; i < length; ++i) s += bases[dis(gen)];
  return s;
}

// 从参考序列生成带突变的 reads
std::vector<std::shared_ptr<navigamer::BioSequence>> generate_reads(
    const std::string& ref, size_t num_reads, size_t read_len, double mutation_rate, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<size_t> pos_dis(0, ref.size() > read_len ? ref.size() - read_len : 0);
  std::vector<std::shared_ptr<navigamer::BioSequence>> reads;
  for (size_t i = 0; i < num_reads; ++i) {
    size_t start = pos_dis(gen);
    std::string fragment = ref.substr(start, read_len);
    if (fragment.size() < read_len) continue;
    // 简单替换突变
    std::uniform_real_distribution<> mut_dis(0, 1);
    for (char& c : fragment) {
      if (mut_dis(gen) < mutation_rate) {
        static const char bases[] = "ATCG";
        std::uniform_int_distribution<> b(0, 3);
        c = bases[b(gen)];
      }
    }
    reads.push_back(std::make_shared<navigamer::BioSequence>("read_" + std::to_string(i), fragment));
  }
  return reads;
}

void run_demo(int size) {
  using namespace navigamer;
  std::cerr << "NavigaMer v7 (C++) - Demo with " << size << " reads\n";
  std::string ref = generate_reference(50000, 42);
  auto reads = generate_reads(ref, size, 20, 0.0, 42);

  BioGeometryIndexBuilder builder;
  builder.build(reads);

  auto stats = builder.get_statistics();
  std::cout << "Index: SW=" << builder.layers[1].size()
            << " MW=" << builder.layers[2].size()
            << " LW=" << builder.layers[3].size()
            << " compression=" << (stats.compression_ratio * 100) << "%\n";

  BioGeometrySearchEngine engine(builder);
  std::vector<std::shared_ptr<BioSequence>> unique_list;
  for (const auto& p : builder.unique_sequences) unique_list.push_back(p.second);

  int tolerance = 2;
  size_t adaptive_ok = 0, exhaustive_ok = 0, bf_ok = 0;
  for (size_t i = 0; i < std::min(size_t(50), reads.size()); ++i) {
    auto [adaptive_res, st_adapt] = engine.search_adaptive(*reads[i], tolerance);
    auto [exhaustive_res, st_ex] = engine.search_exhaustive(*reads[i], tolerance);
    auto [bf_res, st_bf] = engine.search_brute_force(*reads[i], tolerance, unique_list);
    if (!bf_res.empty()) bf_ok++;
    bool a_ok = false, e_ok = false;
    for (const auto& h : bf_res) {
      if (std::find_if(adaptive_res.begin(), adaptive_res.end(),
                       [&h](const std::shared_ptr<BioSequence>& x) { return x->id == h->id; }) != adaptive_res.end())
        a_ok = true;
      if (std::find_if(exhaustive_res.begin(), exhaustive_res.end(),
                       [&h](const std::shared_ptr<BioSequence>& x) { return x->id == h->id; }) != exhaustive_res.end())
        e_ok = true;
    }
    if (a_ok) adaptive_ok++;
    if (e_ok) exhaustive_ok++;
  }
  std::cout << "Recall (sample 50): adaptive=" << adaptive_ok << "/" << bf_ok
            << " exhaustive=" << exhaustive_ok << "/" << bf_ok << "\n";
  std::cerr << "Demo done.\n";
}

void run_build(const std::string& ref_input, const std::string& reads_input) {
  using namespace navigamer;
  auto [ref_id, ref_seq] = load_reference(ref_input);
  std::cerr << "Reference: " << ref_id << " length=" << ref_seq.size() << "\n";
  auto reads = load_reads(reads_input, ref_id);
  std::cerr << "Reads: " << reads.size() << "\n";

  BioGeometryIndexBuilder builder;
  builder.build(reads);
  std::cerr << "Build done. (Index serialization not implemented; use run for full pipeline.)\n";
}

void run_query(const std::string& /*ref_input*/, const std::string& reads_input,
               const std::string& query_seq, int tolerance, const std::string& mode) {
  using namespace navigamer;
  auto reads = load_reads(reads_input, "ref");
  if (reads.empty()) {
    std::cerr << "No reads loaded.\n";
    return;
  }
  BioGeometryIndexBuilder builder;
  builder.build(reads);

  BioGeometrySearchEngine engine(builder);
  BioSequence q("query", query_seq);

  if (mode == "greedy") {
    auto [res, st] = engine.search_greedy(q, tolerance);
    std::cout << "Greedy hits: " << res.size() << " (dist_calcs=" << st.dist_calc_count << ")\n";
    for (const auto& h : res) std::cout << "  " << h->id << " dist=" << compute_distance(query_seq, h->seq) << "\n";
  } else if (mode == "exhaustive") {
    auto [res, st] = engine.search_exhaustive(q, tolerance);
    std::cout << "Exhaustive hits: " << res.size() << " (dist_calcs=" << st.dist_calc_count << ")\n";
    for (const auto& h : res) std::cout << "  " << h->id << " dist=" << compute_distance(query_seq, h->seq) << "\n";
  } else {
    auto [res, st] = engine.search_adaptive(q, tolerance);
    std::cout << "Adaptive hits: " << res.size() << " (dist_calcs=" << st.dist_calc_count
              << " prune_rate=" << st.pruning_rate() << ")\n";
    for (const auto& h : res) std::cout << "  " << h->id << " dist=" << compute_distance(query_seq, h->seq) << "\n";
  }
}

void run_full(const std::string& ref_input, const std::string& reads_input,
              int tolerance, const std::string& out_tsv) {
  using namespace navigamer;
  auto [ref_id, ref_seq] = load_reference(ref_input);
  auto reads = load_reads(reads_input, ref_id);
  if (reads.empty()) {
    std::cerr << "No reads.\n";
    return;
  }
  BioGeometryIndexBuilder builder;
  builder.build(reads);
  BioGeometrySearchEngine engine(builder);

  std::vector<std::vector<std::string>> all_rows;
  std::vector<std::string> columns = {
      "query_id", "hit_id", "distance", "ref_positions", "read_id", "read_len",
      "ref_id", "strand", "query_start", "reference_start", "aligned_length",
      "score", "edit_distance", "query_fragment", "reference_fragment"};

  for (const auto& read : reads) {
    auto [res, st] = engine.search_adaptive(*read, tolerance);
    for (const auto& hit : res) {
      int ed = compute_distance(read->seq, hit->seq);
      auto rows = search_results_to_tsv_rows(read->id, read->seq, 0, *hit, ed);
      for (const auto& r : rows) {
        all_rows.push_back({
            r.query_id, r.hit_id, r.distance_str, r.ref_positions_json,
            r.read_id, r.read_len, r.ref_id, r.strand, r.query_start, r.reference_start,
            r.aligned_length, r.score, r.edit_distance, r.query_fragment, r.reference_fragment});
      }
    }
  }
  if (!out_tsv.empty())
    write_tsv(out_tsv, columns, all_rows);
  std::cerr << "Total rows: " << all_rows.size() << "\n";
}

// Benchmark: reference windows -> index; query reads -> search; output hits + SearchStats
void run_benchmark(const std::string& ref_input, const std::string& query_input,
                   int tolerance, int window_size, int stride,
                   const std::string& out_tsv) {
  using namespace navigamer;
  auto [ref_id, ref_seq] = load_reference(ref_input);
  if (ref_seq.size() < static_cast<size_t>(window_size)) {
    std::cerr << "Reference too short for window_size=" << window_size << "\n";
    return;
  }
  std::vector<std::shared_ptr<BioSequence>> index_seqs;
  for (int start = 0; start + window_size <= static_cast<int>(ref_seq.size()); start += stride) {
    std::string frag = ref_seq.substr(static_cast<size_t>(start), static_cast<size_t>(window_size));
    auto seq = std::make_shared<BioSequence>("ref_" + std::to_string(start), frag);
    seq->add_occurrence(ref_id, start, start + window_size, "+");
    index_seqs.push_back(seq);
  }
  std::cerr << "Index: " << index_seqs.size() << " windows from reference\n";

  BioGeometryIndexBuilder builder;
  builder.build(index_seqs);
  BioGeometrySearchEngine engine(builder);

  auto queries = load_reads(query_input, ref_id);
  if (queries.empty()) {
    std::cerr << "No query reads loaded.\n";
    return;
  }
  std::cerr << "Queries: " << queries.size() << "\n";

  std::vector<std::string> columns = {
      "query_id", "hit_id", "distance", "ref_positions", "read_id", "read_len",
      "ref_id", "strand", "query_start", "reference_start", "aligned_length",
      "score", "edit_distance", "query_fragment", "reference_fragment",
      "dist_calcs", "leaf_verify_count", "candidate_count_for_prune", "beacon_prune_count"};
  std::vector<std::vector<std::string>> all_rows;

  for (const auto& read : queries) {
    auto [res, st] = engine.search_adaptive(*read, tolerance);
    if (res.empty()) {
      all_rows.push_back({
          read->id, "", "", "", read->id, std::to_string(static_cast<int>(read->seq.size())),
          "", "+", "0", "0", "0", "0", "", read->seq, "",
          std::to_string(st.dist_calc_count), std::to_string(st.leaf_verify_count),
          std::to_string(st.candidate_count_for_prune), std::to_string(st.beacon_prune_count)});
    } else {
      for (const auto& hit : res) {
        int ed = compute_distance(read->seq, hit->seq);
        auto rows = search_results_to_tsv_rows(read->id, read->seq, 0, *hit, ed);
        for (const auto& r : rows) {
          all_rows.push_back({
              r.query_id, r.hit_id, r.distance_str, r.ref_positions_json,
              r.read_id, r.read_len, r.ref_id, r.strand, r.query_start, r.reference_start,
              r.aligned_length, r.score, r.edit_distance, r.query_fragment, r.reference_fragment,
              std::to_string(st.dist_calc_count), std::to_string(st.leaf_verify_count),
              std::to_string(st.candidate_count_for_prune), std::to_string(st.beacon_prune_count)});
        }
      }
    }
  }
  if (!out_tsv.empty())
    write_tsv(out_tsv, columns, all_rows);
  std::cerr << "Benchmark rows: " << all_rows.size() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }
  std::string cmd = argv[1];
  std::string ref_input, reads_input, query_seq, mode = "adaptive", out_tsv;
  int tolerance = 2;
  int demo_size = 500;
  int window_size = 200;
  int stride = 1;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ref" && i + 1 < argc) { ref_input = argv[++i]; continue; }
    if (a == "--reads" && i + 1 < argc) { reads_input = argv[++i]; continue; }
    if (a == "--query" && i + 1 < argc) { query_seq = argv[++i]; continue; }
    if (a == "--tolerance" && i + 1 < argc) { tolerance = std::atoi(argv[++i]); continue; }
    if (a == "--mode" && i + 1 < argc) { mode = argv[++i]; continue; }
    if (a == "--out" && i + 1 < argc) { out_tsv = argv[++i]; continue; }
    if (a == "--size" && i + 1 < argc) { demo_size = std::atoi(argv[++i]); continue; }
    if (a == "--window" && i + 1 < argc) { window_size = std::atoi(argv[++i]); continue; }
    if (a == "--stride" && i + 1 < argc) { stride = std::atoi(argv[++i]); continue; }
  }

  try {
    if (cmd == "demo") {
      run_demo(demo_size);
      return 0;
    }
    if (cmd == "build") {
      if (ref_input.empty() || reads_input.empty()) {
        std::cerr << "build requires --ref and --reads\n";
        return 1;
      }
      run_build(ref_input, reads_input);
      return 0;
    }
    if (cmd == "query") {
      if (reads_input.empty() || query_seq.empty()) {
        std::cerr << "query requires --reads and --query\n";
        return 1;
      }
      run_query(ref_input.empty() ? "ref" : ref_input, reads_input, query_seq, tolerance, mode);
      return 0;
    }
    if (cmd == "run") {
      if (ref_input.empty() || reads_input.empty()) {
        std::cerr << "run requires --ref and --reads\n";
        return 1;
      }
      run_full(ref_input, reads_input, tolerance, out_tsv);
      return 0;
    }
    if (cmd == "benchmark") {
      if (ref_input.empty() || reads_input.empty()) {
        std::cerr << "benchmark requires --ref and --reads\n";
        return 1;
      }
      run_benchmark(ref_input, reads_input, tolerance, window_size, stride, out_tsv);
      return 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  std::cerr << "Unknown command: " << cmd << "\n";
  usage(argv[0]);
  return 1;
}
