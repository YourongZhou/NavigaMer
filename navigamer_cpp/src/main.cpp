/**
 * NavigaMer v8 C++ reference CLI.
 * Usage:
 *   navigamer build --ref <fasta|sequence> --reads <fastq|sequence>
 *   navigamer query --reads <fastq|sequence> --query <sequence> [--tolerance 2] [--mode adaptive|greedy|exhaustive]
 *   navigamer demo  [--size 500]
 *   navigamer boundary --ref <fasta> [--length 250] [--error-rates ...] [--tolerance-rates ...]
 */

#include "structure.hpp"
#include "index_builder.hpp"
#include "search_engine.hpp"
#include "io_utils.hpp"
#include "tools.hpp"
#include "experiment_utils.hpp"
#include "map150.hpp"
#include "query_benchmark.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <chrono>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <omp.h>

namespace {

void usage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " demo [--size N] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " build --ref <path|seq> --reads <path|seq> [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " query --ref <path|seq> --reads <path|seq> --query <seq> [--tolerance 2] [--mode adaptive] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " run  --ref <path|seq> --reads <path|seq> [--tolerance 2] [--out <tsv>] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " map150 --ref <path|seq> --reads <path|seq> --tolerance <N> --out <tsv> [--mode adaptive] [--locator refpos|seqan] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " benchmark --ref <fasta> --reads <fastq> [--tolerance 5] [--window 200] [--stride 1] [--out <tsv>] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " query-benchmark --ref <fasta|sequence> --out <detail.tsv> --summary-out <summary.tsv> --json-out <summary.json> [--window 200] [--query-length 200] [--tolerance 2]\n"
            << "  " << prog << " boundary --ref <fasta> [--length 250] [--error-rates csv] [--tolerance-rates csv] [--queries-per-cell 200] [--stride-mode sparse|dense] [--seed 42] [--out <tsv>] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " layer-radius-experiment --ref <fasta> [--length 250] [--tolerance 2] [--query-edits N] [--queries-per-cell 200] [--stride N | --stride-mode sparse|dense] [--seed 42] [--L-values csv] [--r-leaf-values csv] [--alpha-values csv] [--out <csv>]\n"
            << "Global build flags: [--link-mode full|indexed] [--leaf-attach-mode full|indexed] [--range-candidate-mode auto|pigeonhole|qgram|hybrid|full] [--qgram-q 5] [--auto-pigeonhole-max-candidates 4096] [--auto-pigeonhole-max-ratio 0.25] [--auto-hybrid-on-large-candidates true] [--range-min-seed-length 8] [--range-max-seed-length 20] [--min-rect-index-fanout 64]\n"
            << "Global adaptive-search flags: [--mbb-filter-mode scan|rect] [--visited-mode string|epoch] [--graph-view original|flat] [--simd-mode auto|scalar|avx2|avx512] [--search-qgram-prefilter off|on] [--search-qgram-q 5]\n";
}

std::string format_double(double value) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(6) << value;
  return os.str();
}

std::string csv_escape(const std::string& value) {
  if (value.find_first_of(",\"\n") == std::string::npos) return value;
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '"') escaped += "\"\"";
    else escaped += c;
  }
  escaped += "\"";
  return escaped;
}

std::vector<double> parse_rate_csv(const std::string& csv) {
  std::vector<double> values;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) continue;
    values.push_back(std::stod(token));
  }
  if (values.empty()) throw std::runtime_error("rate list must not be empty");
  return values;
}

std::vector<int> parse_int_csv(const std::string& csv) {
  std::vector<int> values;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) continue;
    values.push_back(std::stoi(token));
  }
  if (values.empty()) throw std::runtime_error("primary radii list must not be empty");
  return values;
}

size_t parse_positive_size(const std::string& value, const std::string& flag) {
  if (value.empty() || value.front() == '-') {
    throw std::runtime_error(flag + " must be a positive integer");
  }
  size_t parsed_chars = 0;
  unsigned long long parsed = std::stoull(value, &parsed_chars);
  if (parsed_chars != value.size() || parsed == 0) {
    throw std::runtime_error(flag + " must be a positive integer");
  }
  return static_cast<size_t>(parsed);
}

size_t parse_nonnegative_size(const std::string& value, const std::string& flag) {
  if (value.empty() || value.front() == '-') {
    throw std::runtime_error(flag + " must be a non-negative integer");
  }
  size_t parsed_chars = 0;
  unsigned long long parsed = std::stoull(value, &parsed_chars);
  if (parsed_chars != value.size()) {
    throw std::runtime_error(flag + " must be a non-negative integer");
  }
  return static_cast<size_t>(parsed);
}

bool parse_bool(const std::string& value, const std::string& flag) {
  if (value == "true") return true;
  if (value == "false") return false;
  throw std::runtime_error(flag + " must be true or false");
}

bool parse_on_off(const std::string& value, const std::string& flag) {
  if (value == "on") return true;
  if (value == "off") return false;
  throw std::runtime_error(flag + " must be off or on");
}

void write_csv(const std::string& output_path,
               const std::vector<std::string>& columns,
               const std::vector<std::vector<std::string>>& rows) {
  std::ofstream out(output_path);
  for (size_t i = 0; i < columns.size(); ++i) {
    if (i) out << ',';
    out << csv_escape(columns[i]);
  }
  out << '\n';
  for (const auto& row : rows) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i) out << ',';
      out << csv_escape(row[i]);
    }
    out << '\n';
  }
}

std::string format_primary_radii(const navigamer::HierarchyConfig& config) {
  std::ostringstream os;
  for (size_t i = 0; i < config.primary_radii.size(); ++i) {
    if (i) os << ",";
    os << config.primary_radii[i];
  }
  return os.str();
}

std::string mutate_with_exact_substitutions(const std::string& seq, int edit_count,
                                            std::mt19937& gen) {
  if (seq.empty() || edit_count <= 0) return seq;
  std::string out = seq;
  int edits = std::min(edit_count, static_cast<int>(seq.size()));
  std::vector<size_t> positions(seq.size());
  std::iota(positions.begin(), positions.end(), 0);
  std::shuffle(positions.begin(), positions.end(), gen);
  static const char bases[] = "ATCG";
  std::uniform_int_distribution<> base_dis(0, 3);
  for (int i = 0; i < edits; ++i) {
    size_t pos = positions[static_cast<size_t>(i)];
    char orig = out[pos];
    char repl = orig;
    while (repl == orig) repl = bases[base_dis(gen)];
    out[pos] = repl;
  }
  return out;
}

std::vector<std::shared_ptr<navigamer::BioSequence>> build_reference_windows(
    const std::string& ref_id, const std::string& ref_seq, int window_size, int stride) {
  using namespace navigamer;
  std::vector<std::shared_ptr<BioSequence>> windows;
  for (int start = 0; start + window_size <= static_cast<int>(ref_seq.size()); start += stride) {
    std::string frag = ref_seq.substr(static_cast<size_t>(start), static_cast<size_t>(window_size));
    auto seq = std::make_shared<BioSequence>("ref_" + std::to_string(start), frag);
    seq->add_occurrence(ref_id, start, start + window_size, "+");
    windows.push_back(seq);
  }
  return windows;
}

struct BoundaryQuery {
  std::string source_id;
  navigamer::BioSequence query;
};

std::vector<BoundaryQuery> generate_boundary_queries(
    const std::vector<std::shared_ptr<navigamer::BioSequence>>& index_seqs,
    size_t query_count, int edit_count, unsigned seed) {
  std::vector<BoundaryQuery> queries;
  if (index_seqs.empty() || query_count == 0) return queries;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<size_t> pick(0, index_seqs.size() - 1);
  queries.reserve(query_count);
  for (size_t i = 0; i < query_count; ++i) {
    const auto& source = index_seqs[pick(gen)];
    queries.push_back({
        source->id,
        navigamer::BioSequence(
            "query_" + std::to_string(i),
            mutate_with_exact_substitutions(source->seq, edit_count, gen))});
  }
  return queries;
}

struct BoundaryCellStats {
  size_t query_count = 0;
  size_t source_recovery_count = 0;
  size_t any_hit_count = 0;
  size_t total_hit_count = 0;
  size_t total_dist_calcs = 0;
  size_t total_leaf_verify_count = 0;
  size_t total_candidate_count_for_prune = 0;
  size_t total_beacon_prune_count = 0;
  size_t bf_sample_count = 0;
  size_t bf_source_recovery_count = 0;
  size_t bf_agreement_count = 0;
  size_t bf_source_mismatch_count = 0;
};

// Generate a reproducible random DNA reference.
std::string generate_reference(size_t length, unsigned seed) {
  static const char bases[] = "ATCG";
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(0, 3);
  std::string s;
  s.reserve(length);
  for (size_t i = 0; i < length; ++i) s += bases[dis(gen)];
  return s;
}

// Generate reads from a reference with simple substitution mutations.
std::vector<std::shared_ptr<navigamer::BioSequence>> generate_reads(
    const std::string& ref, size_t num_reads, size_t read_len, double mutation_rate, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<size_t> pos_dis(0, ref.size() > read_len ? ref.size() - read_len : 0);
  std::vector<std::shared_ptr<navigamer::BioSequence>> reads;
  for (size_t i = 0; i < num_reads; ++i) {
    size_t start = pos_dis(gen);
    std::string fragment = ref.substr(start, read_len);
    if (fragment.size() < read_len) continue;
    // Apply independent substitution mutations to the sampled read.
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

void run_demo(int size, const navigamer::HierarchyConfig& config,
              const navigamer::BuildRangeConfig& range_config,
              const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
  std::cerr << "NavigaMer v8 (C++) - Demo with " << size << " reads"
            << " (primary_radii=" << format_primary_radii(config) << ")\n";
  std::string ref = generate_reference(50000, 42);
  auto reads = generate_reads(ref, size, 20, 0.0, 42);

  BioGeometryIndexBuilder builder(config, range_config);
  builder.build(reads);

  auto stats = builder.get_statistics();
  std::cout << "Index primary layers=" << builder.num_primary_layers()
            << " finest_nodes=" << builder.primary_layer(builder.finest_primary_layer_index()).size()
            << " compression=" << (stats.compression_ratio * 100) << "%\n";

  BioGeometrySearchEngine engine(builder, search_config);
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

void run_build(const std::string& ref_input, const std::string& reads_input,
               const navigamer::HierarchyConfig& config,
               const navigamer::BuildRangeConfig& range_config) {
  using namespace navigamer;
  auto [ref_id, ref_seq] = load_reference(ref_input);
  std::cerr << "Reference: " << ref_id << " length=" << ref_seq.size() << "\n";
  auto reads = load_reads(reads_input, ref_id);
  std::cerr << "Reads: " << reads.size() << "\n";

  BioGeometryIndexBuilder builder(config, range_config);
  builder.build(reads);
  std::cerr << "Build done. (Index serialization not implemented; use run for full pipeline.)\n";
}

void run_query(const std::string& /*ref_input*/, const std::string& reads_input,
               const std::string& query_seq, int tolerance, const std::string& mode,
               const navigamer::HierarchyConfig& config,
               const navigamer::BuildRangeConfig& range_config,
               const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
  auto reads = load_reads(reads_input, "ref");
  if (reads.empty()) {
    std::cerr << "No reads loaded.\n";
    return;
  }
  BioGeometryIndexBuilder builder(config, range_config);
  builder.build(reads);

  BioGeometrySearchEngine engine(builder, search_config);
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
              << " prune_rate=" << st.pruning_rate()
              << " mbb_filter_mode=" << mbb_filter_mode_name(search_config.mbb_filter_mode)
              << " mbb_scan_child_checks=" << st.mbb_scan_child_checks
              << " mbb_rect_index_queries=" << st.mbb_rect_index_queries
              << " mbb_rect_candidate_children=" << st.mbb_rect_candidate_children
              << " mbb_rect_fallback_count=" << st.mbb_rect_fallback_count
              << " center_distance_calls_after_mbb="
              << st.center_distance_calls_after_mbb
              << " search_qgram_prefilter_enabled="
              << (st.search_qgram_prefilter_enabled ? "true" : "false")
              << " search_qgram_q=" << st.search_qgram_q
              << " search_qgram_signature_build_count="
              << st.search_qgram_signature_build_count
              << " search_qgram_signature_missing_count="
              << st.search_qgram_signature_missing_count
              << " search_qgram_checks=" << st.search_qgram_checks
              << " search_qgram_pruned_children="
              << st.search_qgram_pruned_children
              << " search_qgram_passed_children="
              << st.search_qgram_passed_children
              << " center_distance_calls_before_qgram="
              << st.center_distance_calls_before_qgram
              << " center_distance_calls_after_qgram="
              << st.center_distance_calls_after_qgram
              << " qgram_prune_ratio=" << st.qgram_prune_ratio()
              << " result_count=" << st.result_count << ")\n";
    for (const auto& h : res) std::cout << "  " << h->id << " dist=" << compute_distance(query_seq, h->seq) << "\n";
  }
}

void run_full(const std::string& ref_input, const std::string& reads_input,
              int tolerance, const std::string& out_tsv,
              const navigamer::HierarchyConfig& config,
              const navigamer::BuildRangeConfig& range_config,
              const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
  auto [ref_id, ref_seq] = load_reference(ref_input);
  auto reads = load_reads(reads_input, ref_id);
  if (reads.empty()) {
    std::cerr << "No reads.\n";
    return;
  }
  BioGeometryIndexBuilder builder(config, range_config);
  builder.build(reads);
  BioGeometrySearchEngine engine(builder, search_config);

  std::vector<std::string> columns = {
      "query_id", "hit_id", "distance", "ref_positions", "read_id", "read_len",
      "ref_id", "strand", "query_start", "reference_start", "aligned_length",
      "score", "edit_distance", "query_fragment", "reference_fragment",
      "bwt_start", "bwt_end"};

  std::vector<std::vector<std::vector<std::string>>> per_read_rows(reads.size());

  #pragma omp parallel for schedule(dynamic)
  for (size_t ri = 0; ri < reads.size(); ++ri) {
    const auto& read = reads[ri];
    auto [res, st] = engine.search_adaptive(*read, tolerance);
    for (const auto& hit : res) {
      int ed = compute_distance(read->seq, hit->seq);
      auto rows = search_results_to_tsv_rows(read->id, read->seq, 0, *hit, ed);
      for (const auto& r : rows) {
        per_read_rows[ri].push_back({
            r.query_id, r.hit_id, r.distance_str, r.ref_positions_json,
            r.read_id, r.read_len, r.ref_id, r.strand, r.query_start, r.reference_start,
            r.aligned_length, r.score, r.edit_distance, r.query_fragment, r.reference_fragment,
            r.bwt_start, r.bwt_end});
      }
    }
  }

  std::vector<std::vector<std::string>> all_rows;
  for (auto& rows : per_read_rows)
    for (auto& row : rows)
      all_rows.push_back(std::move(row));

  if (!out_tsv.empty())
    write_tsv(out_tsv, columns, all_rows);
  std::cerr << "Total rows: " << all_rows.size() << "\n";
}

// Benchmark: reference windows -> index; query reads -> search; output hits + SearchStats
void run_benchmark(const std::string& ref_input, const std::string& query_input,
                   int tolerance, int window_size, int stride,
                   const std::string& out_tsv,
                   const navigamer::HierarchyConfig& config,
                   const navigamer::BuildRangeConfig& range_config,
                   const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
  auto [ref_id, ref_seq] = load_reference(ref_input);
  if (ref_seq.size() < static_cast<size_t>(window_size)) {
    std::cerr << "Reference too short for window_size=" << window_size << "\n";
    return;
  }
  std::vector<std::shared_ptr<BioSequence>> index_seqs =
      build_reference_windows(ref_id, ref_seq, window_size, stride);
  std::cerr << "Index: " << index_seqs.size() << " windows from reference\n";

  BioGeometryIndexBuilder builder(config, range_config);
  builder.build(index_seqs);
  BioGeometrySearchEngine engine(builder, search_config);

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
      "bwt_start", "bwt_end",
      "dist_calcs", "leaf_verify_count", "candidate_count_for_prune", "beacon_prune_count",
      "mbb_filter_mode", "mbb_scan_child_checks", "mbb_rect_index_queries",
      "mbb_rect_candidate_children", "mbb_rect_fallback_count",
      "mbb_surviving_child_count", "mbb_scalar_checks",
      "mbb_simd_batches", "mbb_simd_fallbacks",
      "center_distance_calls_after_mbb",
      "leaf_beacon_scalar_checks", "leaf_beacon_simd_batches",
      "leaf_beacon_simd_fallbacks",
      "search_qgram_prefilter_enabled", "search_qgram_q",
      "search_qgram_signature_build_count", "search_qgram_signature_missing_count",
      "search_qgram_checks", "search_qgram_pruned_children",
      "search_qgram_passed_children", "center_distance_calls_before_qgram",
      "center_distance_calls_after_qgram", "qgram_prune_ratio", "result_count",
      "avg_mbb_candidates_per_parent",
      "avg_center_distance_calls_per_query", "query_time_ms"};

  std::vector<std::vector<std::vector<std::string>>> per_query_rows(queries.size());

  #pragma omp parallel for schedule(dynamic)
  for (size_t qi = 0; qi < queries.size(); ++qi) {
    const auto& read = queries[qi];
    auto query_start = std::chrono::high_resolution_clock::now();
    auto [res, st] = engine.search_adaptive(*read, tolerance);
    auto query_end = std::chrono::high_resolution_clock::now();
    double query_time_ms =
        std::chrono::duration<double, std::milli>(query_end - query_start).count();
    double avg_mbb_candidates =
        st.mbb_filter_parent_count == 0
            ? 0.0
            : static_cast<double>(st.mbb_surviving_child_count) /
                  static_cast<double>(st.mbb_filter_parent_count);
    std::vector<std::string> search_stats = {
        mbb_filter_mode_name(search_config.mbb_filter_mode),
        std::to_string(st.mbb_scan_child_checks),
        std::to_string(st.mbb_rect_index_queries),
        std::to_string(st.mbb_rect_candidate_children),
        std::to_string(st.mbb_rect_fallback_count),
        std::to_string(st.mbb_surviving_child_count),
        std::to_string(st.mbb_scalar_checks),
        std::to_string(st.mbb_simd_batches),
        std::to_string(st.mbb_simd_fallbacks),
        std::to_string(st.center_distance_calls_after_mbb),
        std::to_string(st.leaf_beacon_scalar_checks),
        std::to_string(st.leaf_beacon_simd_batches),
        std::to_string(st.leaf_beacon_simd_fallbacks),
        st.search_qgram_prefilter_enabled ? "true" : "false",
        std::to_string(st.search_qgram_q),
        std::to_string(st.search_qgram_signature_build_count),
        std::to_string(st.search_qgram_signature_missing_count),
        std::to_string(st.search_qgram_checks),
        std::to_string(st.search_qgram_pruned_children),
        std::to_string(st.search_qgram_passed_children),
        std::to_string(st.center_distance_calls_before_qgram),
        std::to_string(st.center_distance_calls_after_qgram),
        format_double(st.qgram_prune_ratio()),
        std::to_string(st.result_count),
        format_double(avg_mbb_candidates),
        format_double(static_cast<double>(st.center_distance_calls_after_qgram)),
        format_double(query_time_ms)};
    if (res.empty()) {
      std::vector<std::string> row = {
          read->id, "", "", "", read->id, std::to_string(static_cast<int>(read->seq.size())),
          "", "+", "0", "0", "0", "0", "", read->seq, "",
          "-1", "-1",
          std::to_string(st.dist_calc_count), std::to_string(st.leaf_verify_count),
          std::to_string(st.candidate_count_for_prune),
          std::to_string(st.beacon_prune_count)};
      row.insert(row.end(), search_stats.begin(), search_stats.end());
      per_query_rows[qi].push_back(std::move(row));
    } else {
      for (const auto& hit : res) {
        int ed = compute_distance(read->seq, hit->seq);
        auto rows = search_results_to_tsv_rows(read->id, read->seq, 0, *hit, ed);
        for (const auto& r : rows) {
          std::vector<std::string> row = {
              r.query_id, r.hit_id, r.distance_str, r.ref_positions_json,
              r.read_id, r.read_len, r.ref_id, r.strand, r.query_start, r.reference_start,
              r.aligned_length, r.score, r.edit_distance, r.query_fragment, r.reference_fragment,
              r.bwt_start, r.bwt_end,
              std::to_string(st.dist_calc_count), std::to_string(st.leaf_verify_count),
              std::to_string(st.candidate_count_for_prune),
              std::to_string(st.beacon_prune_count)};
          row.insert(row.end(), search_stats.begin(), search_stats.end());
          per_query_rows[qi].push_back(std::move(row));
        }
      }
    }
  }

  std::vector<std::vector<std::string>> all_rows;
  for (auto& rows : per_query_rows)
    for (auto& row : rows)
      all_rows.push_back(std::move(row));

  if (!out_tsv.empty())
    write_tsv(out_tsv, columns, all_rows);
  std::cerr << "Benchmark rows: " << all_rows.size() << "\n";
}

void write_tsv_with_header_even_if_empty(
    const std::string& output_path,
    const std::vector<std::string>& columns,
    const std::vector<std::vector<std::string>>& rows) {
  if (!rows.empty()) {
    navigamer::write_tsv(output_path, columns, rows);
    return;
  }
  std::ofstream out(output_path);
  for (size_t i = 0; i < columns.size(); ++i) {
    if (i) out << '\t';
    out << columns[i];
  }
  out << '\n';
}

void run_map150(const std::string& ref_input,
                const std::string& reads_input,
                int tolerance,
                const std::string& mode,
                const std::string& locator_kind,
                const std::string& out_tsv,
                const navigamer::HierarchyConfig& config,
                const navigamer::BuildRangeConfig& range_config,
                const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
  auto [ref_id, ref_seq] = load_reference(ref_input);
  auto reads = load_reads(reads_input, ref_id);
  if (reads.empty()) {
    throw std::runtime_error("map150 loaded no reads");
  }

  auto index_seqs = build_map150_reference_windows(ref_id, ref_seq);
  std::cerr << "map150: windows=" << index_seqs.size()
            << " reads=" << reads.size()
            << " tolerance=" << tolerance
            << " candidate_tolerance=" << map150_candidate_tolerance(tolerance)
            << " locator=" << locator_kind << "\n";

  BioGeometryIndexBuilder builder(config, range_config);
  builder.build(index_seqs);

  std::unique_ptr<OccurrenceLocator> locator;
  if (locator_kind == "refpos") {
    locator = std::make_unique<RefPositionLocator>();
  } else if (locator_kind == "seqan") {
    locator = make_seqan_fm_locator(ref_id, normalize_acgt_sequence(ref_seq, "reference"), builder);
  } else {
    throw std::runtime_error("map150 --locator must be refpos or seqan");
  }

  auto results = map150_reads_with_locator(
      ref_id, ref_seq, reads, tolerance, mode, config, *locator, builder,
      search_config);
  auto rows = map150_results_to_tsv_rows(results);
  if (!out_tsv.empty()) {
    write_tsv_with_header_even_if_empty(out_tsv, map150_tsv_columns(), rows);
  }
  std::cerr << "map150 rows: " << rows.size() << "\n";
}

void run_boundary(const std::string& ref_input, int length,
                  const std::vector<double>& error_rates,
                  const std::vector<double>& tolerance_rates,
                  size_t queries_per_cell, const std::string& stride_mode,
                  unsigned seed, const std::string& out_tsv,
                  const navigamer::HierarchyConfig& config,
                  const navigamer::BuildRangeConfig& range_config,
                  const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
  if (length != 250) {
    throw std::runtime_error("boundary currently only supports --length 250");
  }
  if (stride_mode != "sparse" && stride_mode != "dense") {
    throw std::runtime_error("boundary --stride-mode must be sparse or dense");
  }

  auto [ref_id, ref_seq] = load_reference(ref_input);
  if (ref_seq.size() < static_cast<size_t>(length)) {
    throw std::runtime_error("reference too short for boundary window length");
  }

  int stride_for_mode = (stride_mode == "dense") ? 62 : length;
  auto index_seqs = build_reference_windows(ref_id, ref_seq, length, stride_for_mode);
  if (index_seqs.empty()) {
    throw std::runtime_error("boundary could not generate reference windows");
  }

  std::cerr << "Boundary: stride_mode=" << stride_mode
            << " length=" << length
            << " stride=" << stride_for_mode
            << " windows=" << index_seqs.size() << "\n";

  BioGeometryIndexBuilder builder(config, range_config);
  builder.build(index_seqs);
  BioGeometrySearchEngine engine(builder, search_config);

  std::vector<std::shared_ptr<BioSequence>> unique_list;
  unique_list.reserve(builder.unique_sequences.size());
  for (const auto& p : builder.unique_sequences) unique_list.push_back(p.second);

  std::vector<std::string> columns = {
      "length", "stride_mode", "num_index_seqs",
      "error_rate", "error_edits", "tolerance_rate", "tolerance_edits",
      "query_count",
      "source_recovery_rate", "any_hit_rate", "avg_hit_count",
      "avg_dist_calcs", "avg_leaf_verify_count",
      "avg_candidate_count_for_prune", "avg_beacon_prune_count", "avg_pruning_rate",
      "bf_sample_count", "bf_source_recovery_rate", "bf_agreement_rate", "bf_source_mismatch_count"};
  std::vector<std::vector<std::string>> rows;
  rows.reserve(error_rates.size() * tolerance_rates.size());

  for (size_t ei = 0; ei < error_rates.size(); ++ei) {
    double error_rate = error_rates[ei];
    int error_edits = static_cast<int>(std::llround(error_rate * static_cast<double>(length)));
    auto queries = generate_boundary_queries(
        index_seqs, queries_per_cell, error_edits,
        seed + static_cast<unsigned>(ei * 7919));

    for (double tolerance_rate : tolerance_rates) {
      int tolerance_edits =
          static_cast<int>(std::llround(tolerance_rate * static_cast<double>(length)));
      BoundaryCellStats cell;
      cell.query_count = queries.size();
      size_t bf_limit = std::min<size_t>(50, queries.size());

      for (size_t qi = 0; qi < queries.size(); ++qi) {
        const auto& q = queries[qi];
        auto [res, st] = engine.search_adaptive(q.query, tolerance_edits);

        bool source_found = false;
        for (const auto& hit : res) {
          if (hit->id == q.source_id) {
            source_found = true;
            break;
          }
        }

        if (source_found) cell.source_recovery_count++;
        if (!res.empty()) cell.any_hit_count++;
        cell.total_hit_count += res.size();
        cell.total_dist_calcs += st.dist_calc_count;
        cell.total_leaf_verify_count += st.leaf_verify_count;
        cell.total_candidate_count_for_prune += st.candidate_count_for_prune;
        cell.total_beacon_prune_count += st.beacon_prune_count;

        if (qi < bf_limit) {
          auto [bf_res, bf_st] = engine.search_brute_force(q.query, tolerance_edits, unique_list);
          bool bf_source_found = false;
          for (const auto& hit : bf_res) {
            if (hit->id == q.source_id) {
              bf_source_found = true;
              break;
            }
          }
          cell.bf_sample_count++;
          if (bf_source_found) cell.bf_source_recovery_count++;
          if (bf_source_found == source_found) {
            cell.bf_agreement_count++;
          } else {
            cell.bf_source_mismatch_count++;
          }
          (void)bf_st;
        }
      }

      double query_den = cell.query_count == 0 ? 1.0 : static_cast<double>(cell.query_count);
      double prune_den = cell.total_candidate_count_for_prune == 0
                             ? 1.0
                             : static_cast<double>(cell.total_candidate_count_for_prune);
      double bf_den = cell.bf_sample_count == 0 ? 1.0 : static_cast<double>(cell.bf_sample_count);

      rows.push_back({
          std::to_string(length),
          stride_mode,
          std::to_string(index_seqs.size()),
          format_double(error_rate),
          std::to_string(error_edits),
          format_double(tolerance_rate),
          std::to_string(tolerance_edits),
          std::to_string(cell.query_count),
          format_double(static_cast<double>(cell.source_recovery_count) / query_den),
          format_double(static_cast<double>(cell.any_hit_count) / query_den),
          format_double(static_cast<double>(cell.total_hit_count) / query_den),
          format_double(static_cast<double>(cell.total_dist_calcs) / query_den),
          format_double(static_cast<double>(cell.total_leaf_verify_count) / query_den),
          format_double(static_cast<double>(cell.total_candidate_count_for_prune) / query_den),
          format_double(static_cast<double>(cell.total_beacon_prune_count) / query_den),
          format_double(static_cast<double>(cell.total_beacon_prune_count) / prune_den),
          std::to_string(cell.bf_sample_count),
          format_double(static_cast<double>(cell.bf_source_recovery_count) / bf_den),
          format_double(static_cast<double>(cell.bf_agreement_count) / bf_den),
          std::to_string(cell.bf_source_mismatch_count)});

      if (cell.bf_source_mismatch_count > 0) {
        std::cerr << "Boundary warning: stride_mode=" << stride_mode
                  << " error_rate=" << format_double(error_rate)
                  << " tolerance_rate=" << format_double(tolerance_rate)
                  << " bf_source_mismatch_count=" << cell.bf_source_mismatch_count << "\n";
      }
    }
  }

  if (!out_tsv.empty()) write_tsv(out_tsv, columns, rows);
  std::cerr << "Boundary rows: " << rows.size() << "\n";
}

void run_layer_radius_experiment(const std::string& ref_input,
                                 int length,
                                 int tolerance,
                                 int query_edits,
                                 const std::vector<int>& layer_values,
                                 const std::vector<int>& r_leaf_values,
                                 const std::vector<double>& alpha_values,
                                 size_t queries_per_cell,
                                 int stride_override,
                                 const std::string& stride_mode,
                                 unsigned seed,
                                 const std::string& out_csv,
                                 const navigamer::BuildRangeConfig& range_config,
                                 const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
  if (length <= 0) throw std::runtime_error("experiment --length must be positive");
  if (tolerance < 0) throw std::runtime_error("experiment --tolerance must be non-negative");
  if (query_edits < 0) throw std::runtime_error("experiment --query-edits must be non-negative");
  if (stride_override <= 0 && stride_mode != "sparse" && stride_mode != "dense") {
    throw std::runtime_error("experiment --stride-mode must be sparse or dense");
  }

  auto [ref_id, ref_seq] = load_reference(ref_input);
  if (ref_seq.size() < static_cast<size_t>(length)) {
    throw std::runtime_error("reference too short for experiment window length");
  }

  int stride_for_mode = stride_override > 0
                            ? stride_override
                            : ((stride_mode == "dense") ? std::max(1, length / 4) : length);
  auto index_seqs = build_reference_windows(ref_id, ref_seq, length, stride_for_mode);
  if (index_seqs.empty()) {
    throw std::runtime_error("experiment could not generate reference windows");
  }

  auto queries = generate_boundary_queries(index_seqs, queries_per_cell, query_edits, seed);
  std::cerr << "Layer-radius experiment: windows=" << index_seqs.size()
            << " queries=" << queries.size()
            << " stride=" << stride_for_mode
            << " stride_mode=" << stride_mode
            << " length=" << length
            << " query_edits=" << query_edits << "\n";

  std::vector<std::string> columns = {
      "dataset", "query_id", "query_length", "L", "r_leaf", "alpha",
      "radius_schedule", "query_time_ms",
      "world_access_count", "node_access_count", "edge_access_count",
      "anchor_distance_count", "bound_check_count",
      "candidate_count", "candidate_verify_count"};
  std::vector<std::vector<std::string>> rows;
  rows.reserve(layer_values.size() * r_leaf_values.size() * alpha_values.size() * queries.size());

  for (int L : layer_values) {
    for (int r_leaf : r_leaf_values) {
      for (double alpha : alpha_values) {
        auto radius_schedule = generate_geometric_radius_schedule(L, r_leaf, alpha);
        BioGeometryIndexBuilder builder(HierarchyConfig(radius_schedule), range_config);
        builder.build(index_seqs);
        BioGeometrySearchEngine engine(builder, search_config);
        std::string schedule_str = join_radius_schedule(radius_schedule);

        for (size_t query_idx = 0; query_idx < queries.size(); ++query_idx) {
          const auto& query = queries[query_idx].query;
          auto start = std::chrono::high_resolution_clock::now();
          auto [results, stats] = engine.search_adaptive(query, tolerance);
          auto end = std::chrono::high_resolution_clock::now();
          (void)results;
          double query_time_ms =
              std::chrono::duration<double, std::milli>(end - start).count();

          rows.push_back({
              ref_id,
              std::to_string(query_idx),
              std::to_string(static_cast<int>(query.seq.size())),
              std::to_string(L),
              std::to_string(r_leaf),
              format_double(alpha),
              schedule_str,
              format_double(query_time_ms),
              std::to_string(stats.world_access_count),
              std::to_string(stats.node_access_count),
              std::to_string(stats.edge_access_count),
              std::to_string(stats.anchor_distance_count),
              std::to_string(stats.bound_check_count),
              std::to_string(stats.candidate_count),
              std::to_string(stats.candidate_verify_count),
          });
        }
      }
    }
  }

  write_csv(out_csv.empty() ? "layer_radius_search_stats.csv" : out_csv, columns, rows);
  std::cerr << "Layer-radius experiment rows: " << rows.size() << "\n";
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
  bool stride_explicit = false;
  int boundary_length = 250;
  size_t queries_per_cell = 200;
  unsigned seed = 42;
  std::string stride_mode = "sparse";
  std::string locator_kind = "refpos";
  std::string error_rates_csv = "0,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20";
  std::string tolerance_rates_csv = "0,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20";
  std::string primary_radii_csv;
  std::string layer_values_csv = "2,3,4,5";
  std::string r_leaf_values_csv = "4,8,12";
  std::string alpha_values_csv = "0.5,0.7";
  std::string link_mode = "indexed";
  std::string leaf_attach_mode = "indexed";
  std::string range_candidate_mode = "auto";
  std::string mbb_filter_mode = "scan";
  std::string visited_mode = "epoch";
  std::string graph_view_mode = "flat";
  std::string simd_mode = "auto";
  std::string search_qgram_prefilter = "off";
  int range_min_seed_length = 8;
  int range_max_seed_length = 20;
  int qgram_q = 5;
  int search_qgram_q = 5;
  size_t auto_pigeonhole_max_candidates = 4096;
  double auto_pigeonhole_max_ratio = 0.25;
  bool auto_hybrid_on_large_candidates = true;
  size_t min_rect_index_fanout = 64;
  int r_sw = navigamer::R_SW;
  int r_mw = navigamer::R_MW;
  int r_lw = navigamer::R_LW;
  int query_edits = -1;
  size_t reference_subset_length = 0;
  int query_length = 200;
  int threads = 1;
  size_t queries_per_class = 1;
  size_t warmup_iterations = 2;
  size_t measured_iterations = 10;
  size_t cold_cache_bytes = 256ULL * 1024ULL * 1024ULL;
  std::string summary_out;
  std::string json_out;

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
    if (a == "--stride" && i + 1 < argc) {
      stride = std::atoi(argv[++i]);
      stride_explicit = true;
      continue;
    }
    if (a == "--length" && i + 1 < argc) { boundary_length = std::atoi(argv[++i]); continue; }
    if (a == "--queries-per-cell" && i + 1 < argc) {
      queries_per_cell = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
      continue;
    }
    if (a == "--query-edits" && i + 1 < argc) { query_edits = std::atoi(argv[++i]); continue; }
    if (a == "--reference-subset-length" && i + 1 < argc) {
      reference_subset_length =
          parse_nonnegative_size(argv[++i], "--reference-subset-length");
      continue;
    }
    if (a == "--query-length" && i + 1 < argc) {
      query_length = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--threads" && i + 1 < argc) {
      threads = static_cast<int>(parse_positive_size(argv[++i], "--threads"));
      continue;
    }
    if (a == "--queries-per-class" && i + 1 < argc) {
      queries_per_class =
          parse_positive_size(argv[++i], "--queries-per-class");
      continue;
    }
    if (a == "--warmup-iterations" && i + 1 < argc) {
      warmup_iterations =
          parse_nonnegative_size(argv[++i], "--warmup-iterations");
      continue;
    }
    if (a == "--measured-iterations" && i + 1 < argc) {
      measured_iterations =
          parse_positive_size(argv[++i], "--measured-iterations");
      continue;
    }
    if (a == "--cold-cache-bytes" && i + 1 < argc) {
      cold_cache_bytes =
          parse_nonnegative_size(argv[++i], "--cold-cache-bytes");
      continue;
    }
    if (a == "--summary-out" && i + 1 < argc) {
      summary_out = argv[++i];
      continue;
    }
    if (a == "--json-out" && i + 1 < argc) {
      json_out = argv[++i];
      continue;
    }
    if (a == "--seed" && i + 1 < argc) {
      seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
      continue;
    }
    if (a == "--stride-mode" && i + 1 < argc) { stride_mode = argv[++i]; continue; }
    if (a == "--locator" && i + 1 < argc) { locator_kind = argv[++i]; continue; }
    if (a == "--L-values" && i + 1 < argc) { layer_values_csv = argv[++i]; continue; }
    if (a == "--r-leaf-values" && i + 1 < argc) { r_leaf_values_csv = argv[++i]; continue; }
    if (a == "--alpha-values" && i + 1 < argc) { alpha_values_csv = argv[++i]; continue; }
    if (a == "--error-rates" && i + 1 < argc) { error_rates_csv = argv[++i]; continue; }
    if (a == "--tolerance-rates" && i + 1 < argc) { tolerance_rates_csv = argv[++i]; continue; }
    if (a == "--primary-radii" && i + 1 < argc) { primary_radii_csv = argv[++i]; continue; }
    if (a == "--r-sw" && i + 1 < argc) { r_sw = std::atoi(argv[++i]); continue; }
    if (a == "--r-mw" && i + 1 < argc) { r_mw = std::atoi(argv[++i]); continue; }
    if (a == "--r-lw" && i + 1 < argc) { r_lw = std::atoi(argv[++i]); continue; }
    if (a == "--link-mode" && i + 1 < argc) { link_mode = argv[++i]; continue; }
    if (a == "--leaf-attach-mode" && i + 1 < argc) {
      leaf_attach_mode = argv[++i];
      continue;
    }
    if (a == "--range-candidate-mode" && i + 1 < argc) {
      range_candidate_mode = argv[++i];
      continue;
    }
    if (a == "--qgram-q" && i + 1 < argc) {
      qgram_q = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--auto-pigeonhole-max-candidates" && i + 1 < argc) {
      auto_pigeonhole_max_candidates =
          parse_nonnegative_size(argv[++i], "--auto-pigeonhole-max-candidates");
      continue;
    }
    if (a == "--auto-pigeonhole-max-ratio" && i + 1 < argc) {
      auto_pigeonhole_max_ratio = std::stod(argv[++i]);
      continue;
    }
    if (a == "--auto-hybrid-on-large-candidates" && i + 1 < argc) {
      auto_hybrid_on_large_candidates =
          parse_bool(argv[++i], "--auto-hybrid-on-large-candidates");
      continue;
    }
    if (a == "--mbb-filter-mode" && i + 1 < argc) {
      mbb_filter_mode = argv[++i];
      continue;
    }
    if (a == "--visited-mode" && i + 1 < argc) {
      visited_mode = argv[++i];
      continue;
    }
    if (a == "--graph-view" && i + 1 < argc) {
      graph_view_mode = argv[++i];
      continue;
    }
    if (a == "--simd-mode" && i + 1 < argc) {
      simd_mode = argv[++i];
      continue;
    }
    if (a == "--search-qgram-prefilter" && i + 1 < argc) {
      search_qgram_prefilter = argv[++i];
      continue;
    }
    if (a == "--search-qgram-q" && i + 1 < argc) {
      search_qgram_q = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--min-rect-index-fanout" && i + 1 < argc) {
      min_rect_index_fanout =
          parse_positive_size(argv[++i], "--min-rect-index-fanout");
      continue;
    }
    if (a == "--range-min-seed-length" && i + 1 < argc) {
      range_min_seed_length = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--range-max-seed-length" && i + 1 < argc) {
      range_max_seed_length = std::atoi(argv[++i]);
      continue;
    }
  }

  try {
    navigamer::HierarchyConfig hierarchy =
        primary_radii_csv.empty()
            ? navigamer::HierarchyConfig({r_lw, r_mw, r_sw})
            : navigamer::HierarchyConfig(parse_int_csv(primary_radii_csv));
    navigamer::BuildRangeConfig range_config;
    range_config.link_mode = navigamer::parse_build_range_mode(link_mode);
    range_config.leaf_attach_mode =
        navigamer::parse_build_range_mode(leaf_attach_mode);
    range_config.range_join.min_seed_len = range_min_seed_length;
    range_config.range_join.max_seed_len = range_max_seed_length;
    range_config.range_join.qgram_q = qgram_q;
    range_config.range_join.candidate_mode =
        navigamer::parse_range_candidate_mode(range_candidate_mode);
    range_config.range_join.auto_pigeonhole_max_candidates =
        auto_pigeonhole_max_candidates;
    range_config.range_join.auto_pigeonhole_max_ratio =
        auto_pigeonhole_max_ratio;
    range_config.range_join.auto_hybrid_on_large_candidates =
        auto_hybrid_on_large_candidates;
    range_config.min_rect_index_fanout = min_rect_index_fanout;
    navigamer::SearchConfig search_config;
    search_config.mbb_filter_mode = navigamer::parse_mbb_filter_mode(mbb_filter_mode);
    search_config.visited_mode = navigamer::parse_visited_mode(visited_mode);
    search_config.graph_view_mode = navigamer::parse_graph_view_mode(graph_view_mode);
    search_config.simd_mode = navigamer::parse_simd_mode(simd_mode);
    search_config.search_qgram_prefilter =
        parse_on_off(search_qgram_prefilter, "--search-qgram-prefilter");
    search_config.search_qgram_q = search_qgram_q;

    if (cmd == "demo") {
      run_demo(demo_size, hierarchy, range_config, search_config);
      return 0;
    }
    if (cmd == "build") {
      if (ref_input.empty() || reads_input.empty()) {
        std::cerr << "build requires --ref and --reads\n";
        return 1;
      }
      run_build(ref_input, reads_input, hierarchy, range_config);
      return 0;
    }
    if (cmd == "query") {
      if (reads_input.empty() || query_seq.empty()) {
        std::cerr << "query requires --reads and --query\n";
        return 1;
      }
      run_query(ref_input.empty() ? "ref" : ref_input, reads_input, query_seq,
                tolerance, mode, hierarchy, range_config, search_config);
      return 0;
    }
    if (cmd == "run") {
      if (ref_input.empty() || reads_input.empty()) {
        std::cerr << "run requires --ref and --reads\n";
        return 1;
      }
      run_full(ref_input, reads_input, tolerance, out_tsv, hierarchy, range_config,
               search_config);
      return 0;
    }
    if (cmd == "map150") {
      if (ref_input.empty() || reads_input.empty() || out_tsv.empty()) {
        std::cerr << "map150 requires --ref, --reads, and --out\n";
        return 1;
      }
      run_map150(ref_input, reads_input, tolerance, mode, locator_kind, out_tsv,
                 hierarchy, range_config, search_config);
      return 0;
    }
    if (cmd == "benchmark") {
      if (ref_input.empty() || reads_input.empty()) {
        std::cerr << "benchmark requires --ref and --reads\n";
        return 1;
      }
      run_benchmark(ref_input, reads_input, tolerance, window_size, stride, out_tsv,
                    hierarchy, range_config, search_config);
      return 0;
    }
    if (cmd == "query-benchmark") {
      if (ref_input.empty() || out_tsv.empty() || summary_out.empty() ||
          json_out.empty()) {
        std::cerr << "query-benchmark requires --ref, --out, --summary-out, "
                     "and --json-out\n";
        return 1;
      }
      navigamer::QueryBenchmarkConfig config;
      config.ref_input = ref_input;
      config.reference_subset_length = reference_subset_length;
      config.window_length = window_size;
      config.stride = stride;
      config.query_length = query_length;
      config.tolerance = tolerance;
      config.seed = seed;
      config.threads = threads;
      config.queries_per_class = queries_per_class;
      config.warmup_iterations = warmup_iterations;
      config.measured_iterations = measured_iterations;
      config.cold_cache_bytes = cold_cache_bytes;
      config.detail_tsv_path = out_tsv;
      config.summary_tsv_path = summary_out;
      config.json_path = json_out;
      auto benchmark_result =
          navigamer::run_query_benchmark(config, hierarchy, range_config,
                                         search_config);
      if (!benchmark_result.gate_passed) {
        std::cerr << "query-benchmark gate failed: mismatches="
                  << benchmark_result.mismatch_count << "\n";
        return 2;
      }
      std::cerr << "query-benchmark gate passed\n";
      return 0;
    }
    if (cmd == "boundary") {
      if (ref_input.empty()) {
        std::cerr << "boundary requires --ref\n";
        return 1;
      }
      auto error_rates = parse_rate_csv(error_rates_csv);
      auto tolerance_rates = parse_rate_csv(tolerance_rates_csv);
      run_boundary(ref_input, boundary_length, error_rates, tolerance_rates,
                   queries_per_cell, stride_mode, seed, out_tsv, hierarchy,
                   range_config, search_config);
      return 0;
    }
    if (cmd == "layer-radius-experiment") {
      if (ref_input.empty()) {
        std::cerr << "layer-radius-experiment requires --ref\n";
        return 1;
      }
      auto L_values = parse_int_csv(layer_values_csv);
      auto r_leaf_values = parse_int_csv(r_leaf_values_csv);
      auto alpha_values = parse_rate_csv(alpha_values_csv);
      run_layer_radius_experiment(ref_input, boundary_length, tolerance,
                                  query_edits >= 0 ? query_edits : tolerance,
                                  L_values, r_leaf_values, alpha_values,
                                  queries_per_cell,
                                  stride_explicit ? stride : -1,
                                  stride_mode, seed, out_tsv, range_config,
                                  search_config);
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
