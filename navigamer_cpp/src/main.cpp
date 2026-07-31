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
#include "index_persistence.hpp"
#include "candidate_verifier.hpp"
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
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <filesystem>
#include <omp.h>

namespace {

void usage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " demo [--size N] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " build --ref <path|seq> --reads <path|seq> [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " build-scale --ref <path|seq> --window 250 --stride 1 --prefix-lengths csv --out <csv> [--index <file>] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " query --ref <path|seq> --reads <path|seq> --query <seq> [--index <file>] [--tolerance 2] [--mode adaptive] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " query-index --index <file> --query <seq> [--tolerance 2] [--mode adaptive]\n"
            << "  " << prog << " query-index-batch --index <file> --reads <fastq> [--tolerance 2] [--out <tsv>] [--path-trace-out <tsv>] [--mode adaptive]\n"
            << "  " << prog << " run  --ref <path|seq> --reads <path|seq> [--tolerance 2] [--out <tsv>] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " map150 --ref <path|seq> --reads <path|seq> --tolerance <N> --out <tsv> [--mode adaptive] [--locator refpos|seqan] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " benchmark --ref <fasta> --reads <fastq> [--tolerance 5] [--window 200] [--stride 1] [--out <tsv>] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
	            << "  " << prog << " query-benchmark --ref <fasta|sequence> --out <detail.tsv> --summary-out <summary.tsv> --json-out <summary.json> [--window 200] [--query-length 200] [--tolerance 2] [--query-benchmark-ablations 0|1] [--proximal-oracle 0|1] [--proximal-oracle-k 1,2,4]\n"
	            << "  " << prog << " candidate-verify --ref <fasta|sequence> --reads <fastq> --candidates <candidate.tsv> --out <detail.tsv> --summary-out <summary.tsv> [--window 150] [--stride 1] [--tolerance 5] [--truth source|exhaustive]\n"
	            << "  " << prog << " locality-benchmark --index <navidx> --ref <fasta|sequence> --out <summary.tsv> [--query-fastq-out <queries.fq>] [--query-count 256] [--query-length 250] [--query-edits 5] [--tolerance 5] [--scenarios low-fanout,high-fanout,repeat,batch-locality,oracle,all] [--locality-profiles baseline,path_reuse,optimized] [--locality-datasets same_template,nearby_windows,random_windows] [--batch-schedules original,random,minimizer,qgram-signature,router-signature,source-oracle]\n"
            << "  " << prog << " query-locality-benchmark --index <navidx> --ref <fasta|sequence> --out <summary.tsv> [same flags as locality-benchmark]\n"
            << "  " << prog << " query-locality-report --ref <fasta|sequence> --out-dir <dir> [--index <navidx>] [--window 250] [--stride 1] [--query-count 256] [--query-length 250] [--query-edits 5] [--tolerance 5] [--scenarios low-fanout,high-fanout,repeat,batch-locality,oracle,all] [--locality-profiles baseline,path_reuse,optimized] [--locality-datasets same_template,nearby_windows,random_windows] [--batch-schedules original,random,minimizer,qgram-signature,router-signature,source-oracle]\n"
            << "  " << prog << " boundary --ref <fasta> [--length 250] [--error-rates csv] [--tolerance-rates csv] [--queries-per-cell 200] [--stride-mode sparse|dense] [--seed 42] [--out <tsv>] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " layer-radius-experiment --ref <fasta> [--length 250] [--tolerance 2] [--query-edits N] [--queries-per-cell 200] [--stride N | --stride-mode sparse|dense] [--seed 42] [--L-values csv] [--r-leaf-values csv] [--alpha-values csv] [--out <csv>]\n"
            << "Global build flags: [--link-mode full|indexed] [--leaf-attach-mode full|indexed] [--leaf-attach-direction auto|seq-to-world|world-to-seq] [--phase2-qgram-postfilter on|off] [--leaf-qgram-postfilter on|off] [--range-candidate-mode auto|pigeonhole|qgram|hybrid|full] [--qgram-q 5] [--auto-pigeonhole-max-candidates 4096] [--auto-pigeonhole-max-ratio 0.25] [--auto-hybrid-on-large-candidates true] [--range-min-seed-length 8] [--range-max-seed-length 20] [--min-rect-index-fanout 64] [--phase1-metric-min-fanout 12] [--phase1-qgram-min-fanout 12] [--phase1-qgram-max-touched 250000] [--progress-interval-seconds 600]\n"
            << "Global adaptive-search flags: [--mbb-filter-mode scan|rect] [--visited-mode string|epoch] [--graph-view original|flat] [--simd-mode auto|scalar|avx2|avx512] [--distance-mode dp|myers|edlib|auto] [--build-distance-mode dp|edlib|auto] [--search-prefetch off|on] [--search-qgram-prefilter off|on] [--search-qgram-q 5] [--query-profile 0|1] [--path-reuse 0|1] [--router-hints 0|1] [--router-hint-qgram-q N] [--router-hint-minimizer-k N] [--router-hint-minimizer-w N] [--local-router 0|1] [--local-router-max-anchors N] [--local-router-max-children N] [--local-router-score anchor-envelope] [--best-first 0|1] [--safe-child-router 0|1] [--safe-child-router-min-fanout N] [--safe-child-router-max-candidates N] [--safe-child-router-max-ratio R] [--safe-child-router-min-seed-len N] [--safe-child-router-mode auto|pigeonhole|qgram|mbb|full-fallback] [--safe-child-router-validate 0|1] [--query-planner 0|1] [--planner-direct-verify-max-candidates N] [--planner-router-min-fanout N] [--planner-safe-child-router-min-fanout N] [--planner-allow-direct-qgram-verify 0|1] [--proximal-oracle 0|1] [--proximal-oracle-k csv]\n";
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

size_t parse_positive_size(const std::string& value, const std::string& flag);

std::vector<size_t> parse_size_csv(const std::string& csv,
                                   const std::string& flag) {
  std::vector<size_t> values;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) continue;
    values.push_back(parse_positive_size(token, flag));
  }
  if (values.empty()) throw std::runtime_error(flag + " must not be empty");
  return values;
}

std::vector<std::string> parse_string_csv(const std::string& csv,
                                          const std::string& flag) {
  std::vector<std::string> values;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) throw std::runtime_error(flag + " contains an empty item");
    values.push_back(item);
  }
  if (values.empty()) throw std::runtime_error(flag + " must not be empty");
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

bool parse_zero_one(const std::string& value, const std::string& flag) {
  if (value == "0") return false;
  if (value == "1") return true;
  throw std::runtime_error(flag + " must be 0 or 1");
}

double average_values(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
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

template <typename Id>
std::string join_id_path(const std::vector<Id>& path) {
  std::ostringstream os;
  for (size_t i = 0; i < path.size(); ++i) {
    if (i) os << ',';
    os << path[i];
  }
  return os.str();
}

template <typename Id>
size_t unique_count(const std::vector<Id>& values) {
  return std::unordered_set<Id>(values.begin(), values.end()).size();
}

template <typename Id>
std::pair<std::string, std::string> previous_overlap_summary(
    const std::vector<Id>& current,
    const std::vector<Id>& previous,
    bool has_previous) {
  if (!has_previous) return {"NA", "NA"};
  std::unordered_set<Id> current_set(current.begin(), current.end());
  std::unordered_set<Id> previous_set(previous.begin(), previous.end());
  size_t overlap = 0;
  for (const auto& id : current_set) {
    if (previous_set.count(id)) overlap++;
  }
  const size_t union_size = current_set.size() + previous_set.size() - overlap;
  if (union_size == 0) return {"NA", std::to_string(overlap)};
  return {format_double(static_cast<double>(overlap) /
                        static_cast<double>(union_size)),
          std::to_string(overlap)};
}

std::string format_primary_radii(const navigamer::HierarchyConfig& config) {
  std::ostringstream os;
  for (size_t i = 0; i < config.primary_radii.size(); ++i) {
    if (i) os << ",";
    os << config.primary_radii[i];
  }
  return os.str();
}

std::vector<std::pair<std::string, double>> sorted_descending(
    std::vector<std::pair<std::string, double>> values) {
  std::sort(values.begin(), values.end(),
            [](const auto& left, const auto& right) {
              return left.second > right.second;
            });
  return values;
}

std::string build_query_batch_key(const std::string& sequence) {
  constexpr size_t kScheduleQ = 4;
  if (sequence.size() < kScheduleQ) return "seq:" + sequence;
  std::vector<std::string> qgrams;
  qgrams.reserve(sequence.size() - kScheduleQ + 1);
  for (size_t i = 0; i + kScheduleQ <= sequence.size(); ++i) {
    qgrams.push_back(sequence.substr(i, kScheduleQ));
  }
  std::sort(qgrams.begin(), qgrams.end());
  qgrams.erase(std::unique(qgrams.begin(), qgrams.end()), qgrams.end());
  std::ostringstream os;
  os << "len:" << sequence.size();
  const size_t limit = std::min<size_t>(8, qgrams.size());
  for (size_t i = 0; i < limit; ++i) {
    os << "|" << qgrams[i];
  }
  return os.str();
}

std::vector<size_t> build_scheduled_query_indices(
    const std::vector<std::shared_ptr<navigamer::BioSequence>>& queries,
    bool path_reuse_enabled) {
  std::vector<size_t> indices(queries.size());
  std::iota(indices.begin(), indices.end(), 0);
  if (!path_reuse_enabled) return indices;
  const bool has_source_positions = std::any_of(
      queries.begin(), queries.end(),
      [](const std::shared_ptr<navigamer::BioSequence>& query) {
        return query && query->has_source_pos;
      });
  if (has_source_positions) {
    std::stable_sort(indices.begin(), indices.end(),
                     [&queries](size_t left, size_t right) {
                       const auto& left_query = queries[left];
                       const auto& right_query = queries[right];
                       const bool left_has = left_query && left_query->has_source_pos;
                       const bool right_has = right_query && right_query->has_source_pos;
                       if (left_has && right_has) {
                         if (left_query->source_pos != right_query->source_pos) {
                           return left_query->source_pos < right_query->source_pos;
                         }
                         return left < right;
                       }
                       if (left_has != right_has) return left_has;
                       return build_query_batch_key(left_query ? left_query->seq : "") <
                              build_query_batch_key(right_query ? right_query->seq : "");
                     });
    return indices;
  }
  std::stable_sort(indices.begin(), indices.end(),
                   [&queries](size_t left, size_t right) {
                     return build_query_batch_key(queries[left]->seq) <
                            build_query_batch_key(queries[right]->seq);
                   });
  return indices;
}

void print_top_entries(const std::string& title,
                       std::vector<std::pair<std::string, double>> values,
                       double total_ms) {
  values = sorted_descending(std::move(values));
  std::cerr << title << ":\n";
  const size_t limit = std::min<size_t>(3, values.size());
  for (size_t i = 0; i < limit; ++i) {
    const double pct = total_ms <= 0.0 ? 0.0 : values[i].second * 100.0 / total_ms;
    std::cerr << "  " << (i + 1) << ". " << values[i].first << ": "
              << format_double(values[i].second) << " ms ("
              << format_double(pct) << "%)\n";
  }
}

std::string leaf_attach_direction_used(
    const navigamer::BioGeometryIndexBuilder::Statistics& stats) {
  return navigamer::leaf_attach_direction_name(stats.leaf_attach_direction_used);
}

void print_build_scale_bottleneck_summary(
    size_t prefix_len,
    const navigamer::BioGeometryIndexBuilder::Statistics& stats) {
  std::cerr << "Build-scale prefix=" << prefix_len
            << " total_build_ms=" << format_double(stats.total_build_ms) << "\n";
  print_top_entries(
      "Top build phases",
      {
          {"phase0_dedup", stats.phase0_dedup_ms},
          {"phase1_sketch", stats.phase1_sketch_ms},
          {"phase2_rebinding", stats.phase2_rebinding_ms},
          {"phase3_mbb", stats.phase3_mbb_ms},
          {"phase4_attach", stats.phase4_attach_ms},
          {"assign_ids", stats.assign_ids_ms},
          {"graph_view", stats.graph_view_ms},
      },
      stats.total_build_ms);
  print_top_entries(
      "Top substeps",
      {
          {"phase2_index_build", stats.phase2_index_build_ms},
          {"phase2_candidate_query", stats.phase2_candidate_query_ms},
          {"phase2_exact_verify", stats.phase2_exact_verify_ms},
          {"phase2_candidate_query_worker",
           stats.phase2_candidate_query_worker_ms},
          {"phase2_exact_verify_worker", stats.phase2_exact_verify_worker_ms},
          {"phase2_edge_insert", stats.phase2_edge_insert_ms},
          {"phase3_collect_beacons", stats.phase3_collect_beacons_ms},
          {"phase3_collapse_children", stats.phase3_collapse_children_ms},
          {"phase3_child_mbb_distance", stats.phase3_child_mbb_distance_ms},
          {"phase3_rect_index_build", stats.phase3_rect_index_build_ms},
          {"leaf_index_build", stats.leaf_index_build_ms},
          {"leaf_candidate_query", stats.leaf_candidate_query_ms},
          {"leaf_exact_verify", stats.leaf_exact_verify_ms},
          {"leaf_tuple_emit", stats.leaf_tuple_emit_ms},
          {"leaf_tuple_merge_sort", stats.leaf_tuple_merge_sort_ms},
          {"leaf_populate", stats.leaf_populate_ms},
          {"leaf_beacon_distance", stats.leaf_beacon_distance_ms},
          {"range_posting_lookup", stats.range_posting_lookup_ms},
          {"range_seed_union", stats.range_seed_union_ms},
          {"range_length_filter", stats.range_length_filter_ms},
          {"range_qgram_query", stats.range_qgram_query_ms},
          {"range_hybrid_intersection", stats.range_hybrid_intersection_ms},
          {"range_full_scan", stats.range_full_scan_ms},
      },
      stats.total_build_ms);
  std::cerr << "  phase2_candidate_reduction="
            << format_double(stats.phase2_candidate_reduction_ratio * 100.0)
            << "% phase2_exact_reduction="
            << format_double(stats.phase2_exact_distance_reduction_ratio * 100.0)
            << "% leaf_candidate_reduction="
            << format_double(stats.leaf_candidate_reduction_ratio * 100.0)
            << "% leaf_exact_reduction="
            << format_double(stats.leaf_exact_distance_reduction_ratio * 100.0)
            << "%\n";
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
  std::string source_ref_id;
  int source_start = -1;
  int source_end = -1;
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
    const auto* occ = source->ref_positions.empty() ? nullptr
                                                    : &source->ref_positions.front();
    queries.push_back({
        source->id,
        occ ? occ->ref_id : "",
        occ ? occ->start : -1,
        occ ? occ->end : -1,
        navigamer::BioSequence(
            "query_" + std::to_string(i),
            mutate_with_exact_substitutions(source->seq, edit_count, gen))});
  }
  return queries;
}

bool recovers_source_locus(
    const navigamer::BioSequence* hit,
    const BoundaryQuery& query) {
  if (!hit) return false;
  if (hit->id == query.source_id) return true;
  for (const auto& occ : hit->ref_positions) {
    if (occ.ref_id == query.source_ref_id &&
        occ.start == query.source_start &&
        occ.end == query.source_end) {
      return true;
    }
  }
  return false;
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
            << " finest_nodes="
            << builder.primary_layer_size(builder.finest_primary_layer_index())
            << " compression=" << (stats.compression_ratio * 100) << "%\n";

  BioGeometrySearchEngine engine(builder, search_config);

  int tolerance = 2;
  size_t adaptive_ok = 0, exhaustive_ok = 0, bf_ok = 0;
  for (size_t i = 0; i < std::min(size_t(50), reads.size()); ++i) {
    auto [adaptive_res, st_adapt] = engine.search_adaptive(*reads[i], tolerance);
    auto [exhaustive_res, st_ex] = engine.search_exhaustive(*reads[i], tolerance);
    auto [bf_res, st_bf] = engine.search_brute_force(*reads[i], tolerance);
    if (!bf_res.empty()) bf_ok++;
    bool a_ok = false, e_ok = false;
    for (const auto& h : bf_res) {
      if (std::find_if(adaptive_res.begin(), adaptive_res.end(),
                       [&h](const BioSequence* x) { return x->id == h->id; }) != adaptive_res.end())
        a_ok = true;
      if (std::find_if(exhaustive_res.begin(), exhaustive_res.end(),
                       [&h](const BioSequence* x) { return x->id == h->id; }) != exhaustive_res.end())
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
               const navigamer::BuildRangeConfig& range_config,
               const std::string& index_path) {
  using namespace navigamer;
  auto [ref_id, ref_seq] = load_reference(ref_input);
  std::cerr << "Reference: " << ref_id << " length=" << ref_seq.size() << "\n";
  auto reads = load_reads(reads_input, ref_id);
  std::cerr << "Reads: " << reads.size() << "\n";

  BioGeometryIndexBuilder builder(config, range_config);
  builder.build(reads);
  if (!index_path.empty()) {
    IndexBuildManifest manifest =
        make_index_manifest(ref_input, reads_input, config, range_config);
    save_index(index_path, builder, manifest);
    IndexBuildManifest stored = read_index_manifest(index_path);
    std::cerr << "Index saved: " << index_path
              << " signature=" << stored.signature
              << " sequences=" << stored.sequence_count
              << " world_nodes=" << stored.world_node_count
              << " edges=" << stored.edge_count
              << " leaf_links=" << stored.leaf_link_count << "\n";
  } else {
    std::cerr << "Build done. Use --index <file> to persist the index.\n";
  }
}

void run_build_scale(const std::string& ref_input,
                     int window_size,
                     int stride,
                     const std::vector<size_t>& prefix_lengths,
                     const std::string& out_csv,
                     const navigamer::HierarchyConfig& config,
                     const navigamer::BuildRangeConfig& range_config,
                     const std::string& leaf_attach_direction,
                     const std::string& index_path) {
  using namespace navigamer;
  if (window_size <= 0) throw std::runtime_error("build-scale --window must be positive");
  if (stride <= 0) throw std::runtime_error("build-scale --stride must be positive");
  if (out_csv.empty()) throw std::runtime_error("build-scale requires --out");
  if (!index_path.empty() && prefix_lengths.size() != 1) {
    throw std::runtime_error(
        "build-scale --index requires exactly one prefix length");
  }
  if (leaf_attach_direction != "auto" &&
      leaf_attach_direction != "seq-to-world" &&
      leaf_attach_direction != "seq_to_world" &&
      leaf_attach_direction != "world-to-seq" &&
      leaf_attach_direction != "world_to_seq") {
    throw std::runtime_error(
        "build-scale --leaf-attach-direction must be auto, seq-to-world, or world-to-seq");
  }

  auto [ref_id, ref_seq] = load_reference(ref_input);
  if (ref_seq.empty()) throw std::runtime_error("build-scale reference is empty");

  const std::vector<std::string> columns = {
      "prefix_len",
      "window_count",
      "unique_count",
      "world_node_count",
      "finest_world_count",
      "total_build_ms",
      "phase0_dedup_ms",
      "phase1_sketch_ms",
      "phase2_rebinding_ms",
      "phase2_index_build_ms",
      "phase2_candidate_query_ms",
      "phase2_exact_verify_ms",
      "phase2_distance_batches",
      "phase2_edge_insert_ms",
      "phase3_mbb_ms",
      "phase3_collect_beacons_ms",
      "phase3_collapse_children_ms",
      "phase3_child_mbb_distance_ms",
      "phase3_rect_index_build_ms",
      "phase4_attach_ms",
      "leaf_index_build_ms",
      "leaf_candidate_query_ms",
      "leaf_exact_verify_ms",
      "leaf_tuple_emit_ms",
      "leaf_tuple_merge_sort_ms",
      "leaf_populate_ms",
      "leaf_beacon_distance_ms",
      "assign_ids_ms",
      "graph_view_ms",
      "phase2_total_possible_pairs",
      "phase2_candidate_pairs",
      "phase2_exact_distance_calls",
      "phase2_edges_added",
      "leaf_total_possible_pairs",
      "leaf_candidate_pairs",
      "leaf_exact_distance_calls",
      "leaf_attachments_added",
      "phase2_full_scan_fallback_count",
      "leaf_full_scan_fallback_count",
      "phase2_seed_candidate_pairs_before_length_filter",
      "phase2_seed_length_pruned_candidates",
      "phase2_pigeonhole_early_abort_count",
      "phase2_range_final_candidate_pairs",
      "leaf_seed_candidate_pairs_before_length_filter",
      "leaf_seed_length_pruned_candidates",
      "leaf_pigeonhole_early_abort_count",
      "leaf_range_final_candidate_pairs",
      "leaf_attach_direction_used",
      "range_candidate_mode",
      "qgram_q",
      "phase2_candidate_query_worker_ms",
      "phase2_exact_verify_worker_ms",
  };

  std::ofstream out(out_csv);
  if (!out) throw std::runtime_error("could not open build-scale output CSV");
  auto write_row = [&](const std::vector<std::string>& row) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i) out << ',';
      out << csv_escape(row[i]);
    }
    out << '\n';
    out.flush();
  };
  write_row(columns);
  size_t rows_written = 0;

  for (size_t requested_prefix : prefix_lengths) {
    const size_t actual_prefix = std::min(requested_prefix, ref_seq.size());
    const std::string prefix_seq = ref_seq.substr(0, actual_prefix);
    auto windows = build_reference_windows(ref_id, prefix_seq, window_size, stride);
    std::cerr << "build-scale: prefix_len=" << actual_prefix
              << " requested=" << requested_prefix
              << " windows=" << windows.size()
              << " window=" << window_size
              << " stride=" << stride << "\n";

    BioGeometryIndexBuilder builder(config, range_config);
    builder.build(windows);
    if (!index_path.empty()) {
      IndexBuildManifest manifest = make_reference_window_index_manifest(
          ref_input, actual_prefix, window_size, stride, config, range_config);
      std::cerr << "Saving index: " << index_path << "\n";
      save_index(index_path, builder, manifest);
      const IndexBuildManifest stored = read_index_manifest(index_path);
      std::cerr << "Index saved: " << index_path
                << " signature=" << stored.signature
                << " sequences=" << stored.sequence_count
                << " world_nodes=" << stored.world_node_count
                << " edges=" << stored.edge_count
                << " leaf_links=" << stored.leaf_link_count << "\n";
    }
    const auto stats = builder.get_statistics();
    const size_t finest_count =
        builder.primary_layer_size(builder.finest_primary_layer_index());

    write_row({
        std::to_string(actual_prefix),
        std::to_string(windows.size()),
        std::to_string(stats.unique_sequences),
        std::to_string(builder.num_world_nodes()),
        std::to_string(finest_count),
        format_double(stats.total_build_ms),
        format_double(stats.phase0_dedup_ms),
        format_double(stats.phase1_sketch_ms),
        format_double(stats.phase2_rebinding_ms),
        format_double(stats.phase2_index_build_ms),
        format_double(stats.phase2_candidate_query_ms),
        format_double(stats.phase2_exact_verify_ms),
        std::to_string(stats.phase2_distance_batches),
        format_double(stats.phase2_edge_insert_ms),
        format_double(stats.phase3_mbb_ms),
        format_double(stats.phase3_collect_beacons_ms),
        format_double(stats.phase3_collapse_children_ms),
        format_double(stats.phase3_child_mbb_distance_ms),
        format_double(stats.phase3_rect_index_build_ms),
        format_double(stats.phase4_attach_ms),
        format_double(stats.leaf_index_build_ms),
        format_double(stats.leaf_candidate_query_ms),
        format_double(stats.leaf_exact_verify_ms),
        format_double(stats.leaf_tuple_emit_ms),
        format_double(stats.leaf_tuple_merge_sort_ms),
        format_double(stats.leaf_populate_ms),
        format_double(stats.leaf_beacon_distance_ms),
        format_double(stats.assign_ids_ms),
        format_double(stats.graph_view_ms),
        std::to_string(stats.phase2_total_possible_pairs),
        std::to_string(stats.phase2_candidate_pairs),
        std::to_string(stats.phase2_exact_distance_calls),
        std::to_string(stats.phase2_edges_added),
        std::to_string(stats.total_possible_leaf_pairs),
        std::to_string(stats.leaf_candidate_pairs),
        std::to_string(stats.leaf_exact_distance_calls),
        std::to_string(stats.leaf_attachments_added),
        std::to_string(stats.phase2_full_scan_fallback_count),
        std::to_string(stats.leaf_full_scan_fallback_count),
        std::to_string(stats.phase2_seed_candidate_pairs_before_length_filter),
        std::to_string(stats.phase2_seed_length_pruned_candidates),
        std::to_string(stats.phase2_pigeonhole_early_abort_count),
        std::to_string(stats.phase2_range_final_candidate_pairs),
        std::to_string(stats.leaf_seed_candidate_pairs_before_length_filter),
        std::to_string(stats.leaf_seed_length_pruned_candidates),
        std::to_string(stats.leaf_pigeonhole_early_abort_count),
        std::to_string(stats.leaf_range_final_candidate_pairs),
        leaf_attach_direction_used(stats),
        range_candidate_mode_name(range_config.range_join.candidate_mode),
        std::to_string(range_config.range_join.qgram_q),
        format_double(stats.phase2_candidate_query_worker_ms),
        format_double(stats.phase2_exact_verify_worker_ms),
    });
    rows_written++;

    print_build_scale_bottleneck_summary(actual_prefix, stats);
  }

  std::cerr << "build-scale rows: " << rows_written << "\n";
}

void run_query_on_builder(const navigamer::BioGeometryIndexBuilder& builder,
                          const std::string& query_seq, int tolerance,
                          const std::string& mode,
                          const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
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
              << " query_profile_enabled="
              << (st.query_profile_enabled ? "true" : "false")
              << " query_total_ms=" << format_double(st.query_total_ms)
              << " anchor_distance_ms=" << format_double(st.anchor_distance_ms)
              << " mbb_filter_ms=" << format_double(st.mbb_filter_ms)
              << " center_distance_ms=" << format_double(st.center_distance_ms)
              << " leaf_collect_ms=" << format_double(st.leaf_collect_ms)
              << " leaf_verify_ms=" << format_double(st.leaf_verify_ms)
              << " result_dedup_ms=" << format_double(st.result_dedup_ms)
              << " path_reuse_enabled="
              << (search_config.path_reuse_enabled ? "true" : "false")
              << " path_reuse_ms=" << format_double(st.path_reuse_ms)
              << " path_reuse_attempt_count=" << st.path_reuse_attempt_count
              << " path_reuse_hit_count=" << st.path_reuse_hit_count
              << " anchor_cache_hit_count=" << st.anchor_cache_hit_count
              << " child_shortlist_reuse_hit_count="
              << st.child_shortlist_reuse_hit_count
              << " router_hint_enabled="
              << (search_config.router_hint_enabled ? "true" : "false")
              << " router_hint_invoked_count="
              << st.router_hint_invoked_count
              << " router_qgram_ranked_count="
              << st.router_qgram_ranked_count
              << " router_minimizer_ranked_count="
              << st.router_minimizer_ranked_count
              << " router_pigeonhole_query_count="
              << st.router_pigeonhole_query_count
              << " router_candidate_count="
              << st.router_candidate_count
              << " router_candidate_hit_count="
              << st.router_candidate_hit_count
              << " router_fallback_count="
              << st.router_fallback_count
              << " local_router_enabled="
              << (search_config.local_router_enabled ? "true" : "false")
              << " local_router_score_mode="
              << search_config.local_router_score_mode
              << " local_router_invoked_count="
              << st.local_router_invoked_count
              << " local_router_empty_count="
              << st.local_router_empty_count
              << " local_router_shortlist_child_count="
              << st.local_router_shortlist_child_count
              << " local_router_remaining_child_count="
              << st.local_router_remaining_child_count
              << " local_router_fallback_count="
              << st.local_router_fallback_count
              << " best_first_enabled="
              << (search_config.best_first_enabled ? "true" : "false")
              << " best_first_invoked_count="
              << st.best_first_invoked_count
              << " best_first_reordered_count="
              << st.best_first_reordered_count
              << " best_first_bound_candidate_count="
              << st.best_first_bound_candidate_count
              << " child_safe_bound_pruned_count="
              << st.child_safe_bound_pruned_count
              << " safe_child_router_enabled="
              << (search_config.safe_child_router_enabled ? "true" : "false")
              << " safe_child_router_invoked_count="
              << st.safe_child_router_invoked_count
              << " safe_child_router_skipped_low_fanout_count="
              << st.safe_child_router_skipped_low_fanout_count
              << " safe_child_router_fallback_count="
              << st.safe_child_router_fallback_count
              << " safe_child_router_candidate_count="
              << st.safe_child_router_candidate_count
              << " safe_child_router_pruned_by_not_candidate_count="
              << st.safe_child_router_pruned_by_not_candidate_count
              << " safe_child_router_exact_verify_count="
              << st.safe_child_router_exact_verify_count
              << " child_count_before_router="
              << st.child_count_before_router
              << " post_mbb_survivor_count="
              << st.post_mbb_survivor_count
              << " safe_router_candidate_count="
              << st.safe_router_candidate_count
              << " candidate_ratio_to_all_children="
              << format_double(st.candidate_ratio_to_all_children)
              << " candidate_ratio_to_post_mbb_survivors="
              << format_double(st.candidate_ratio_to_post_mbb_survivors)
              << " children_actually_processed="
              << st.children_actually_processed
              << " center_checks_saved="
              << st.center_checks_saved
              << " mbb_filter_mode=" << mbb_filter_mode_name(search_config.mbb_filter_mode)
              << " mbb_scan_child_checks=" << st.mbb_scan_child_checks
              << " mbb_rect_index_queries=" << st.mbb_rect_index_queries
              << " mbb_rect_candidate_children=" << st.mbb_rect_candidate_children
              << " mbb_rect_fallback_count=" << st.mbb_rect_fallback_count
              << " center_distance_calls_after_mbb="
              << st.center_distance_calls_after_mbb
              << " frontier_max_size=" << st.frontier_max_size
              << " frontier_total_pushed=" << st.frontier_total_pushed
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
              << " world_access_count=" << st.world_access_count
              << " anchor_distance_count=" << st.anchor_distance_count
              << " center_distance_count=" << st.center_distance_count
              << " raw_candidate_count=" << st.raw_candidate_count
              << " result_count=" << st.result_count << ")\n";
    for (const auto& h : res) std::cout << "  " << h->id << " dist=" << compute_distance(query_seq, h->seq) << "\n";
  }
}

void run_query(const std::string& ref_input, const std::string& reads_input,
               const std::string& query_seq, int tolerance, const std::string& mode,
               const navigamer::HierarchyConfig& config,
               const navigamer::BuildRangeConfig& range_config,
               const navigamer::SearchConfig& search_config,
               const std::string& index_path) {
  using namespace navigamer;
  if (!index_path.empty() && reads_input.empty()) {
    LoadedIndex loaded = load_index(index_path);
    std::cerr << "Loaded index: " << index_path
              << " signature=" << loaded.manifest.signature
              << " sequences=" << loaded.manifest.sequence_count
              << " world_nodes=" << loaded.manifest.world_node_count << "\n";
    run_query_on_builder(loaded.builder, query_seq, tolerance, mode, search_config);
    return;
  }

  if (reads_input.empty()) {
    std::cerr << "No reads loaded.\n";
    return;
  }

  if (!index_path.empty()) {
    IndexBuildManifest stored;
    bool have_stored = false;
    try {
      stored = read_index_manifest(index_path);
      have_stored = true;
    } catch (...) {
      have_stored = false;
    }
    const std::string manifest_ref_input =
        !ref_input.empty() ? ref_input : (have_stored ? stored.ref_input : "ref");
    IndexBuildManifest expected =
        make_index_manifest(manifest_ref_input, reads_input, config, range_config);
    std::string reason;
    if (index_matches_manifest(index_path, expected, &stored, &reason)) {
      LoadedIndex loaded = load_index(index_path);
      std::cerr << "Reusing persisted index: " << index_path
                << " signature=" << loaded.manifest.signature
                << " sequences=" << loaded.manifest.sequence_count
                << " world_nodes=" << loaded.manifest.world_node_count << "\n";
      run_query_on_builder(loaded.builder, query_seq, tolerance, mode, search_config);
      return;
    }
    std::cerr << "Persisted index not reused: " << reason << "\n";

    auto reads = load_reads(reads_input, "ref");
    if (reads.empty()) {
      std::cerr << "No reads loaded.\n";
      return;
    }
    BioGeometryIndexBuilder builder(config, range_config);
    builder.build(reads);
    save_index(index_path, builder, expected);
    IndexBuildManifest written = read_index_manifest(index_path);
    std::cerr << "Rebuilt and saved index: " << index_path
              << " signature=" << written.signature
              << " sequences=" << written.sequence_count
              << " world_nodes=" << written.world_node_count << "\n";
    run_query_on_builder(builder, query_seq, tolerance, mode, search_config);
    return;
  }

  auto reads = load_reads(reads_input, "ref");
  if (reads.empty()) {
    std::cerr << "No reads loaded.\n";
    return;
  }
  BioGeometryIndexBuilder builder(config, range_config);
  builder.build(reads);
  run_query_on_builder(builder, query_seq, tolerance, mode, search_config);
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
  const auto scheduled_indices =
      build_scheduled_query_indices(reads, search_config.path_reuse_enabled);

  #pragma omp parallel for schedule(static)
  for (size_t pos = 0; pos < scheduled_indices.size(); ++pos) {
    const size_t ri = scheduled_indices[pos];
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
      "query_profile_enabled", "query_count", "query_total_ms",
      "router_lookup_ms", "anchor_distance_ms", "mbb_filter_ms",
      "child_bound_ms", "center_distance_ms", "best_first_queue_ms",
      "leaf_collect_ms", "leaf_mbb_filter_ms", "leaf_verify_ms",
      "result_dedup_ms", "path_reuse_ms",
      "path_reuse_attempt_count", "path_reuse_hit_count",
      "anchor_cache_hit_count", "child_shortlist_reuse_hit_count",
      "router_hint_enabled", "router_hint_qgram_q",
      "router_hint_minimizer_k", "router_hint_minimizer_w",
      "router_hint_invoked_count", "router_qgram_ranked_count",
      "router_minimizer_ranked_count", "router_pigeonhole_query_count",
      "router_candidate_count", "router_candidate_hit_count",
      "router_fallback_count",
      "local_router_enabled", "local_router_score_mode",
      "best_first_enabled",
      "local_router_enabled_count", "local_router_invoked_count",
      "local_router_empty_count", "local_router_shortlist_child_count",
      "local_router_remaining_child_count", "local_router_fallback_count",
      "best_first_invoked_count", "best_first_reordered_count",
      "best_first_bound_candidate_count", "child_safe_bound_pruned_count",
      "safe_child_router_enabled", "safe_child_router_invoked_count",
      "safe_child_router_skipped_low_fanout_count",
      "safe_child_router_fallback_count",
      "safe_child_router_candidate_count",
      "safe_child_router_pruned_by_not_candidate_count",
      "safe_child_router_exact_verify_count",
      "child_count_before_router", "post_mbb_survivor_count",
      "safe_router_candidate_count", "candidate_ratio_to_all_children",
      "candidate_ratio_to_post_mbb_survivors",
      "children_actually_processed", "center_checks_saved",
      "query_planner_enabled", "planner_invoked_count",
      "planner_strategy_baseline_count",
      "planner_strategy_direct_qgram_count",
      "planner_strategy_router_count",
      "planner_strategy_safe_child_router_count",
      "planner_strategy_path_reuse_count", "planner_fallback_count",
      "planner_decision_ms",
      "mbb_filter_mode", "mbb_scan_child_checks", "mbb_rect_index_queries",
      "mbb_rect_candidate_children", "mbb_rect_fallback_count",
      "mbb_surviving_child_count", "mbb_scalar_checks",
      "mbb_simd_batches", "mbb_simd_fallbacks",
      "center_distance_calls_after_mbb", "frontier_max_size",
      "frontier_total_pushed", "world_access_count",
      "anchor_distance_count", "center_distance_count", "raw_candidate_count",
      "leaf_beacon_scalar_checks", "leaf_beacon_simd_batches",
      "leaf_beacon_simd_fallbacks",
      "search_qgram_prefilter_enabled", "search_qgram_q",
      "search_qgram_signature_build_count", "search_qgram_signature_missing_count",
      "search_qgram_checks", "search_qgram_pruned_children",
	      "search_qgram_passed_children", "center_distance_calls_before_qgram",
	      "center_distance_calls_after_qgram", "qgram_prune_ratio", "result_count",
	      "avg_mbb_candidates_per_parent",
	      "avg_center_distance_calls_per_query", "query_time_ms",
	      "near_query_leaf_triangle_pruned_count",
	      "near_query_leaf_distance_reused_count",
	      "near_query_leaf_bound_fallback_count"};

  std::vector<std::vector<std::vector<std::string>>> per_query_rows(queries.size());
  std::vector<double> summary_query_ms;
  std::vector<double> summary_world_access;
  std::vector<double> summary_anchor_distance;
  std::vector<double> summary_center_distance;
  std::vector<double> summary_raw_candidates;
  std::vector<double> summary_result_count;
  std::vector<double> summary_local_router_invoked;
  std::vector<double> summary_local_router_shortlisted;
  std::vector<double> summary_router_hint_invoked;
  std::vector<double> summary_router_hint_hits;
  std::vector<double> summary_path_reuse_hits;
  std::vector<double> summary_anchor_cache_hits;
  const auto scheduled_indices =
      build_scheduled_query_indices(queries, search_config.path_reuse_enabled);

  #pragma omp parallel for schedule(static)
  for (size_t pos = 0; pos < scheduled_indices.size(); ++pos) {
    const size_t qi = scheduled_indices[pos];
    const auto& read = queries[qi];
    auto query_start = std::chrono::high_resolution_clock::now();
    auto [res, st] = engine.search_adaptive(*read, tolerance);
    auto query_end = std::chrono::high_resolution_clock::now();
    double query_time_ms =
        std::chrono::duration<double, std::milli>(query_end - query_start).count();
    double profiled_query_ms =
        st.query_total_ms > 0.0 ? st.query_total_ms : query_time_ms;
    double avg_mbb_candidates =
        st.mbb_filter_parent_count == 0
            ? 0.0
            : static_cast<double>(st.mbb_surviving_child_count) /
                  static_cast<double>(st.mbb_filter_parent_count);
    std::vector<std::string> search_stats = {
        st.query_profile_enabled ? "true" : "false",
        std::to_string(st.query_count),
        format_double(st.query_total_ms),
        format_double(st.router_lookup_ms),
        format_double(st.anchor_distance_ms),
        format_double(st.mbb_filter_ms),
        format_double(st.child_bound_ms),
        format_double(st.center_distance_ms),
        format_double(st.best_first_queue_ms),
        format_double(st.leaf_collect_ms),
        format_double(st.leaf_mbb_filter_ms),
        format_double(st.leaf_verify_ms),
        format_double(st.result_dedup_ms),
        format_double(st.path_reuse_ms),
        std::to_string(st.path_reuse_attempt_count),
        std::to_string(st.path_reuse_hit_count),
        std::to_string(st.anchor_cache_hit_count),
        std::to_string(st.child_shortlist_reuse_hit_count),
        search_config.router_hint_enabled ? "true" : "false",
        std::to_string(search_config.router_hint_qgram_q),
        std::to_string(search_config.router_hint_minimizer_k),
        std::to_string(search_config.router_hint_minimizer_w),
        std::to_string(st.router_hint_invoked_count),
        std::to_string(st.router_qgram_ranked_count),
        std::to_string(st.router_minimizer_ranked_count),
        std::to_string(st.router_pigeonhole_query_count),
        std::to_string(st.router_candidate_count),
        std::to_string(st.router_candidate_hit_count),
        std::to_string(st.router_fallback_count),
        search_config.local_router_enabled ? "true" : "false",
        search_config.local_router_score_mode,
        search_config.best_first_enabled ? "true" : "false",
        std::to_string(st.local_router_enabled_count),
        std::to_string(st.local_router_invoked_count),
        std::to_string(st.local_router_empty_count),
        std::to_string(st.local_router_shortlist_child_count),
        std::to_string(st.local_router_remaining_child_count),
        std::to_string(st.local_router_fallback_count),
        std::to_string(st.best_first_invoked_count),
        std::to_string(st.best_first_reordered_count),
        std::to_string(st.best_first_bound_candidate_count),
        std::to_string(st.child_safe_bound_pruned_count),
        search_config.safe_child_router_enabled ? "true" : "false",
        std::to_string(st.safe_child_router_invoked_count),
        std::to_string(st.safe_child_router_skipped_low_fanout_count),
        std::to_string(st.safe_child_router_fallback_count),
        std::to_string(st.safe_child_router_candidate_count),
        std::to_string(st.safe_child_router_pruned_by_not_candidate_count),
        std::to_string(st.safe_child_router_exact_verify_count),
        std::to_string(st.child_count_before_router),
        std::to_string(st.post_mbb_survivor_count),
        std::to_string(st.safe_router_candidate_count),
        format_double(st.candidate_ratio_to_all_children),
        format_double(st.candidate_ratio_to_post_mbb_survivors),
        std::to_string(st.children_actually_processed),
        std::to_string(st.center_checks_saved),
        search_config.query_planner_enabled ? "true" : "false",
        std::to_string(st.planner_invoked_count),
        std::to_string(st.planner_strategy_baseline_count),
        std::to_string(st.planner_strategy_direct_qgram_count),
        std::to_string(st.planner_strategy_router_count),
        std::to_string(st.planner_strategy_safe_child_router_count),
        std::to_string(st.planner_strategy_path_reuse_count),
        std::to_string(st.planner_fallback_count),
        format_double(st.planner_decision_ms),
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
        std::to_string(st.frontier_max_size),
        std::to_string(st.frontier_total_pushed),
        std::to_string(st.world_access_count),
        std::to_string(st.anchor_distance_count),
        std::to_string(st.center_distance_count),
        std::to_string(st.raw_candidate_count),
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
	        format_double(profiled_query_ms),
	        std::to_string(st.near_query_leaf_triangle_pruned_count),
	        std::to_string(st.near_query_leaf_distance_reused_count),
	        std::to_string(st.near_query_leaf_bound_fallback_count)};
    #pragma omp critical
    {
      summary_query_ms.push_back(profiled_query_ms);
      summary_world_access.push_back(static_cast<double>(st.world_access_count));
      summary_anchor_distance.push_back(static_cast<double>(st.anchor_distance_count));
      summary_center_distance.push_back(static_cast<double>(st.center_distance_count));
      summary_raw_candidates.push_back(static_cast<double>(st.raw_candidate_count));
      summary_result_count.push_back(static_cast<double>(st.result_count));
      summary_local_router_invoked.push_back(
          static_cast<double>(st.local_router_invoked_count));
      summary_local_router_shortlisted.push_back(
          static_cast<double>(st.local_router_shortlist_child_count));
      summary_router_hint_invoked.push_back(
          static_cast<double>(st.router_hint_invoked_count));
      summary_router_hint_hits.push_back(
          static_cast<double>(st.router_candidate_hit_count));
      summary_path_reuse_hits.push_back(
          static_cast<double>(st.path_reuse_hit_count));
      summary_anchor_cache_hits.push_back(
          static_cast<double>(st.anchor_cache_hit_count));
    }
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
  if (!summary_query_ms.empty()) {
    std::cerr << "Benchmark query summary:"
              << " mean_query_ms=" << format_double(average_values(summary_query_ms))
              << " p50_query_ms="
              << format_double(navigamer::nearest_rank_percentile(summary_query_ms, 0.50))
              << " p95_query_ms="
              << format_double(navigamer::nearest_rank_percentile(summary_query_ms, 0.95))
              << " mean_world_access="
              << format_double(average_values(summary_world_access))
              << " mean_anchor_distance_calls="
              << format_double(average_values(summary_anchor_distance))
              << " mean_center_distance_calls="
              << format_double(average_values(summary_center_distance))
              << " mean_raw_candidates="
              << format_double(average_values(summary_raw_candidates))
              << " mean_router_hint_invocations="
              << format_double(average_values(summary_router_hint_invoked))
              << " mean_router_hint_hits="
              << format_double(average_values(summary_router_hint_hits))
              << " mean_local_router_invocations="
              << format_double(average_values(summary_local_router_invoked))
              << " mean_local_router_shortlisted_children="
              << format_double(average_values(summary_local_router_shortlisted))
              << " mean_path_reuse_hits="
              << format_double(average_values(summary_path_reuse_hits))
              << " mean_anchor_cache_hits="
              << format_double(average_values(summary_anchor_cache_hits))
              << " mean_result_count="
              << format_double(average_values(summary_result_count)) << "\n";
  }
  std::cerr << "Benchmark rows: " << all_rows.size() << "\n";
}

void run_query_index_batch(const std::string& index_path,
                           const std::string& query_input,
                           int tolerance,
                           const std::string& out_tsv,
                           const std::string& path_trace_tsv,
                           const std::string& mode,
                           const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
  if (mode != "adaptive") {
    throw std::runtime_error(
        "query-index-batch currently supports only --mode adaptive");
  }

  LoadedIndex loaded = load_index(index_path);
  std::cerr << "Loaded index: " << index_path
            << " signature=" << loaded.manifest.signature
            << " sequences=" << loaded.manifest.sequence_count
            << " world_nodes=" << loaded.manifest.world_node_count << "\n";

  auto queries = load_reads(query_input, "ref");
  if (queries.empty()) {
    std::cerr << "No query reads loaded.\n";
    return;
  }
  std::cerr << "Queries: " << queries.size() << "\n";

  BioGeometrySearchEngine engine(loaded.builder, search_config);
  const std::vector<std::string> columns = {
      "query_id", "hit_id", "distance", "ref_positions", "read_id", "read_len",
      "ref_id", "strand", "query_start", "reference_start", "aligned_length",
      "score", "edit_distance", "query_fragment", "reference_fragment",
      "bwt_start", "bwt_end", "dist_calcs", "leaf_verify_count",
      "candidate_count_for_prune", "beacon_prune_count", "mbb_filter_mode",
      "mbb_scan_child_checks", "mbb_rect_index_queries",
      "mbb_rect_candidate_children", "mbb_rect_fallback_count",
      "mbb_surviving_child_count", "query_path_class",
      "path_contained_step_count", "path_overlap_step_count",
      "path_uncovered_step_count", "search_qgram_prefilter_enabled",
      "search_prefetch_enabled", "search_qgram_q", "result_count",
      "query_time_ms"};

  std::vector<std::vector<std::string>> all_rows;
  std::vector<SearchStats> per_query_stats(queries.size());
  for (size_t qi = 0; qi < queries.size(); ++qi) {
    const auto& read = queries[qi];
    auto query_start = std::chrono::high_resolution_clock::now();
    auto [res, st] = engine.search_adaptive(*read, tolerance);
    auto query_end = std::chrono::high_resolution_clock::now();
    const double query_time_ms =
        std::chrono::duration<double, std::milli>(query_end - query_start).count();

    const std::vector<std::string> search_stats = {
        mbb_filter_mode_name(search_config.mbb_filter_mode),
        std::to_string(st.mbb_scan_child_checks),
        std::to_string(st.mbb_rect_index_queries),
        std::to_string(st.mbb_rect_candidate_children),
        std::to_string(st.mbb_rect_fallback_count),
        std::to_string(st.mbb_surviving_child_count),
        st.query_path_class(),
        std::to_string(st.path_contained_step_count),
        std::to_string(st.path_overlap_step_count),
        std::to_string(st.path_uncovered_step_count),
        st.search_qgram_prefilter_enabled ? "true" : "false",
        st.search_prefetch_enabled ? "true" : "false",
        std::to_string(st.search_qgram_q),
        std::to_string(st.result_count),
        format_double(query_time_ms)};

    if (res.empty()) {
      std::vector<std::string> row = {
          read->id, "", "", "", read->id,
          std::to_string(static_cast<int>(read->seq.size())), "", "+",
          "0", "0", "0", "0", "", read->seq, "", "-1", "-1",
          std::to_string(st.dist_calc_count),
          std::to_string(st.leaf_verify_count),
          std::to_string(st.candidate_count_for_prune),
          std::to_string(st.beacon_prune_count)};
      row.insert(row.end(), search_stats.begin(), search_stats.end());
      all_rows.push_back(std::move(row));
    } else {
      for (const auto& hit : res) {
        int ed = compute_distance(read->seq, hit->seq);
        auto rows = search_results_to_tsv_rows(read->id, read->seq, 0, *hit, ed);
        for (const auto& r : rows) {
          std::vector<std::string> row = {
              r.query_id, r.hit_id, r.distance_str, r.ref_positions_json,
              r.read_id, r.read_len, r.ref_id, r.strand, r.query_start,
              r.reference_start, r.aligned_length, r.score, r.edit_distance,
              r.query_fragment, r.reference_fragment, r.bwt_start, r.bwt_end,
              std::to_string(st.dist_calc_count),
              std::to_string(st.leaf_verify_count),
              std::to_string(st.candidate_count_for_prune),
              std::to_string(st.beacon_prune_count)};
          row.insert(row.end(), search_stats.begin(), search_stats.end());
          all_rows.push_back(std::move(row));
        }
      }
    }
    per_query_stats[qi] = std::move(st);
  }

  if (!out_tsv.empty()) write_tsv(out_tsv, columns, all_rows);
  if (!path_trace_tsv.empty()) {
    const std::vector<std::string> trace_columns = {
        "query_id", "query_ordinal", "world_visit_count", "leaf_visit_count",
        "world_unique_count", "leaf_unique_count", "prev_world_jaccard",
        "prev_leaf_jaccard", "prev_world_overlap_count",
        "prev_leaf_overlap_count", "world_path", "leaf_path"};
    std::vector<std::vector<std::string>> trace_rows;
    trace_rows.reserve(queries.size());
    for (size_t qi = 0; qi < queries.size(); ++qi) {
      const auto& stats = per_query_stats[qi];
      const bool has_previous = qi > 0;
      const auto world_overlap = previous_overlap_summary(
          stats.world_trace,
          has_previous ? per_query_stats[qi - 1].world_trace
                       : stats.world_trace,
          has_previous);
      const auto leaf_overlap = previous_overlap_summary(
          stats.leaf_trace,
          has_previous ? per_query_stats[qi - 1].leaf_trace
                       : stats.leaf_trace,
          has_previous);
      trace_rows.push_back({
          queries[qi]->id,
          std::to_string(qi),
          std::to_string(stats.world_trace.size()),
          std::to_string(stats.leaf_trace.size()),
          std::to_string(unique_count(stats.world_trace)),
          std::to_string(unique_count(stats.leaf_trace)),
          world_overlap.first,
          leaf_overlap.first,
          world_overlap.second,
          leaf_overlap.second,
          join_id_path(stats.world_trace),
          join_id_path(stats.leaf_trace),
      });
    }
    write_tsv(path_trace_tsv, trace_columns, trace_rows);
  }
  std::cerr << "Batch query rows: " << all_rows.size() << "\n";
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
          if (recovers_source_locus(hit, q)) {
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
          auto [bf_res, bf_st] =
              engine.search_brute_force(q.query, tolerance_edits);
          bool bf_source_found = false;
          for (const auto& hit : bf_res) {
            if (recovers_source_locus(hit, q)) {
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
      "dataset", "query_id", "source_id", "query_length", "L", "r_leaf", "alpha",
      "radius_schedule", "query_time_ms",
      "world_access_count", "node_access_count", "edge_access_count",
      "anchor_distance_count", "bound_check_count",
      "candidate_count", "candidate_verify_count",
      "result_count", "source_recovered", "no_fn"};
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
          bool source_recovered = false;
          for (const auto& hit : results) {
            if (recovers_source_locus(hit, queries[query_idx])) {
              source_recovered = true;
              break;
            }
          }
          double query_time_ms =
              std::chrono::duration<double, std::milli>(end - start).count();

          rows.push_back({
              ref_id,
              std::to_string(query_idx),
              queries[query_idx].source_id,
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
              std::to_string(results.size()),
              source_recovered ? "1" : "0",
              source_recovered ? "1" : "0",
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
  std::string path_trace_out;
  std::string candidates_tsv;
  std::string index_path;
  int tolerance = 2;
	  int demo_size = 500;
	  int window_size = 200;
	  bool window_explicit = false;
	  int stride = 1;
  bool stride_explicit = false;
  int boundary_length = 250;
  size_t queries_per_cell = 200;
  unsigned seed = 42;
  std::string stride_mode = "sparse";
  std::string locator_kind = "refpos";
  std::string error_rates_csv = "0,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20";
  std::string tolerance_rates_csv = "0,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20";
  std::string prefix_lengths_csv;
  std::string primary_radii_csv;
  std::string layer_values_csv = "2,3,4,5";
  std::string r_leaf_values_csv = "4,8,12";
  std::string alpha_values_csv = "0.5,0.7";
  std::string link_mode = "indexed";
  std::string leaf_attach_mode = "indexed";
  std::string leaf_attach_direction = "auto";
  std::string phase2_qgram_postfilter = "off";
  std::string leaf_qgram_postfilter = "off";
  std::string range_candidate_mode = "auto";
  std::string mbb_filter_mode = "scan";
  std::string visited_mode = "epoch";
  std::string graph_view_mode = "flat";
  std::string simd_mode = "auto";
  std::string distance_mode = "myers";
  std::string build_distance_mode = "edlib";
  std::string search_prefetch = "off";
  std::string search_qgram_prefilter = "off";
  bool router_hint_enabled = false;
  int router_hint_qgram_q = 5;
  int router_hint_minimizer_k = 4;
  int router_hint_minimizer_w = 8;
  bool path_reuse_enabled = true;
  bool local_router_enabled = false;
  size_t local_router_max_anchors = 4;
  size_t local_router_max_children = 64;
  std::string local_router_score_mode = "anchor-envelope";
  bool best_first_enabled = false;
  bool safe_child_router_enabled = false;
  size_t safe_child_router_min_fanout = 64;
  size_t safe_child_router_max_candidates = 4096;
  double safe_child_router_max_ratio = 0.5;
  int safe_child_router_min_seed_len = 8;
  std::string safe_child_router_mode = "auto";
  bool safe_child_router_validate = false;
  bool query_planner_enabled = false;
  size_t planner_direct_verify_max_candidates = 32;
  size_t planner_router_min_fanout = 64;
  size_t planner_safe_child_router_min_fanout = 64;
  bool planner_allow_direct_qgram_verify = true;
  bool proximal_oracle_enabled = false;
  std::vector<size_t> proximal_oracle_k_values = {1, 2, 4};
  bool query_profile = false;
  int range_min_seed_length = 8;
  int range_max_seed_length = 20;
  int qgram_q = 5;
  int search_qgram_q = 5;
  size_t auto_pigeonhole_max_candidates = 4096;
  double auto_pigeonhole_max_ratio = 0.25;
  bool auto_hybrid_on_large_candidates = true;
  size_t min_rect_index_fanout = 64;
  size_t phase1_metric_min_fanout = 12;
  size_t phase1_qgram_min_fanout = 12;
  size_t phase1_qgram_max_touched = 250000;
  int progress_interval_seconds = 600;
  int r_sw = navigamer::R_SW;
  int r_mw = navigamer::R_MW;
  int r_lw = navigamer::R_LW;
  int query_edits = -1;
  size_t reference_subset_length = 0;
  int query_length = 200;
  int threads = 1;
  size_t queries_per_class = 1;
  size_t query_count = 256;
  size_t warmup_iterations = 2;
  size_t measured_iterations = 10;
  size_t cold_cache_bytes = 256ULL * 1024ULL * 1024ULL;
  bool query_benchmark_ablations = false;
  std::string locality_profiles_csv = "baseline,path_reuse,optimized";
  std::string locality_datasets_csv =
      "same_template,nearby_windows,random_windows";
  std::string locality_scenarios_csv;
  std::string batch_schedules_csv = "source-oracle";
  std::string query_fastq_out;
  std::string summary_out;
  std::string json_out;
  std::string out_dir;
  std::string candidate_truth_mode = "source";

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ref" && i + 1 < argc) { ref_input = argv[++i]; continue; }
    if (a == "--reads" && i + 1 < argc) { reads_input = argv[++i]; continue; }
    if (a == "--query" && i + 1 < argc) { query_seq = argv[++i]; continue; }
    if (a == "--tolerance" && i + 1 < argc) { tolerance = std::atoi(argv[++i]); continue; }
    if (a == "--mode" && i + 1 < argc) { mode = argv[++i]; continue; }
	    if (a == "--out" && i + 1 < argc) { out_tsv = argv[++i]; continue; }
	    if (a == "--path-trace-out" && i + 1 < argc) {
	      path_trace_out = argv[++i];
	      continue;
	    }
	    if (a == "--candidates" && i + 1 < argc) {
	      candidates_tsv = argv[++i];
	      continue;
	    }
    if (a == "--index" && i + 1 < argc) { index_path = argv[++i]; continue; }
    if (a == "--size" && i + 1 < argc) { demo_size = std::atoi(argv[++i]); continue; }
	    if (a == "--window" && i + 1 < argc) {
	      window_size = std::atoi(argv[++i]);
	      window_explicit = true;
	      continue;
	    }
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
    if (a == "--query-count" && i + 1 < argc) {
      query_count = parse_positive_size(argv[++i], "--query-count");
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
	    if (a == "--out-dir" && i + 1 < argc) {
	      out_dir = argv[++i];
	      continue;
	    }
	    if (a == "--truth" && i + 1 < argc) {
	      candidate_truth_mode = argv[++i];
	      continue;
	    }
    if (a == "--query-benchmark-ablations" && i + 1 < argc) {
      query_benchmark_ablations =
          parse_zero_one(argv[++i], "--query-benchmark-ablations");
      continue;
    }
    if (a == "--proximal-oracle" && i + 1 < argc) {
      proximal_oracle_enabled = parse_zero_one(argv[++i], "--proximal-oracle");
      continue;
    }
    if (a == "--proximal-oracle-k" && i + 1 < argc) {
      proximal_oracle_k_values =
          parse_size_csv(argv[++i], "--proximal-oracle-k");
      continue;
    }
    if (a == "--locality-profiles" && i + 1 < argc) {
      locality_profiles_csv = argv[++i];
      continue;
    }
    if (a == "--locality-datasets" && i + 1 < argc) {
      locality_datasets_csv = argv[++i];
      continue;
    }
    if ((a == "--scenarios" || a == "--scenario") && i + 1 < argc) {
      locality_scenarios_csv = argv[++i];
      continue;
    }
    if (a == "--batch-schedules" && i + 1 < argc) {
      batch_schedules_csv = argv[++i];
      continue;
    }
    if (a == "--query-fastq-out" && i + 1 < argc) {
      query_fastq_out = argv[++i];
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
    if (a == "--prefix-lengths" && i + 1 < argc) { prefix_lengths_csv = argv[++i]; continue; }
    if (a == "--primary-radii" && i + 1 < argc) { primary_radii_csv = argv[++i]; continue; }
    if (a == "--r-sw" && i + 1 < argc) { r_sw = std::atoi(argv[++i]); continue; }
    if (a == "--r-mw" && i + 1 < argc) { r_mw = std::atoi(argv[++i]); continue; }
    if (a == "--r-lw" && i + 1 < argc) { r_lw = std::atoi(argv[++i]); continue; }
    if (a == "--link-mode" && i + 1 < argc) { link_mode = argv[++i]; continue; }
    if (a == "--leaf-attach-mode" && i + 1 < argc) {
      leaf_attach_mode = argv[++i];
      continue;
    }
    if (a == "--leaf-attach-direction" && i + 1 < argc) {
      leaf_attach_direction = argv[++i];
      continue;
    }
    if (a == "--phase2-qgram-postfilter" && i + 1 < argc) {
      phase2_qgram_postfilter = argv[++i];
      continue;
    }
    if (a == "--leaf-qgram-postfilter" && i + 1 < argc) {
      leaf_qgram_postfilter = argv[++i];
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
    if (a == "--distance-mode" && i + 1 < argc) {
      distance_mode = argv[++i];
      continue;
    }
    if (a == "--build-distance-mode" && i + 1 < argc) {
      build_distance_mode = argv[++i];
      continue;
    }
	    if (a == "--phase2-distance-backend") {
	      throw std::runtime_error(
	          "--phase2-distance-backend was removed; Phase2 uses CPU");
	    }
	    if (a == "--search-prefetch" && i + 1 < argc) {
	      search_prefetch = argv[++i];
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
    if (a == "--query-profile" && i + 1 < argc) {
      query_profile = parse_zero_one(argv[++i], "--query-profile");
      continue;
    }
    if (a == "--path-reuse" && i + 1 < argc) {
      path_reuse_enabled = parse_zero_one(argv[++i], "--path-reuse");
      continue;
    }
    if (a == "--router-hints" && i + 1 < argc) {
      router_hint_enabled = parse_zero_one(argv[++i], "--router-hints");
      continue;
    }
    if (a == "--router-hint-qgram-q" && i + 1 < argc) {
      router_hint_qgram_q = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--router-hint-minimizer-k" && i + 1 < argc) {
      router_hint_minimizer_k = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--router-hint-minimizer-w" && i + 1 < argc) {
      router_hint_minimizer_w = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--local-router" && i + 1 < argc) {
      local_router_enabled = parse_zero_one(argv[++i], "--local-router");
      continue;
    }
    if (a == "--local-router-max-anchors" && i + 1 < argc) {
      local_router_max_anchors =
          parse_nonnegative_size(argv[++i], "--local-router-max-anchors");
      continue;
    }
    if (a == "--local-router-max-children" && i + 1 < argc) {
      local_router_max_children =
          parse_nonnegative_size(argv[++i], "--local-router-max-children");
      continue;
    }
    if (a == "--local-router-score" && i + 1 < argc) {
      local_router_score_mode = argv[++i];
      continue;
    }
    if (a == "--best-first" && i + 1 < argc) {
      best_first_enabled = parse_zero_one(argv[++i], "--best-first");
      continue;
    }
    if (a == "--safe-child-router" && i + 1 < argc) {
      safe_child_router_enabled =
          parse_zero_one(argv[++i], "--safe-child-router");
      continue;
    }
    if (a == "--safe-child-router-min-fanout" && i + 1 < argc) {
      safe_child_router_min_fanout =
          parse_nonnegative_size(argv[++i],
                                 "--safe-child-router-min-fanout");
      continue;
    }
    if (a == "--safe-child-router-max-candidates" && i + 1 < argc) {
      safe_child_router_max_candidates =
          parse_nonnegative_size(argv[++i],
                                 "--safe-child-router-max-candidates");
      continue;
    }
    if (a == "--safe-child-router-max-ratio" && i + 1 < argc) {
      safe_child_router_max_ratio = std::stod(argv[++i]);
      continue;
    }
    if (a == "--safe-child-router-min-seed-len" && i + 1 < argc) {
      safe_child_router_min_seed_len = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--safe-child-router-mode" && i + 1 < argc) {
      safe_child_router_mode = argv[++i];
      continue;
    }
    if (a == "--safe-child-router-validate" && i + 1 < argc) {
      safe_child_router_validate =
          parse_zero_one(argv[++i], "--safe-child-router-validate");
      continue;
    }
    if (a == "--query-planner" && i + 1 < argc) {
      query_planner_enabled = parse_zero_one(argv[++i], "--query-planner");
      continue;
    }
    if (a == "--planner-direct-verify-max-candidates" && i + 1 < argc) {
      planner_direct_verify_max_candidates =
          parse_nonnegative_size(argv[++i],
                                 "--planner-direct-verify-max-candidates");
      continue;
    }
    if (a == "--planner-router-min-fanout" && i + 1 < argc) {
      planner_router_min_fanout =
          parse_nonnegative_size(argv[++i], "--planner-router-min-fanout");
      continue;
    }
    if (a == "--planner-safe-child-router-min-fanout" && i + 1 < argc) {
      planner_safe_child_router_min_fanout =
          parse_nonnegative_size(argv[++i],
                                 "--planner-safe-child-router-min-fanout");
      continue;
    }
    if (a == "--planner-allow-direct-qgram-verify" && i + 1 < argc) {
      planner_allow_direct_qgram_verify =
          parse_zero_one(argv[++i], "--planner-allow-direct-qgram-verify");
      continue;
    }
    if (a == "--min-rect-index-fanout" && i + 1 < argc) {
      min_rect_index_fanout =
          parse_positive_size(argv[++i], "--min-rect-index-fanout");
      continue;
    }
    if (a == "--phase1-metric-min-fanout" && i + 1 < argc) {
      phase1_metric_min_fanout =
          parse_positive_size(argv[++i], "--phase1-metric-min-fanout");
      continue;
    }
    if (a == "--phase1-qgram-min-fanout" && i + 1 < argc) {
      phase1_qgram_min_fanout =
          parse_positive_size(argv[++i], "--phase1-qgram-min-fanout");
      continue;
    }
    if (a == "--phase1-qgram-max-touched" && i + 1 < argc) {
      phase1_qgram_max_touched =
          parse_positive_size(argv[++i], "--phase1-qgram-max-touched");
      continue;
    }
    if (a == "--progress-interval-seconds" && i + 1 < argc) {
      const size_t value =
          parse_nonnegative_size(argv[++i], "--progress-interval-seconds");
      if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "--progress-interval-seconds is too large");
      }
      progress_interval_seconds = static_cast<int>(value);
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
    range_config.leaf_attach_direction =
        navigamer::parse_leaf_attach_direction(leaf_attach_direction);
    range_config.phase2_qgram_postfilter =
        parse_on_off(phase2_qgram_postfilter,
                     "--phase2-qgram-postfilter");
    range_config.leaf_qgram_postfilter =
        parse_on_off(leaf_qgram_postfilter, "--leaf-qgram-postfilter");
    range_config.distance_mode =
        navigamer::parse_build_distance_mode(build_distance_mode);
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
    range_config.phase1_metric_min_fanout = phase1_metric_min_fanout;
    range_config.phase1_qgram_min_fanout = phase1_qgram_min_fanout;
    range_config.phase1_qgram_max_touched = phase1_qgram_max_touched;
    range_config.progress_interval_seconds = progress_interval_seconds;
    navigamer::SearchConfig search_config;
    search_config.mbb_filter_mode = navigamer::parse_mbb_filter_mode(mbb_filter_mode);
    search_config.visited_mode = navigamer::parse_visited_mode(visited_mode);
    search_config.graph_view_mode = navigamer::parse_graph_view_mode(graph_view_mode);
    search_config.simd_mode = navigamer::parse_simd_mode(simd_mode);
	    search_config.distance_mode = navigamer::parse_distance_mode(distance_mode);
	    search_config.search_prefetch =
	        parse_on_off(search_prefetch, "--search-prefetch");
	    search_config.search_qgram_prefilter =
	        parse_on_off(search_qgram_prefilter, "--search-qgram-prefilter");
	    search_config.search_qgram_q = search_qgram_q;
	    search_config.trace_paths = !path_trace_out.empty();
    search_config.query_profile = query_profile;
    search_config.path_reuse_enabled = path_reuse_enabled;
    search_config.router_hint_enabled = router_hint_enabled;
    search_config.router_hint_qgram_q = router_hint_qgram_q;
    search_config.router_hint_minimizer_k = router_hint_minimizer_k;
    search_config.router_hint_minimizer_w = router_hint_minimizer_w;
    search_config.local_router_enabled = local_router_enabled;
    search_config.local_router_max_anchors = local_router_max_anchors;
    search_config.local_router_max_children = local_router_max_children;
    search_config.local_router_score_mode = local_router_score_mode;
    search_config.best_first_enabled = best_first_enabled;
    search_config.safe_child_router_enabled = safe_child_router_enabled;
    search_config.safe_child_router_min_fanout = safe_child_router_min_fanout;
    search_config.safe_child_router_max_candidates =
        safe_child_router_max_candidates;
    search_config.safe_child_router_max_ratio = safe_child_router_max_ratio;
    search_config.safe_child_router_min_seed_len =
        safe_child_router_min_seed_len;
    search_config.safe_child_router_mode = safe_child_router_mode;
    search_config.safe_child_router_validate = safe_child_router_validate;
    search_config.query_planner_enabled = query_planner_enabled;
    search_config.planner_direct_verify_max_candidates =
        planner_direct_verify_max_candidates;
    search_config.planner_router_min_fanout = planner_router_min_fanout;
    search_config.planner_safe_child_router_min_fanout =
        planner_safe_child_router_min_fanout;
    search_config.planner_allow_direct_qgram_verify =
        planner_allow_direct_qgram_verify;
    search_config.proximal_oracle_enabled = proximal_oracle_enabled;
    if (search_config.local_router_score_mode != "anchor-envelope") {
      throw std::runtime_error(
          "--local-router-score must currently be anchor-envelope");
    }

    if (cmd == "demo") {
      run_demo(demo_size, hierarchy, range_config, search_config);
      return 0;
    }
    if (cmd == "build") {
      if (ref_input.empty() || reads_input.empty()) {
        std::cerr << "build requires --ref and --reads\n";
        return 1;
      }
      run_build(ref_input, reads_input, hierarchy, range_config, index_path);
      return 0;
    }
    if (cmd == "build-scale") {
      if (ref_input.empty() || out_tsv.empty() || prefix_lengths_csv.empty()) {
        std::cerr << "build-scale requires --ref, --prefix-lengths, and --out\n";
        return 1;
      }
      const auto prefix_lengths =
          parse_size_csv(prefix_lengths_csv, "--prefix-lengths");
      run_build_scale(ref_input, window_size, stride, prefix_lengths,
                      out_tsv, hierarchy, range_config,
                      leaf_attach_direction, index_path);
      return 0;
    }
    if (cmd == "query") {
      if ((reads_input.empty() && index_path.empty()) || query_seq.empty()) {
        std::cerr << "query requires --query and either --reads or --index\n";
        return 1;
      }
      run_query(ref_input, reads_input, query_seq, tolerance, mode, hierarchy,
                range_config, search_config, index_path);
      return 0;
    }
	    if (cmd == "query-index") {
	      if (index_path.empty() || query_seq.empty()) {
	        std::cerr << "query-index requires --index and --query\n";
	        return 1;
	      }
	      run_query(ref_input, "", query_seq, tolerance, mode, hierarchy,
	                range_config, search_config, index_path);
	      return 0;
	    }
	    if (cmd == "query-index-batch") {
	      if (index_path.empty() || reads_input.empty() || out_tsv.empty()) {
	        std::cerr << "query-index-batch requires --index, --reads, and --out\n";
	        return 1;
	      }
	      run_query_index_batch(index_path, reads_input, tolerance, out_tsv,
	                            path_trace_out, mode, search_config);
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
      config.enable_ablation_profiles = query_benchmark_ablations;
      config.proximal_oracle_enabled = proximal_oracle_enabled;
      config.proximal_oracle_k_values = proximal_oracle_k_values;
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
	    if (cmd == "candidate-verify") {
	      if (ref_input.empty() || reads_input.empty() || candidates_tsv.empty() ||
	          out_tsv.empty() || summary_out.empty()) {
	        std::cerr << "candidate-verify requires --ref, --reads, --candidates, "
	                     "--out, and --summary-out\n";
	        return 1;
	      }
	      navigamer::CandidateVerifierConfig config;
	      config.reference_input = ref_input;
	      config.reads_fastq_path = reads_input;
	      config.candidates_tsv_path = candidates_tsv;
	      config.detail_tsv_path = out_tsv;
	      config.summary_tsv_path = summary_out;
	      config.tolerance = tolerance;
	      config.window_length = window_explicit ? window_size : 150;
	      config.stride = stride;
	      config.truth_mode =
	          navigamer::parse_candidate_truth_mode(candidate_truth_mode);
	      const auto summary = navigamer::run_candidate_verifier(config);
	      std::cerr << "candidate-verify done"
	                << " queries=" << summary.query_count
	                << " raw_candidates=" << summary.raw_candidate_count
	                << " verified=" << summary.verified_match_count
	                << " truth=" << summary.truth_match_count
	                << " tp=" << summary.tp_count
	                << " fp=" << summary.fp_count
	                << " fn=" << summary.fn_count
	                << " verify_ms=" << format_double(summary.verify_ms)
	                << " truth_ms=" << format_double(summary.truth_ms)
	                << "\n";
	      return 0;
	    }
	    if (cmd == "locality-benchmark" || cmd == "query-locality-benchmark") {
      if (index_path.empty() || ref_input.empty() || out_tsv.empty()) {
        std::cerr << cmd << " requires --index, --ref, and --out\n";
        return 1;
      }
      navigamer::LocalityBenchmarkConfig config;
      config.index_path = index_path;
      config.ref_input = ref_input;
      config.query_count = query_count;
      config.query_length = query_length;
      config.tolerance = tolerance;
      config.edits = query_edits >= 0 ? query_edits : tolerance;
      config.seed = seed;
      config.profiles =
          parse_string_csv(locality_profiles_csv, "--locality-profiles");
      config.datasets =
          parse_string_csv(locality_datasets_csv, "--locality-datasets");
      if (!locality_scenarios_csv.empty()) {
        config.scenarios =
            parse_string_csv(locality_scenarios_csv, "--scenarios");
      }
      config.batch_schedules =
          parse_string_csv(batch_schedules_csv, "--batch-schedules");
      config.out_tsv_path = out_tsv;
      config.query_fastq_out_path = query_fastq_out;
      auto locality_result =
          navigamer::run_persisted_locality_benchmark(config);
      if (!locality_result.gate_passed) {
        std::cerr << cmd << " gate failed\n";
        return 2;
      }
      std::cerr << cmd << " gate passed"
                << " load_ms=" << format_double(locality_result.load_ms)
                << " rows=" << locality_result.rows.size() << "\n";
      return 0;
    }
    if (cmd == "query-locality-report") {
      if (ref_input.empty() || out_dir.empty()) {
        std::cerr << "query-locality-report requires --ref and --out-dir\n";
        return 1;
      }
      std::filesystem::create_directories(out_dir);
      const std::string summary_path = out_dir + "/summary.tsv";
      const std::string json_path = out_dir + "/summary.json";
      const std::string markdown_path = out_dir + "/report.md";
      std::string report_index_path = index_path;
      if (report_index_path.empty()) {
        report_index_path = out_dir + "/query_locality.navidx";
        auto [ref_id, ref_seq] = navigamer::load_reference(ref_input);
        const size_t actual_prefix =
            reference_subset_length == 0
                ? ref_seq.size()
                : std::min(reference_subset_length, ref_seq.size());
        auto manifest = navigamer::make_reference_window_index_manifest(
            ref_input, actual_prefix, window_size, stride, hierarchy,
            range_config);
        std::string reason;
        if (navigamer::index_matches_manifest(report_index_path, manifest,
                                              nullptr, &reason)) {
          std::cerr << "Reusing report index: " << report_index_path << "\n";
        } else {
          if (std::filesystem::exists(report_index_path)) {
            std::cerr << "Report index not reused: " << reason << "\n";
          }
          std::string index_ref_seq = ref_seq.substr(0, actual_prefix);
          auto windows =
              build_reference_windows(ref_id, index_ref_seq, window_size, stride);
          navigamer::BioGeometryIndexBuilder builder(hierarchy, range_config);
          builder.build(windows);
          navigamer::save_index(report_index_path, builder, manifest);
          std::cerr << "Built report index: " << report_index_path << "\n";
        }
      }

      navigamer::LocalityBenchmarkConfig config;
      config.index_path = report_index_path;
      config.ref_input = ref_input;
      config.query_count = query_count;
      config.query_length = query_length;
      config.tolerance = tolerance;
      config.edits = query_edits >= 0 ? query_edits : tolerance;
      config.seed = seed;
      config.profiles =
          parse_string_csv(locality_profiles_csv, "--locality-profiles");
      config.datasets =
          parse_string_csv(locality_datasets_csv, "--locality-datasets");
      if (!locality_scenarios_csv.empty()) {
        config.scenarios =
            parse_string_csv(locality_scenarios_csv, "--scenarios");
      }
      config.batch_schedules =
          parse_string_csv(batch_schedules_csv, "--batch-schedules");
      config.out_tsv_path = summary_path;
      config.query_fastq_out_path = query_fastq_out;
      auto locality_result =
          navigamer::run_persisted_locality_benchmark(config);
      navigamer::write_locality_report_outputs(
          locality_result, json_path, markdown_path);
      if (!locality_result.gate_passed) {
        std::cerr << "query-locality-report gate failed\n";
        return 2;
      }
      std::cerr << "query-locality-report gate passed"
                << " summary=" << summary_path
                << " json=" << json_path
                << " markdown=" << markdown_path << "\n";
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
