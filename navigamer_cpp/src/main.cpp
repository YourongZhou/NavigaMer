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
#include "sharded_index.hpp"
#include "candidate_verifier.hpp"
#include <array>
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
#include <exception>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <omp.h>

namespace {

void usage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " demo [--size N] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " build --ref <path|seq> --reads <path|seq> [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " build-scale --ref <path|seq> --window 250 --stride 1 --prefix-lengths csv --out <csv> [--index <file>] [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
            << "  " << prog << " build-sharded --ref <path|seq> --window 250 --stride 1 --shard-windows N --shard-build-jobs N --index <manifest> [--primary-radii csv | --r-sw 5 --r-mw 15 --r-lw 30]\n"
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
            << "Global build flags: [--link-mode full|indexed] [--leaf-attach-mode full|indexed] [--leaf-attach-direction auto|seq-to-world|world-to-seq] [--phase2-qgram-postfilter on|off] [--leaf-qgram-postfilter on|off] [--range-candidate-mode auto|pigeonhole|qgram|hybrid|full] [--qgram-q 5] [--auto-pigeonhole-max-candidates 4096] [--auto-pigeonhole-max-ratio 0.25] [--auto-hybrid-on-large-candidates true] [--range-min-seed-length 6] [--range-max-seed-length 20] [--min-rect-index-fanout 64] [--phase1-metric-min-fanout 12] [--phase1-qgram-min-fanout 12] [--phase1-qgram-max-touched 250000] [--progress-interval-seconds 600]\n"
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
    navigamer::LeafId hit_id,
    const navigamer::SequenceStore& sequence_store,
    const BoundaryQuery& query) {
  if (hit_id >= sequence_store.size()) return false;
  if (sequence_store.reference_backed) {
    return query.source_start >= 0 &&
           sequence_store.source_position(hit_id) ==
               static_cast<size_t>(query.source_start);
  }
  const auto& hit = sequence_store.at(hit_id);
  if (hit.id == query.source_id) return true;
  for (const auto& occ : hit.ref_positions) {
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
    for (LeafId h : bf_res) {
      if (std::find(adaptive_res.begin(), adaptive_res.end(), h) !=
          adaptive_res.end())
        a_ok = true;
      if (std::find(exhaustive_res.begin(), exhaustive_res.end(), h) !=
          exhaustive_res.end())
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

  auto reference = load_reference_genome(ref_input);
  const std::string& ref_id = reference.id;
  const std::string& ref_seq = reference.sequence;
  if (ref_seq.empty()) throw std::runtime_error("build-scale reference is empty");

  const std::vector<std::string> columns = {
      "prefix_len",
      "window_count",
      "invalid_window_count",
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
    std::string prefix_seq = ref_seq.substr(0, actual_prefix);
    std::vector<ReferenceContig> prefix_contigs;
    size_t window_count = 0;
    for (const auto& contig : reference.contigs) {
      if (contig.begin >= actual_prefix) break;
      ReferenceContig prefix_contig = contig;
      prefix_contig.end = static_cast<uint32_t>(
          std::min<size_t>(contig.end, actual_prefix));
      prefix_contigs.push_back(std::move(prefix_contig));
      const size_t contig_length =
          prefix_contigs.back().end - prefix_contigs.back().begin;
      if (contig_length >= static_cast<size_t>(window_size)) {
        window_count +=
            1 + (contig_length - static_cast<size_t>(window_size)) /
                    static_cast<size_t>(stride);
      }
    }
    std::cerr << "build-scale: prefix_len=" << actual_prefix
              << " requested=" << requested_prefix
              << " windows=" << window_count
              << " window=" << window_size
              << " stride=" << stride << "\n";

    BioGeometryIndexBuilder builder(config, range_config);
    builder.build_reference_windows(
        ref_id, std::move(prefix_seq),
        static_cast<size_t>(window_size), static_cast<size_t>(stride),
        std::move(prefix_contigs));
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
        std::to_string(window_count),
        std::to_string(stats.invalid_reference_windows),
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

void run_build_sharded(
    const std::string& ref_input,
    int window_size,
    int stride,
    size_t max_shard_windows,
    size_t shard_build_jobs,
    const std::string& index_path,
    const navigamer::HierarchyConfig& hierarchy,
    const navigamer::BuildRangeConfig& range_config) {
  using namespace navigamer;
  if (window_size <= 0 || stride <= 0) {
    throw std::runtime_error(
        "build-sharded window and stride must be positive");
  }
  if (max_shard_windows == 0) {
    throw std::runtime_error(
        "build-sharded requires positive --shard-windows");
  }
  if (index_path.empty()) {
    throw std::runtime_error("build-sharded requires --index");
  }
  const bool reference_is_regular_file =
      std::filesystem::is_regular_file(ref_input);
  size_t reference_bases = 0;
  ShardedIndexManifest manifest;
  const auto print_start = [&](size_t base_count) {
    std::cerr << "Building sharded index: reference_bases="
              << base_count
              << " max_shard_windows=" << max_shard_windows
              << " shard_build_jobs="
              << (shard_build_jobs == 0
                      ? std::string("auto")
                      : std::to_string(shard_build_jobs))
              << " window=" << window_size
              << " stride=" << stride << "\n";
  };
  if (reference_is_regular_file) {
    constexpr size_t kMinCheckpointStride = size_t{4} << 10;
    constexpr size_t kMaxCheckpointStride = size_t{1} << 20;
    const size_t stride_size = static_cast<size_t>(stride);
    const size_t checkpoint_stride =
        max_shard_windows > kMaxCheckpointStride / stride_size
            ? kMaxCheckpointStride
            : std::clamp(
                  max_shard_windows * stride_size,
                  kMinCheckpointStride, kMaxCheckpointStride);
    const auto reference =
        index_reference_genome_file(ref_input, checkpoint_stride);
    reference_bases = reference.sequence_size;
    if (reference_bases == 0) {
      throw std::runtime_error("build-sharded reference is empty");
    }
    print_start(reference_bases);
    manifest = build_sharded_reference_index(
        index_path, ref_input, reference,
        static_cast<size_t>(window_size),
        static_cast<size_t>(stride), max_shard_windows,
        hierarchy, range_config, shard_build_jobs);
  } else {
    auto reference = load_reference_genome(ref_input);
    reference_bases = reference.sequence.size();
    if (reference_bases == 0) {
      throw std::runtime_error("build-sharded reference is empty");
    }
    print_start(reference_bases);
    manifest = build_sharded_reference_index(
        index_path, ref_input, reference.id, reference.sequence,
        reference.contigs, static_cast<size_t>(window_size),
        static_cast<size_t>(stride), max_shard_windows,
        hierarchy, range_config, shard_build_jobs);
  }
  std::cerr << "Sharded index saved: " << index_path
            << " shards=" << manifest.shards.size()
            << " windows=" << manifest.total_window_count
            << " sequences=" << manifest.total_sequence_count
            << " world_nodes=" << manifest.total_world_node_count
            << " router_entries=" << manifest.router_entry_count
            << "\n";
}

void run_query_on_builder(const navigamer::BioGeometryIndexBuilder& builder,
                          const std::string& query_seq, int tolerance,
                          const std::string& mode,
                          const navigamer::SearchConfig& search_config) {
  using namespace navigamer;
  BioGeometrySearchEngine engine(builder, search_config);
  BioSequence q("query", query_seq);
  const auto& sequences = builder.sequence_store();
  const auto print_hit = [&](LeafId sequence_id) {
    std::string display_id;
    if (!sequences.reference_backed) {
      display_id = sequences.at(sequence_id).id;
    } else {
      const size_t source_pos =
          sequences.source_position(sequence_id);
      const auto& contig =
          sequences.contig_for_position(source_pos);
      display_id = contig.id + "_" +
                   std::to_string(
                       static_cast<size_t>(contig.source_begin) +
                       source_pos - contig.begin);
    }
    std::cout << "  " << display_id << " dist="
              << compute_distance(
                     query_seq, sequences.sequence(sequence_id))
              << "\n";
  };

  if (mode == "greedy") {
    auto [res, st] = engine.search_greedy(q, tolerance);
    std::cout << "Greedy hits: " << res.size() << " (dist_calcs=" << st.dist_calc_count << ")\n";
    for (const auto& h : res) print_hit(h);
  } else if (mode == "exhaustive") {
    auto [res, st] = engine.search_exhaustive(q, tolerance);
    std::cout << "Exhaustive hits: " << res.size() << " (dist_calcs=" << st.dist_calc_count << ")\n";
    for (const auto& h : res) print_hit(h);
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
    for (const auto& h : res) print_hit(h);
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
    if (is_sharded_index(index_path)) {
      const auto manifest =
          read_sharded_index_manifest(index_path);
      ShardedSeedRouter shard_router;
      try {
        shard_router = load_sharded_seed_router(
            index_path, manifest);
      } catch (const std::exception& error) {
        std::cerr << "Shard router disabled: " << error.what()
                  << " (searching all shards)\n";
      }
      BioSequence query("query", query_seq);
      const auto route = shard_router.select(query.seq, tolerance);
      struct DisplayHit {
        std::string id;
        std::string sequence;
      };
      std::vector<DisplayHit> hits;
      std::unordered_multimap<size_t, size_t> hit_by_hash;
      size_t distance_calculations = 0;
      double query_time_ms = 0.0;
      constexpr size_t kMaxResidentQueryShards = 64;
      const size_t active_shard_count =
          route.enabled ? route.shard_ids.size()
                        : manifest.shards.size();
      size_t peak_loaded_shards = 0;
      std::vector<uint32_t> shard_group_ids;
      shard_group_ids.reserve(kMaxResidentQueryShards);
      for (size_t shard_begin = 0;
           shard_begin < active_shard_count;
           shard_begin += kMaxResidentQueryShards) {
        const size_t shard_end = std::min(
            shard_begin + kMaxResidentQueryShards,
            active_shard_count);
        shard_group_ids.resize(shard_end - shard_begin);
        if (route.enabled) {
          std::copy(
              route.shard_ids.begin() + shard_begin,
              route.shard_ids.begin() + shard_end,
              shard_group_ids.begin());
        } else {
          std::iota(
              shard_group_ids.begin(), shard_group_ids.end(),
              static_cast<uint32_t>(shard_begin));
        }
        auto shards = load_sharded_index(
            index_path, manifest, shard_group_ids);
        peak_loaded_shards = std::max(
            peak_loaded_shards, shards.size());
        std::vector<std::unique_ptr<BioGeometrySearchEngine>> engines;
        engines.reserve(shards.size());
        for (auto& shard : shards) {
          engines.push_back(
              std::make_unique<BioGeometrySearchEngine>(
                  shard.builder, search_config));
        }
        std::vector<std::pair<SearchResult, SearchStats>>
            shard_results(engines.size());
        std::vector<std::exception_ptr> shard_errors(engines.size());
        const auto query_start =
            std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) if(engines.size() > 1)
        for (size_t active_idx = 0;
             active_idx < engines.size(); ++active_idx) {
          try {
            if (mode == "greedy") {
              shard_results[active_idx] =
                  engines[active_idx]->search_greedy(
                      query, tolerance);
            } else if (mode == "exhaustive") {
              shard_results[active_idx] =
                  engines[active_idx]->search_exhaustive(
                      query, tolerance);
            } else {
              shard_results[active_idx] =
                  engines[active_idx]->search_adaptive(
                      query, tolerance);
            }
          } catch (...) {
            shard_errors[active_idx] =
                std::current_exception();
          }
        }
        for (const auto& error : shard_errors) {
          if (error) std::rethrow_exception(error);
        }
        query_time_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() -
                query_start)
                .count();

        for (size_t active_idx = 0;
             active_idx < shard_results.size(); ++active_idx) {
          distance_calculations +=
              shard_results[active_idx].second.dist_calc_count;
          const auto& store =
              shards[active_idx].builder.sequence_store();
          for (LeafId hit_id : shard_results[active_idx].first) {
            const std::string_view sequence = store.sequence(hit_id);
            const size_t sequence_hash =
                std::hash<std::string_view>{}(sequence);
            const auto hash_range =
                hit_by_hash.equal_range(sequence_hash);
            bool found = false;
            for (auto hit_it = hash_range.first;
                 hit_it != hash_range.second; ++hit_it) {
              if (std::string_view(hits[hit_it->second].sequence) ==
                  sequence) {
                found = true;
                break;
              }
            }
            if (!found) {
              const size_t combined_idx = hits.size();
              hits.push_back(
                  {store.identifier(hit_id), std::string(sequence)});
              hit_by_hash.emplace(sequence_hash, combined_idx);
            }
          }
        }
      }
      std::cerr << "Loaded sharded index: " << index_path
                << " shards=" << active_shard_count
                << "/" << manifest.shards.size()
                << " peak_shards=" << peak_loaded_shards
                << " sequences=" << manifest.total_sequence_count
                << " world_nodes="
                << manifest.total_world_node_count << "\n";
      const char* mode_label =
          mode == "greedy"
              ? "Greedy"
              : mode == "exhaustive"
                    ? "Exhaustive"
                    : "Adaptive";
      std::cout << mode_label << " hits: " << hits.size()
                << " (shards="
                << active_shard_count
                << "/" << manifest.shards.size()
                << " dist_calcs=" << distance_calculations
                << " query_time_ms="
                << format_double(query_time_ms) << ")\n";
      for (const auto& hit : hits) {
        std::cout << "  " << hit.id << " dist="
                  << compute_distance(
                         query_seq, hit.sequence)
                  << "\n";
      }
      return;
    }
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
  const auto& sequence_store = builder.sequence_store();

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
    for (LeafId hit_id : res) {
      const auto& hit = sequence_store.at(hit_id);
      int ed = compute_distance(read->seq, hit.seq);
      auto rows = search_results_to_tsv_rows(read->id, read->seq, 0, hit, ed);
      for (const auto& r : rows) {
        per_read_rows[ri].push_back({
            r.query_id, r.hit_id, r.distance_str, r.ref_positions_json,
            r.read_id, r.read_len, r.ref_id, r.strand, r.query_start, r.reference_start,
            r.aligned_length, r.score, r.edit_distance, r.query_fragment, r.reference_fragment,
            r.bwt_start, r.bwt_end});
      }
    }
  }

  size_t output_row_count = 0;
  if (!out_tsv.empty()) {
    TsvWriter writer(out_tsv, columns);
    for (const auto& rows : per_read_rows) {
      for (const auto& row : rows) writer.write_row(row);
      output_row_count += rows.size();
    }
    writer.close();
  } else {
    for (const auto& rows : per_read_rows) output_row_count += rows.size();
  }
  std::cerr << "Total rows: " << output_row_count << "\n";
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
  const size_t window_count =
      1 + (ref_seq.size() - static_cast<size_t>(window_size)) /
              static_cast<size_t>(stride);
  std::cerr << "Index: " << window_count << " windows from reference\n";

  BioGeometryIndexBuilder builder(config, range_config);
  builder.build_reference_windows(
      ref_id, std::move(ref_seq), static_cast<size_t>(window_size),
      static_cast<size_t>(stride));
  BioGeometrySearchEngine engine(builder, search_config);
  const auto& sequence_store = builder.sequence_store();

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
      for (LeafId hit_id : res) {
        const std::string_view hit_sequence = sequence_store.sequence(hit_id);
        const int ed = compute_distance(read->seq, hit_sequence);
        BioSequence materialized_hit;
        const BioSequence* output_hit = nullptr;
        if (sequence_store.reference_backed) {
          materialized_hit = sequence_store.materialize(hit_id);
          sequence_store.for_each_occurrence(
              hit_id, [&](uint32_t occurrence) {
                const auto& contig =
                    sequence_store.contig_for_position(occurrence);
                const size_t local_start =
                    static_cast<size_t>(contig.source_begin) + occurrence -
                    contig.begin;
                if (local_start >
                    static_cast<size_t>(std::numeric_limits<int>::max()) -
                        hit_sequence.size()) {
                  throw std::runtime_error(
                      "reference occurrence exceeds RefPosition integer range");
                }
                materialized_hit.add_occurrence(
                    contig.id, static_cast<int>(local_start),
                    static_cast<int>(local_start + hit_sequence.size()), "+");
              });
          output_hit = &materialized_hit;
        } else {
          output_hit = &sequence_store.at(hit_id);
        }
        auto rows = search_results_to_tsv_rows(
            read->id, read->seq, 0, *output_hit, ed);
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

  size_t output_row_count = 0;
  if (!out_tsv.empty()) {
    TsvWriter writer(out_tsv, columns);
    for (const auto& rows : per_query_rows) {
      for (const auto& row : rows) writer.write_row(row);
      output_row_count += rows.size();
    }
    writer.close();
  } else {
    for (const auto& rows : per_query_rows) output_row_count += rows.size();
  }
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
  std::cerr << "Benchmark rows: " << output_row_count << "\n";
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

  if (is_sharded_index(index_path)) {
    if (!path_trace_tsv.empty()) {
      throw std::runtime_error(
          "path traces are not defined across sharded node-id spaces");
    }
    const auto shard_manifest =
        read_sharded_index_manifest(index_path);
    ShardedSeedRouter shard_router;
    try {
      shard_router = load_sharded_seed_router(
          index_path, shard_manifest);
    } catch (const std::exception& error) {
      std::cerr << "Shard router disabled: " << error.what()
                << " (searching all shards)\n";
    }

    constexpr size_t kQueryBlockRecords = 8192;
    QuerySequenceReader query_reader(query_input);
    std::vector<QuerySequence> queries;
    queries.reserve(kQueryBlockRecords);
    if (query_reader.read_block(kQueryBlockRecords, &queries) == 0) {
      std::cerr << "No query reads loaded.\n";
      return;
    }

    const std::vector<std::string> columns = {
        "query_id", "hit_id", "distance", "ref_positions", "read_id",
        "read_len", "ref_id", "strand", "query_start",
        "reference_start", "aligned_length", "score", "edit_distance",
        "query_fragment", "reference_fragment", "bwt_start", "bwt_end",
        "dist_calcs", "leaf_verify_count", "candidate_count_for_prune",
        "beacon_prune_count", "mbb_filter_mode",
        "mbb_scan_child_checks", "mbb_rect_index_queries",
        "mbb_rect_candidate_children", "mbb_rect_fallback_count",
        "mbb_surviving_child_count", "query_path_class",
        "path_contained_step_count", "path_overlap_step_count",
        "path_uncovered_step_count", "search_qgram_prefilter_enabled",
        "search_prefetch_enabled", "search_qgram_q", "result_count",
        "query_time_ms"};

    std::unique_ptr<TsvWriter> tsv_writer;
    if (!out_tsv.empty()) {
      tsv_writer = std::make_unique<TsvWriter>(out_tsv, columns);
    }
    size_t output_row_count = 0;
    const auto emit_row = [&](const std::vector<std::string>& row) {
      if (tsv_writer) tsv_writer->write_row(row);
      ++output_row_count;
    };
    const auto accumulate_search_stats = [](
        SearchStats& combined, const SearchStats& stats) {
      combined.search_qgram_prefilter_enabled =
          combined.search_qgram_prefilter_enabled ||
          stats.search_qgram_prefilter_enabled;
      combined.search_prefetch_enabled =
          combined.search_prefetch_enabled ||
          stats.search_prefetch_enabled;
      if (stats.search_qgram_prefilter_enabled) {
        combined.search_qgram_q = stats.search_qgram_q;
      }
      combined.dist_calc_count += stats.dist_calc_count;
      combined.leaf_verify_count += stats.leaf_verify_count;
      combined.candidate_count_for_prune +=
          stats.candidate_count_for_prune;
      combined.beacon_prune_count += stats.beacon_prune_count;
      combined.mbb_scan_child_checks += stats.mbb_scan_child_checks;
      combined.mbb_rect_index_queries += stats.mbb_rect_index_queries;
      combined.mbb_rect_candidate_children +=
          stats.mbb_rect_candidate_children;
      combined.mbb_rect_fallback_count +=
          stats.mbb_rect_fallback_count;
      combined.mbb_surviving_child_count +=
          stats.mbb_surviving_child_count;
      combined.path_contained_step_count +=
          stats.path_contained_step_count;
      combined.path_overlap_step_count += stats.path_overlap_step_count;
      combined.path_uncovered_step_count +=
          stats.path_uncovered_step_count;
    };
    const auto emit_query_hits = [&](const QuerySequence& read,
                                     SearchStats& combined,
                                     const auto& combined_hits,
                                     double query_time_ms) {
      combined.result_count = combined_hits.size();
      const std::vector<std::string> search_stats = {
          mbb_filter_mode_name(search_config.mbb_filter_mode),
          std::to_string(combined.mbb_scan_child_checks),
          std::to_string(combined.mbb_rect_index_queries),
          std::to_string(combined.mbb_rect_candidate_children),
          std::to_string(combined.mbb_rect_fallback_count),
          std::to_string(combined.mbb_surviving_child_count),
          combined.query_path_class(),
          std::to_string(combined.path_contained_step_count),
          std::to_string(combined.path_overlap_step_count),
          std::to_string(combined.path_uncovered_step_count),
          combined.search_qgram_prefilter_enabled ? "true" : "false",
          combined.search_prefetch_enabled ? "true" : "false",
          std::to_string(combined.search_qgram_q),
          std::to_string(combined.result_count),
          format_double(query_time_ms)};

      for (const auto& materialized_hit : combined_hits) {
        const int ed = compute_distance(read.seq, materialized_hit.seq);
        const auto rows = search_results_to_tsv_rows(
            read.id, read.seq, 0, materialized_hit, ed);
        for (const auto& result : rows) {
          std::vector<std::string> row = {
              result.query_id, result.hit_id,
              result.distance_str,
              result.ref_positions_json, result.read_id,
              result.read_len, result.ref_id, result.strand,
              result.query_start, result.reference_start,
              result.aligned_length, result.score,
              result.edit_distance, result.query_fragment,
              result.reference_fragment, result.bwt_start,
              result.bwt_end,
              std::to_string(combined.dist_calc_count),
              std::to_string(combined.leaf_verify_count),
              std::to_string(combined.candidate_count_for_prune),
              std::to_string(combined.beacon_prune_count)};
          row.insert(row.end(), search_stats.begin(), search_stats.end());
          emit_row(row);
        }
      }
      if (combined_hits.empty()) {
        std::vector<std::string> row = {
            read.id, "", "", "", read.id,
            std::to_string(static_cast<int>(read.seq.size())),
            "", "+", "0", "0", "0", "0", "", read.seq,
            "", "-1", "-1",
            std::to_string(combined.dist_calc_count),
            std::to_string(combined.leaf_verify_count),
            std::to_string(combined.candidate_count_for_prune),
            std::to_string(combined.beacon_prune_count)};
        row.insert(row.end(), search_stats.begin(), search_stats.end());
        emit_row(row);
      }
    };
    struct FallbackQueryAggregate {
      SearchStats stats;
      std::vector<BioSequence> hits;
      std::unordered_multimap<size_t, size_t> hit_by_hash;
      double search_ms = 0.0;
    };
    size_t total_query_count = 0;
    size_t total_batch_count = 0;
    size_t routed_queries = 0;
    size_t searched_shards = 0;
    size_t peak_loaded_shards = 0;
    size_t peak_planned_route_ids = 0;
    size_t exact_block_direct_queries = 0;
    size_t exact_block_routed_shards = 0;
    size_t exact_block_matched_shards = 0;
    size_t exact_block_candidate_windows = 0;
    size_t exact_block_distance_calls = 0;
    bool exact_block_reference_attempted = false;
    std::unique_ptr<IndexedReferenceFile> exact_block_reference;
    std::vector<uint32_t> cached_shard_ids;
    std::vector<LoadedIndex> cached_loaded_shards;
    std::vector<std::unique_ptr<BioGeometrySearchEngine>> cached_engines;
    const auto clear_shard_cache = [&]() {
      cached_engines.clear();
      cached_loaded_shards.clear();
      cached_shard_ids.clear();
    };

    do {
    size_t query_block_begin = 0;
    while (query_block_begin < queries.size()) {
    constexpr size_t kMaxPlannedRouteIds = 65536;

    std::vector<size_t> query_route_offsets;
    query_route_offsets.reserve(
        queries.size() - query_block_begin + 1);
    query_route_offsets.push_back(0);
    std::vector<uint32_t> query_route_shard_ids;
    query_route_shard_ids.reserve(kMaxPlannedRouteIds);
    std::vector<uint8_t> query_routed_bits(
        (queries.size() - query_block_begin + 7) / 8,
        uint8_t{0});
    std::vector<float> query_route_ms;
    query_route_ms.reserve(queries.size() - query_block_begin);
    std::vector<std::unique_ptr<ExactBlockVerificationResult>>
        query_direct_results;
    query_direct_results.reserve(queries.size() - query_block_begin);
    std::vector<std::vector<uint32_t>> query_direct_routes;
    query_direct_routes.reserve(queries.size() - query_block_begin);
    size_t query_block_end = query_block_begin;
    size_t planned_direct_route_ids = 0;
    while (query_block_end < queries.size()) {
      const size_t query_idx = query_block_end - query_block_begin;
      const auto& query = queries[query_block_end];
      const auto route_start =
          std::chrono::high_resolution_clock::now();
      const size_t route_id_begin = query_route_shard_ids.size();
      const bool routed = shard_router.append_selected_shards(
          query.seq, tolerance, &query_route_shard_ids);
      std::vector<uint32_t> direct_route;
      constexpr size_t kExactBlockDirectMinShards = 1;
      const size_t route_count =
          query_route_shard_ids.size() - route_id_begin;
      if (routed && route_count >= kExactBlockDirectMinShards) {
        if (!exact_block_reference_attempted) {
          exact_block_reference_attempted = true;
          try {
            const std::string& reference_path =
                shard_manifest.part_manifest.ref_input;
            if (!std::filesystem::is_regular_file(reference_path)) {
              throw std::runtime_error(
                  "persisted reference path is unavailable");
            }
            const std::string fingerprint_prefix =
                "file:" + reference_path + ":";
            const std::string& stored_fingerprint =
                shard_manifest.part_manifest.ref_fingerprint;
            const size_t hash_separator =
                stored_fingerprint.rfind(':');
            if (stored_fingerprint.compare(
                    0, fingerprint_prefix.size(),
                    fingerprint_prefix) != 0 ||
                hash_separator == std::string::npos ||
                hash_separator <= fingerprint_prefix.size()) {
              throw std::runtime_error(
                  "persisted reference fingerprint is incompatible");
            }
            const uint64_t expected_file_size = std::stoull(
                stored_fingerprint.substr(
                    fingerprint_prefix.size(),
                    hash_separator - fingerprint_prefix.size()));
            if (std::filesystem::file_size(reference_path) !=
                    expected_file_size ||
                std::filesystem::last_write_time(reference_path) >
                    std::filesystem::last_write_time(index_path)) {
              throw std::runtime_error(
                  "persisted reference file changed after index build");
            }
            const std::string fai_path = reference_path + ".fai";
            if (!std::filesystem::is_regular_file(fai_path) ||
                std::filesystem::last_write_time(fai_path) <
                    std::filesystem::last_write_time(reference_path)) {
              throw std::runtime_error(
                  "current FASTA .fai is unavailable");
            }
            exact_block_reference =
                std::make_unique<IndexedReferenceFile>(
                    index_reference_genome_file(reference_path));
          } catch (const std::exception& error) {
            std::cerr << "Exact-block direct verification disabled: "
                      << error.what() << "\n";
          }
        }
        if (exact_block_reference) {
          direct_route.assign(
              query_route_shard_ids.begin() + route_id_begin,
              query_route_shard_ids.end());
          planned_direct_route_ids += direct_route.size();
          query_route_shard_ids.resize(route_id_begin);
        }
      }
      query_direct_results.push_back(nullptr);
      query_direct_routes.push_back(std::move(direct_route));
      const auto route_end =
          std::chrono::high_resolution_clock::now();
      query_route_ms.push_back(static_cast<float>(
          std::chrono::duration<double, std::milli>(
              route_end - route_start)
              .count()));
      if (routed) {
        ++routed_queries;
        query_routed_bits[query_idx >> 3] |= static_cast<uint8_t>(
            uint8_t{1} << (query_idx & 7));
      }
      query_route_offsets.push_back(query_route_shard_ids.size());
      ++query_block_end;
      constexpr size_t kMaxPlannedDirectRouteIds = size_t{1} << 20;
      if (query_route_shard_ids.size() >= kMaxPlannedRouteIds ||
          planned_direct_route_ids >= kMaxPlannedDirectRouteIds) {
        break;
      }
    }
    std::vector<size_t> direct_query_indices;
    std::vector<ExactBlockVerificationRequest> direct_requests;
    direct_query_indices.reserve(query_direct_routes.size());
    direct_requests.reserve(query_direct_routes.size());
    for (size_t query_idx = 0;
         query_idx < query_direct_routes.size(); ++query_idx) {
      const auto& route = query_direct_routes[query_idx];
      if (route.empty()) continue;
      direct_query_indices.push_back(query_idx);
      direct_requests.push_back(
          {queries[query_block_begin + query_idx].seq,
           route.data(), route.data() + route.size()});
    }
    if (!direct_requests.empty()) {
      const auto direct_start =
          std::chrono::high_resolution_clock::now();
      auto verified_results =
          verify_selected_shards_by_exact_blocks_batch(
              tolerance, shard_manifest, *exact_block_reference,
              direct_requests);
      const auto direct_end =
          std::chrono::high_resolution_clock::now();
      const float direct_batch_ms = static_cast<float>(
          std::chrono::duration<double, std::milli>(
              direct_end - direct_start)
              .count());
      const bool direct_enabled =
          verified_results.size() == direct_requests.size() &&
          std::all_of(
              verified_results.begin(), verified_results.end(),
              [](const ExactBlockVerificationResult& result) {
                return result.enabled;
              });
      if (direct_enabled) {
        for (size_t request_idx = 0;
             request_idx < verified_results.size(); ++request_idx) {
          const size_t query_idx = direct_query_indices[request_idx];
          auto& verified = verified_results[request_idx];
          ++exact_block_direct_queries;
          exact_block_routed_shards +=
              query_direct_routes[query_idx].size();
          exact_block_matched_shards += verified.matched_shard_count;
          exact_block_candidate_windows +=
              verified.candidate_window_count;
          exact_block_distance_calls += verified.distance_call_count;
          query_route_ms[query_idx] += direct_batch_ms;
          query_direct_results[query_idx] =
              std::make_unique<ExactBlockVerificationResult>(
                  std::move(verified));
        }
      } else {
        std::vector<uint32_t> restored_shard_ids;
        restored_shard_ids.reserve(
            query_route_shard_ids.size() +
            std::accumulate(
                query_direct_routes.begin(), query_direct_routes.end(),
                size_t{0},
                [](size_t total, const std::vector<uint32_t>& route) {
                  return total + route.size();
                }));
        std::vector<size_t> restored_offsets;
        restored_offsets.reserve(query_route_offsets.size());
        restored_offsets.push_back(0);
        for (size_t query_idx = 0;
             query_idx < query_direct_routes.size(); ++query_idx) {
          const auto& direct_route = query_direct_routes[query_idx];
          if (!direct_route.empty()) {
            restored_shard_ids.insert(
                restored_shard_ids.end(), direct_route.begin(),
                direct_route.end());
          } else {
            restored_shard_ids.insert(
                restored_shard_ids.end(),
                query_route_shard_ids.begin() +
                    query_route_offsets[query_idx],
                query_route_shard_ids.begin() +
                    query_route_offsets[query_idx + 1]);
          }
          restored_offsets.push_back(restored_shard_ids.size());
        }
        query_route_shard_ids.swap(restored_shard_ids);
        query_route_offsets.swap(restored_offsets);
        exact_block_reference.reset();
        std::cerr << "Exact-block direct verification disabled: "
                     "reference metadata does not match the bundle\n";
      }
    }
    const size_t planned_query_count =
        query_block_end - query_block_begin;
    peak_planned_route_ids = std::max(
        peak_planned_route_ids, query_route_shard_ids.size());
    const auto query_is_routed = [&](size_t query_idx) {
      return (query_routed_bits[query_idx >> 3] &
              static_cast<uint8_t>(uint8_t{1} << (query_idx & 7))) != 0;
    };

    struct ShardQueryBatch {
      size_t query_begin = 0;
      size_t query_end = 0;
      std::vector<uint32_t> shard_ids;
      bool full_scan = false;
      bool oversized_route_scan = false;
      bool direct_verified = false;

      bool empty() const { return query_begin == query_end; }
    };
    // A large random human-read batch can touch thousands of 10k-window
    // shards. Keep only one bounded union resident while preserving input
    // order and every exact router-selected shard for each query.
    constexpr size_t kMaxResidentQueryShards = 64;
    constexpr size_t kMaxFullScanQueriesPerBatch = 64;
    std::vector<ShardQueryBatch> shard_query_batches;
    ShardQueryBatch batch;
    const auto flush_batch = [&]() {
      if (batch.empty()) return;
      shard_query_batches.push_back(std::move(batch));
      batch = ShardQueryBatch{};
    };
    const auto append_full_scan_query = [&](size_t query_idx) {
      if (!shard_query_batches.empty()) {
        auto& previous = shard_query_batches.back();
        if (previous.full_scan && previous.query_end == query_idx &&
            previous.query_end - previous.query_begin <
                kMaxFullScanQueriesPerBatch) {
          previous.query_end = query_idx + 1;
          return;
        }
      }
      ShardQueryBatch full_scan_batch;
      full_scan_batch.query_begin = query_idx;
      full_scan_batch.query_end = query_idx + 1;
      full_scan_batch.full_scan = true;
      shard_query_batches.push_back(std::move(full_scan_batch));
    };
    std::array<uint32_t, 2 * kMaxResidentQueryShards>
        merged_shard_ids;
    for (size_t query_idx = 0; query_idx < planned_query_count;
         ++query_idx) {
      if (query_direct_results[query_idx]) {
        flush_batch();
        ShardQueryBatch direct_batch;
        direct_batch.query_begin = query_idx;
        direct_batch.query_end = query_idx + 1;
        direct_batch.direct_verified = true;
        shard_query_batches.push_back(std::move(direct_batch));
        continue;
      }
      if (!query_is_routed(query_idx)) {
        flush_batch();
        append_full_scan_query(query_idx);
        continue;
      }
      const size_t route_begin = query_route_offsets[query_idx];
      const size_t route_end = query_route_offsets[query_idx + 1];
      if (route_end - route_begin == shard_manifest.shards.size()) {
        flush_batch();
        append_full_scan_query(query_idx);
        continue;
      }
      if (route_end - route_begin > kMaxResidentQueryShards) {
        flush_batch();
        ShardQueryBatch oversized_batch;
        oversized_batch.query_begin = query_idx;
        oversized_batch.query_end = query_idx + 1;
        oversized_batch.oversized_route_scan = true;
        shard_query_batches.push_back(std::move(oversized_batch));
        continue;
      }
      const auto merged_end = std::set_union(
          batch.shard_ids.begin(), batch.shard_ids.end(),
          query_route_shard_ids.begin() + route_begin,
          query_route_shard_ids.begin() + route_end,
          merged_shard_ids.begin());
      const size_t merged_shard_count = static_cast<size_t>(
          merged_end - merged_shard_ids.begin());
      if (!batch.empty() &&
          merged_shard_count > kMaxResidentQueryShards) {
        flush_batch();
        batch.shard_ids.assign(
            query_route_shard_ids.begin() + route_begin,
            query_route_shard_ids.begin() + route_end);
      } else {
        batch.shard_ids.assign(
            merged_shard_ids.begin(), merged_end);
      }
      if (batch.empty()) batch.query_begin = query_idx;
      batch.query_end = query_idx + 1;
    }
    flush_batch();
    if (shard_query_batches.empty()) {
      throw std::runtime_error("sharded query batch has no work");
    }

    for (const auto& shard_batch : shard_query_batches) {
      if (shard_batch.direct_verified) {
        clear_shard_cache();
        const size_t query_idx = shard_batch.query_begin;
        auto& verified = *query_direct_results[query_idx];
        SearchStats combined;
        combined.dist_calc_count = verified.distance_call_count;
        combined.leaf_verify_count = verified.distance_call_count;
        combined.candidate_count_for_prune =
            verified.candidate_window_count;
        std::vector<BioSequence> combined_hits;
        std::unordered_map<std::string, size_t> hit_by_sequence;
        hit_by_sequence.reserve(verified.occurrences.size());
        for (const auto& occurrence : verified.occurrences) {
          if (occurrence.contig_id >= shard_manifest.contig_ids.size() ||
              occurrence.source_start >
                  static_cast<uint32_t>(
                      std::numeric_limits<int>::max()) -
                      occurrence.sequence.size()) {
            throw std::runtime_error(
                "directly verified occurrence is out of range");
          }
          auto inserted = hit_by_sequence.emplace(
              occurrence.sequence, combined_hits.size());
          if (inserted.second) {
            combined_hits.emplace_back(
                shard_manifest.contig_ids[occurrence.contig_id] + "_" +
                    std::to_string(occurrence.source_start),
                occurrence.sequence);
          }
          combined_hits[inserted.first->second].add_occurrence(
              shard_manifest.contig_ids[occurrence.contig_id],
              static_cast<int>(occurrence.source_start),
              static_cast<int>(occurrence.source_start +
                               occurrence.sequence.size()),
              "+");
        }
        emit_query_hits(
            queries[query_block_begin + query_idx], combined,
            combined_hits, query_route_ms[query_idx]);
        continue;
      }
      if (shard_batch.full_scan ||
          shard_batch.oversized_route_scan) {
        clear_shard_cache();
        const size_t query_count =
            shard_batch.query_end - shard_batch.query_begin;
        const size_t route_begin =
            query_route_offsets[shard_batch.query_begin];
        const size_t selected_shard_count =
            shard_batch.oversized_route_scan
                ? query_route_offsets[shard_batch.query_end] -
                      route_begin
                : shard_manifest.shards.size();
        std::vector<FallbackQueryAggregate> aggregates(query_count);
        std::vector<uint32_t> fallback_shard_ids;
        fallback_shard_ids.reserve(kMaxResidentQueryShards);
        std::vector<std::pair<SearchResult, SearchStats>> shard_results;
        std::vector<std::exception_ptr> shard_errors;

        for (size_t selected_begin = 0;
             selected_begin < selected_shard_count;
             selected_begin += kMaxResidentQueryShards) {
          const size_t selected_end = std::min(
              selected_begin + kMaxResidentQueryShards,
              selected_shard_count);
          fallback_shard_ids.resize(selected_end - selected_begin);
          if (shard_batch.oversized_route_scan) {
            std::copy(
                query_route_shard_ids.begin() + route_begin +
                    selected_begin,
                query_route_shard_ids.begin() + route_begin +
                    selected_end,
                fallback_shard_ids.begin());
          } else {
            std::iota(
                fallback_shard_ids.begin(), fallback_shard_ids.end(),
                static_cast<uint32_t>(selected_begin));
          }
          auto loaded_shards = load_sharded_index(
              index_path, shard_manifest, fallback_shard_ids);
          peak_loaded_shards = std::max(
              peak_loaded_shards, loaded_shards.size());
          std::vector<std::unique_ptr<BioGeometrySearchEngine>> engines;
          engines.reserve(loaded_shards.size());
          for (auto& shard : loaded_shards) {
            engines.push_back(std::make_unique<BioGeometrySearchEngine>(
                shard.builder, search_config));
          }

          shard_results.resize(engines.size());
          shard_errors.resize(engines.size());
          for (size_t query_offset = 0;
               query_offset < query_count; ++query_offset) {
            const auto& read =
                queries[query_block_begin +
                        shard_batch.query_begin + query_offset];
            std::fill(shard_errors.begin(), shard_errors.end(),
                      std::exception_ptr{});
            const auto search_start =
                std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) if(engines.size() > 1)
            for (size_t engine_idx = 0;
                 engine_idx < engines.size(); ++engine_idx) {
              try {
                shard_results[engine_idx] =
                    engines[engine_idx]->search_adaptive(
                        std::string_view(read.seq), tolerance);
              } catch (...) {
                shard_errors[engine_idx] = std::current_exception();
              }
            }
            for (const auto& error : shard_errors) {
              if (error) std::rethrow_exception(error);
            }
            const auto search_end =
                std::chrono::high_resolution_clock::now();
            auto& aggregate = aggregates[query_offset];
            aggregate.search_ms +=
                std::chrono::duration<double, std::milli>(
                    search_end - search_start)
                    .count();
            searched_shards += engines.size();

            for (size_t engine_idx = 0;
                 engine_idx < shard_results.size(); ++engine_idx) {
              const auto& shard_result = shard_results[engine_idx];
              accumulate_search_stats(
                  aggregate.stats, shard_result.second);
              const auto& sequence_store =
                  loaded_shards[engine_idx].builder.sequence_store();
              for (LeafId hit_id : shard_result.first) {
                const std::string_view hit_sequence =
                    sequence_store.sequence(hit_id);
                const size_t hit_hash =
                    std::hash<std::string_view>{}(hit_sequence);
                const auto hash_range =
                    aggregate.hit_by_hash.equal_range(hit_hash);
                size_t combined_idx =
                    std::numeric_limits<size_t>::max();
                for (auto hit_it = hash_range.first;
                     hit_it != hash_range.second; ++hit_it) {
                  if (std::string_view(
                          aggregate.hits[hit_it->second].seq) ==
                      hit_sequence) {
                    combined_idx = hit_it->second;
                    break;
                  }
                }
                if (combined_idx ==
                    std::numeric_limits<size_t>::max()) {
                  combined_idx = aggregate.hits.size();
                  aggregate.hits.push_back(
                      sequence_store.materialize(hit_id));
                  aggregate.hit_by_hash.emplace(
                      hit_hash, combined_idx);
                }
                BioSequence& materialized_hit =
                    aggregate.hits[combined_idx];
                const auto add_occurrence =
                    [&](uint32_t occurrence) {
                      const auto& contig =
                          sequence_store.contig_for_position(
                              occurrence);
                      const size_t local_start =
                          static_cast<size_t>(
                              contig.source_begin) +
                          occurrence - contig.begin;
                      if (local_start >
                          static_cast<size_t>(
                              std::numeric_limits<int>::max()) -
                              hit_sequence.size()) {
                        throw std::runtime_error(
                            "reference occurrence exceeds "
                            "RefPosition integer range");
                      }
                      materialized_hit.add_occurrence(
                          contig.id,
                          static_cast<int>(local_start),
                          static_cast<int>(
                              local_start + hit_sequence.size()),
                          "+");
                    };
                sequence_store.for_each_occurrence(
                    hit_id, add_occurrence);
              }
            }
          }
        }

        for (size_t query_offset = 0;
             query_offset < query_count; ++query_offset) {
          const size_t query_idx =
              shard_batch.query_begin + query_offset;
          auto& aggregate = aggregates[query_offset];
          emit_query_hits(
              queries[query_block_begin + query_idx],
              aggregate.stats, aggregate.hits,
              aggregate.search_ms + query_route_ms[query_idx]);
        }
        continue;
      }
      if (cached_shard_ids != shard_batch.shard_ids) {
        clear_shard_cache();
        cached_loaded_shards = load_sharded_index(
            index_path, shard_manifest, shard_batch.shard_ids);
        cached_shard_ids = shard_batch.shard_ids;
        cached_engines.reserve(cached_loaded_shards.size());
        for (auto& shard : cached_loaded_shards) {
          cached_engines.push_back(
              std::make_unique<BioGeometrySearchEngine>(
                  shard.builder, search_config));
        }
      }
      peak_loaded_shards = std::max(
          peak_loaded_shards, cached_loaded_shards.size());
      std::vector<uint32_t> active_engine_ids;
      std::vector<std::pair<SearchResult, SearchStats>> shard_results;
      std::vector<std::exception_ptr> shard_errors;
      for (size_t query_idx = shard_batch.query_begin;
           query_idx < shard_batch.query_end; ++query_idx) {
      const auto& read = queries[query_block_begin + query_idx];
      auto query_start =
          std::chrono::high_resolution_clock::now();
      const bool route_enabled = query_is_routed(query_idx);
      const size_t route_begin = query_route_offsets[query_idx];
      const size_t route_end = query_route_offsets[query_idx + 1];
      const size_t active_shard_count =
          route_enabled ? route_end - route_begin
                        : cached_engines.size();
      active_engine_ids.resize(active_shard_count);
      if (route_enabled) {
        size_t loaded_idx = 0;
        for (size_t active_idx = 0; active_idx < active_shard_count;
             ++active_idx) {
          const uint32_t shard_id =
              query_route_shard_ids[route_begin + active_idx];
          while (loaded_idx < shard_batch.shard_ids.size() &&
                 shard_batch.shard_ids[loaded_idx] < shard_id) {
            ++loaded_idx;
          }
          if (loaded_idx == shard_batch.shard_ids.size() ||
              shard_batch.shard_ids[loaded_idx] != shard_id) {
            throw std::runtime_error(
                "routed shard was not selected for loading");
          }
          active_engine_ids[active_idx] =
              static_cast<uint32_t>(loaded_idx);
        }
      } else {
        std::iota(active_engine_ids.begin(), active_engine_ids.end(),
                  uint32_t{0});
      }
      searched_shards += active_shard_count;
      shard_results.resize(active_shard_count);
      shard_errors.resize(active_shard_count);
      std::fill(shard_errors.begin(), shard_errors.end(),
                std::exception_ptr{});
#pragma omp parallel for schedule(static) if(active_shard_count > 1)
      for (size_t active_idx = 0;
           active_idx < active_shard_count; ++active_idx) {
        const size_t shard_idx = active_engine_ids[active_idx];
        try {
          shard_results[active_idx] =
              cached_engines[shard_idx]->search_adaptive(
                  std::string_view(read.seq), tolerance);
        } catch (...) {
          shard_errors[active_idx] = std::current_exception();
        }
      }
      for (const auto& error : shard_errors) {
        if (error) std::rethrow_exception(error);
      }
      auto query_end =
          std::chrono::high_resolution_clock::now();
      const double query_time_ms =
          std::chrono::duration<double, std::milli>(
              query_end - query_start)
              .count() + query_route_ms[query_idx];

      SearchStats combined;
      for (const auto& shard_result : shard_results) {
        accumulate_search_stats(combined, shard_result.second);
      }

      size_t raw_hit_count = 0;
      for (const auto& shard_result : shard_results) {
        raw_hit_count += shard_result.first.size();
      }
      std::vector<BioSequence> combined_hits;
      combined_hits.reserve(raw_hit_count);
      const auto append_occurrences = [](
          const auto& sequence_store, LeafId hit_id,
          std::string_view hit_sequence,
          BioSequence* materialized_hit) {
        const auto add_occurrence = [&](uint32_t occurrence) {
          const auto& contig =
              sequence_store.contig_for_position(occurrence);
          const size_t local_start =
              static_cast<size_t>(contig.source_begin) +
              occurrence - contig.begin;
          if (local_start >
              static_cast<size_t>(
                  std::numeric_limits<int>::max()) -
                  hit_sequence.size()) {
            throw std::runtime_error(
                "reference occurrence exceeds RefPosition integer range");
          }
          materialized_hit->add_occurrence(
              contig.id, static_cast<int>(local_start),
              static_cast<int>(local_start + hit_sequence.size()), "+");
        };
        sequence_store.for_each_occurrence(hit_id, add_occurrence);
      };
      if (shard_results.size() == 1) {
        const auto& sequence_store =
            cached_loaded_shards[active_engine_ids.front()]
                .builder.sequence_store();
        for (LeafId hit_id : shard_results.front().first) {
          const std::string_view hit_sequence =
              sequence_store.sequence(hit_id);
          combined_hits.push_back(sequence_store.materialize(hit_id));
          append_occurrences(
              sequence_store, hit_id, hit_sequence,
              &combined_hits.back());
        }
      } else {
        std::unordered_map<std::string_view, size_t> hit_by_sequence;
        hit_by_sequence.reserve(raw_hit_count);
        for (size_t active_idx = 0;
             active_idx < shard_results.size(); ++active_idx) {
          const size_t shard_idx = active_engine_ids[active_idx];
          const auto& sequence_store =
              cached_loaded_shards[shard_idx]
                  .builder.sequence_store();
          for (LeafId hit_id : shard_results[active_idx].first) {
            const std::string_view hit_sequence =
                sequence_store.sequence(hit_id);
            auto inserted = hit_by_sequence.emplace(
                hit_sequence, combined_hits.size());
            if (inserted.second) {
              combined_hits.push_back(
                  sequence_store.materialize(hit_id));
            }
            append_occurrences(
                sequence_store, hit_id, hit_sequence,
                &combined_hits[inserted.first->second]);
          }
        }
      }
      emit_query_hits(read, combined, combined_hits, query_time_ms);
      }
    }
    total_query_count += planned_query_count;
    total_batch_count += shard_query_batches.size();
    query_block_begin = query_block_end;
    }
    } while (query_reader.read_block(
                 kQueryBlockRecords, &queries) != 0);

    std::cerr << "Queries: " << total_query_count << "\n";
    std::cerr << "Loaded sharded index: " << index_path
              << " peak_shards=" << peak_loaded_shards
              << "/" << shard_manifest.shards.size()
              << " peak_route_ids=" << peak_planned_route_ids
              << " batches=" << total_batch_count
              << " sequences="
              << shard_manifest.total_sequence_count
              << " world_nodes="
              << shard_manifest.total_world_node_count << "\n";
    if (tsv_writer) tsv_writer->close();
    std::cerr << "Batch query rows: " << output_row_count
              << " routed_queries=" << routed_queries
              << "/" << total_query_count
              << " searched_shards=" << searched_shards
              << "/" <<
                  total_query_count * shard_manifest.shards.size()
              << " exact_block_direct_queries="
              << exact_block_direct_queries
              << " exact_block_shards="
              << exact_block_matched_shards << "/"
              << exact_block_routed_shards
              << " exact_block_candidate_windows="
              << exact_block_candidate_windows
              << " exact_block_distance_calls="
              << exact_block_distance_calls
              << "\n";
    return;
  }

  LoadedIndex loaded = load_index(index_path);
  std::cerr << "Loaded index: " << index_path
            << " signature=" << loaded.manifest.signature
            << " sequences=" << loaded.manifest.sequence_count
            << " world_nodes=" << loaded.manifest.world_node_count << "\n";

  QuerySequenceReader query_reader(query_input);
  QuerySequence read;
  if (!query_reader.next(&read)) {
    std::cerr << "No query reads loaded.\n";
    return;
  }

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

  std::unique_ptr<TsvWriter> tsv_writer;
  if (!out_tsv.empty()) {
    tsv_writer = std::make_unique<TsvWriter>(out_tsv, columns);
  }
  size_t output_row_count = 0;
  const auto emit_row = [&](const std::vector<std::string>& row) {
    if (tsv_writer) tsv_writer->write_row(row);
    ++output_row_count;
  };
  const std::vector<std::string> trace_columns = {
      "query_id", "query_ordinal", "world_visit_count",
      "leaf_visit_count", "world_unique_count", "leaf_unique_count",
      "prev_world_jaccard", "prev_leaf_jaccard",
      "prev_world_overlap_count", "prev_leaf_overlap_count",
      "world_path", "leaf_path"};
  std::unique_ptr<TsvWriter> trace_writer;
  if (!path_trace_tsv.empty()) {
    trace_writer = std::make_unique<TsvWriter>(
        path_trace_tsv, trace_columns);
  }
  SearchStats previous_stats;
  bool has_previous_stats = false;
  size_t query_count = 0;
  const auto& sequence_store = loaded.builder.sequence_store();
  do {
    auto query_start = std::chrono::high_resolution_clock::now();
    auto [res, st] = engine.search_adaptive(
        std::string_view(read.seq), tolerance);
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
          read.id, "", "", "", read.id,
          std::to_string(static_cast<int>(read.seq.size())), "", "+",
          "0", "0", "0", "0", "", read.seq, "", "-1", "-1",
          std::to_string(st.dist_calc_count),
          std::to_string(st.leaf_verify_count),
          std::to_string(st.candidate_count_for_prune),
          std::to_string(st.beacon_prune_count)};
      row.insert(row.end(), search_stats.begin(), search_stats.end());
      emit_row(row);
    } else {
      for (LeafId hit_id : res) {
        const std::string_view hit_sequence =
            sequence_store.sequence(hit_id);
        int ed = compute_distance(read.seq, hit_sequence);
        BioSequence materialized_hit;
        const BioSequence* output_hit = nullptr;
        if (sequence_store.reference_backed) {
          materialized_hit = sequence_store.materialize(hit_id);
          const auto add_occurrence = [&](uint32_t occurrence) {
            const auto& contig =
                sequence_store.contig_for_position(occurrence);
            const size_t local_start =
                static_cast<size_t>(contig.source_begin) +
                occurrence - contig.begin;
            if (local_start >
                static_cast<size_t>(std::numeric_limits<int>::max()) -
                    hit_sequence.size()) {
              throw std::runtime_error(
                  "reference occurrence exceeds RefPosition integer range");
            }
            materialized_hit.add_occurrence(
                contig.id,
                static_cast<int>(local_start),
                static_cast<int>(local_start + hit_sequence.size()),
                "+");
          };
          sequence_store.for_each_occurrence(hit_id, add_occurrence);
          output_hit = &materialized_hit;
        } else {
          output_hit = &sequence_store.at(hit_id);
        }
        auto rows = search_results_to_tsv_rows(
            read.id, read.seq, 0, *output_hit, ed);
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
          emit_row(row);
        }
      }
    }
    if (trace_writer) {
      const auto world_overlap = previous_overlap_summary(
          st.world_trace,
          has_previous_stats ? previous_stats.world_trace
                             : st.world_trace,
          has_previous_stats);
      const auto leaf_overlap = previous_overlap_summary(
          st.leaf_trace,
          has_previous_stats ? previous_stats.leaf_trace
                             : st.leaf_trace,
          has_previous_stats);
      trace_writer->write_row({
          read.id,
          std::to_string(query_count),
          std::to_string(st.world_trace.size()),
          std::to_string(st.leaf_trace.size()),
          std::to_string(unique_count(st.world_trace)),
          std::to_string(unique_count(st.leaf_trace)),
          world_overlap.first,
          leaf_overlap.first,
          world_overlap.second,
          leaf_overlap.second,
          join_id_path(st.world_trace),
          join_id_path(st.leaf_trace),
      });
      previous_stats = std::move(st);
      has_previous_stats = true;
    }
    ++query_count;
  } while (query_reader.next(&read));

  std::cerr << "Queries: " << query_count << "\n";
  if (tsv_writer) tsv_writer->close();
  if (trace_writer) trace_writer->close();
  std::cerr << "Batch query rows: " << output_row_count << "\n";
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

  std::string normalized_ref =
      normalize_acgt_sequence(ref_seq, "reference");
  if (normalized_ref.size() < static_cast<size_t>(MAP150_READ_LEN)) {
    throw std::runtime_error("map150 reference must be at least 150 bp");
  }
  const size_t window_count =
      normalized_ref.size() - static_cast<size_t>(MAP150_READ_LEN) + 1;
  std::cerr << "map150: windows=" << window_count
            << " reads=" << reads.size()
            << " tolerance=" << tolerance
            << " candidate_tolerance=" << map150_candidate_tolerance(tolerance)
            << " locator=" << locator_kind << "\n";

  BioGeometryIndexBuilder builder(config, range_config);
  if (locator_kind == "refpos") {
    builder.build_reference_windows(
        ref_id, std::move(normalized_ref), MAP150_READ_LEN, 1);
  } else if (locator_kind == "seqan") {
    auto index_seqs = build_map150_reference_windows(ref_id, normalized_ref);
    builder.build(std::move(index_seqs));
  } else {
    throw std::runtime_error("map150 --locator must be refpos or seqan");
  }

  std::unique_ptr<OccurrenceLocator> locator;
  if (locator_kind == "refpos") {
    locator = std::make_unique<RefPositionLocator>();
  } else if (locator_kind == "seqan") {
    locator = make_seqan_fm_locator(ref_id, normalized_ref, builder);
  }

  std::string().swap(ref_seq);
  const std::string_view mapping_reference =
      builder.sequence_store().reference_backed
          ? builder.sequence_store().reference_view()
          : std::string_view(normalized_ref);

  auto results = map150_reads_with_locator(
      ref_id, mapping_reference, reads, tolerance, mode, config, *locator, builder,
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
          if (recovers_source_locus(
                  hit, builder.sequence_store(), q)) {
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
            if (recovers_source_locus(
                    hit, builder.sequence_store(), q)) {
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
            if (recovers_source_locus(
                    hit, builder.sequence_store(), queries[query_idx])) {
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
  bool path_reuse_enabled = false;
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
  int range_min_seed_length = 6;
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
  size_t max_shard_windows = 0;
  size_t shard_build_jobs = 0;
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
    if (a == "--shard-windows" && i + 1 < argc) {
      max_shard_windows =
          parse_positive_size(argv[++i], "--shard-windows");
      continue;
    }
    if (a == "--shard-build-jobs" && i + 1 < argc) {
      shard_build_jobs =
          parse_positive_size(argv[++i], "--shard-build-jobs");
      continue;
    }
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
    if (cmd == "build-sharded") {
      if (ref_input.empty() || index_path.empty() ||
          max_shard_windows == 0) {
        std::cerr << "build-sharded requires --ref, --index, and "
                     "--shard-windows\n";
        return 1;
      }
      run_build_sharded(
          ref_input, window_size, stride, max_shard_windows,
          shard_build_jobs, index_path, hierarchy, range_config);
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
          navigamer::BioGeometryIndexBuilder builder(hierarchy, range_config);
          builder.build_reference_windows(
              ref_id, std::move(index_ref_seq),
              static_cast<size_t>(window_size), static_cast<size_t>(stride));
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
