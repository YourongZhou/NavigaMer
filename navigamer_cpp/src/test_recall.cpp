/**
 * test_recall.cpp
 *
 * Verify that search_adaptive returns every sequence found by brute force
 * when tolerance < R_SW.
 *
 * Strategy:
 *   1. Generate random reference windows as indexed sequences.
 *   2. Mutate indexed sequences to produce queries.
 *   3. Use brute_force as ground truth and check adaptive results.
 *   4. Run multiple tolerances, index sizes, and radius configurations.
 */

#include "structure.hpp"
#include "index_builder.hpp"
#include "search_engine.hpp"
#include "tools.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <string>
#include <vector>

namespace {

std::string random_dna(size_t len, std::mt19937& gen) {
  static const char bases[] = "ATCG";
  std::uniform_int_distribution<> dis(0, 3);
  std::string s;
  s.reserve(len);
  for (size_t i = 0; i < len; ++i) s += bases[dis(gen)];
  return s;
}

std::string mutate(const std::string& seq, int num_mutations, std::mt19937& gen) {
  std::string out = seq;
  if (out.empty()) return out;
  std::uniform_int_distribution<size_t> pos_dis(0, out.size() - 1);
  std::uniform_int_distribution<> base_dis(0, 3);
  static const char bases[] = "ATCG";
  for (int i = 0; i < num_mutations; ++i) {
    size_t pos = pos_dis(gen);
    char orig = out[pos];
    char c;
    do { c = bases[base_dis(gen)]; } while (c == orig);
    out[pos] = c;
  }
  return out;
}

struct TestConfig {
  std::vector<int> primary_radii;
  int r_sw;
  int r_mw;
  int r_lw;
  int tolerance;
  size_t num_seqs;
  size_t seq_len;
  size_t num_queries;
  int query_mutations;
  unsigned seed;
};

std::string config_label(const TestConfig& cfg) {
  if (!cfg.primary_radii.empty()) {
    std::string label = "primary=";
    for (size_t i = 0; i < cfg.primary_radii.size(); ++i) {
      if (i) label += ",";
      label += std::to_string(cfg.primary_radii[i]);
    }
    return label;
  }
  return "R_SW=" + std::to_string(cfg.r_sw) +
         " R_MW=" + std::to_string(cfg.r_mw) +
         " R_LW=" + std::to_string(cfg.r_lw);
}

struct TestResult {
  bool passed;
  size_t total_queries;
  size_t fn_queries;       // queries where adaptive misses at least one BF hit
  size_t total_bf_hits;
  size_t total_adaptive_hits;
  size_t total_missed_hits; // total missed BF hits
};

TestResult run_test(const TestConfig& cfg) {
  std::mt19937 gen(cfg.seed);

  std::string ref = random_dna(cfg.seq_len * cfg.num_seqs, gen);
  std::vector<std::shared_ptr<navigamer::BioSequence>> seqs;
  for (size_t i = 0; i < cfg.num_seqs; ++i) {
    size_t start = i * cfg.seq_len;
    if (start + cfg.seq_len > ref.size()) break;
    std::string frag = ref.substr(start, cfg.seq_len);
    seqs.push_back(std::make_shared<navigamer::BioSequence>(
        "seq_" + std::to_string(i), frag));
  }

  // Add random sequences to increase diversity.
  for (size_t i = 0; i < cfg.num_seqs / 5; ++i) {
    seqs.push_back(std::make_shared<navigamer::BioSequence>(
        "rand_" + std::to_string(i), random_dna(cfg.seq_len, gen)));
  }

  navigamer::BioGeometryIndexBuilder builder =
      cfg.primary_radii.empty()
          ? navigamer::BioGeometryIndexBuilder(cfg.r_sw, cfg.r_mw, cfg.r_lw)
          : navigamer::BioGeometryIndexBuilder(navigamer::HierarchyConfig(cfg.primary_radii));
  builder.build(seqs);

  navigamer::BioGeometrySearchEngine engine(builder);

  // Generate queries by mutating randomly selected indexed sequences.
  std::vector<navigamer::BioSequence> queries;
  std::uniform_int_distribution<size_t> seq_pick(0, seqs.size() - 1);
  for (size_t i = 0; i < cfg.num_queries; ++i) {
    std::string base_seq = seqs[seq_pick(gen)]->seq;
    std::string q = mutate(base_seq, cfg.query_mutations, gen);
    queries.emplace_back("query_" + std::to_string(i), q);
  }
  // Add fully random queries as negative/background cases.
  for (size_t i = 0; i < cfg.num_queries / 5; ++i) {
    queries.emplace_back("qrand_" + std::to_string(i), random_dna(cfg.seq_len, gen));
  }

  TestResult result{};
  result.total_queries = queries.size();

  for (const auto& q : queries) {
    auto [bf_res, bf_st] = engine.search_brute_force(q, cfg.tolerance);
    auto [ad_res, ad_st] = engine.search_adaptive(q, cfg.tolerance);

    std::unordered_set<navigamer::LeafId> ad_ids(
        ad_res.begin(), ad_res.end());

    result.total_bf_hits += bf_res.size();
    result.total_adaptive_hits += ad_res.size();

    size_t missed = 0;
    for (navigamer::LeafId h : bf_res) {
      if (ad_ids.find(h) == ad_ids.end()) {
        missed++;
      }
    }
    result.total_missed_hits += missed;
    if (missed > 0) result.fn_queries++;
  }

  result.passed = (result.total_missed_hits == 0);
  return result;
}

TestResult run_reference_backed_test() {
  constexpr size_t kSequenceLength = 20;
  constexpr int kTolerance = 2;
  constexpr size_t kQueryCount = 100;
  std::mt19937 gen(8128);
  const std::string contig1 = random_dna(380, gen);
  std::string contig2 = random_dna(340, gen);
  contig2.replace(117, 6, "NNNNNN");
  const std::string reference = contig1 + contig2;
  const std::vector<navigamer::ReferenceContig> contigs = {
      {"contig1", 0, static_cast<uint32_t>(contig1.size())},
      {"contig2", static_cast<uint32_t>(contig1.size()),
       static_cast<uint32_t>(reference.size())}};

  navigamer::BioGeometryIndexBuilder builder(
      navigamer::HierarchyConfig({12, 6, 3}));
  builder.build_reference_windows(
      "contig1", reference, kSequenceLength, 1, contigs);
  assert(builder.sequence_store().reference_backed);
  assert(builder.sequence_store().records.empty());
  assert(builder.sequence_store().reference_positions_global_linear ||
         !builder.sequence_store().reference_position_blocks.empty());
  assert(builder.get_statistics().invalid_reference_windows > 0);
  for (size_t sequence_idx = 0;
       sequence_idx < builder.sequence_store().size(); ++sequence_idx) {
    const auto sequence_id =
        static_cast<navigamer::LeafId>(sequence_idx);
    const auto sequence = builder.sequence_store().sequence(sequence_id);
    assert(sequence.find('N') == std::string_view::npos);
    const size_t start =
        builder.sequence_store().source_position(sequence_id);
    const auto& contig =
        builder.sequence_store().contig_for_position(start);
    assert(start + kSequenceLength <= contig.end);
  }

  navigamer::BioGeometrySearchEngine engine(builder);
  std::uniform_int_distribution<size_t> sequence_pick(
      0, builder.sequence_store().size() - 1);
  TestResult result{};
  result.total_queries = kQueryCount;
  for (size_t query_idx = 0; query_idx < kQueryCount; ++query_idx) {
    const auto sequence_id = static_cast<navigamer::LeafId>(
        sequence_pick(gen));
    navigamer::BioSequence query(
        "reference_query_" + std::to_string(query_idx),
        mutate(std::string(builder.sequence_store().sequence(sequence_id)),
               kTolerance, gen));
    const auto [bf_res, bf_stats] =
        engine.search_brute_force(query, kTolerance);
    const auto [adaptive_res, adaptive_stats] =
        engine.search_adaptive(query, kTolerance);
    (void)bf_stats;
    (void)adaptive_stats;

    std::unordered_set<navigamer::LeafId> adaptive_ids(
        adaptive_res.begin(), adaptive_res.end());
    result.total_bf_hits += bf_res.size();
    result.total_adaptive_hits += adaptive_res.size();
    size_t missed = 0;
    for (navigamer::LeafId hit : bf_res) {
      if (!adaptive_ids.count(hit)) missed++;
    }
    result.total_missed_hits += missed;
    if (missed != 0) result.fn_queries++;
  }
  result.passed = result.total_missed_hits == 0;
  return result;
}

}  // namespace

int main() {
  std::cerr << "=== NavigaMer v8 Recall Test (tolerance < R_SW) ===\n\n";

  std::vector<TestConfig> configs = {
    // Basic small-scale test: tolerance=1 < R_SW=5.
    {{}, 5, 15, 30,  1,  100, 20,  50, 1, 42},
    // Moderate tolerance.
    {{}, 5, 15, 30,  2,  100, 20,  50, 2, 123},
    // Larger tolerance under the SW radius.
    {{}, 5, 15, 30,  3,  200, 20, 100, 3, 456},
    // Boundary case: tolerance is still below R_SW.
    {{}, 5, 15, 30,  4,  200, 20, 100, 4, 789},
    // Longer reads.
    {{}, 5, 15, 30,  2,  100, 50,  50, 2, 1001},
    // Larger index.
    {{}, 5, 15, 30,  2,  500, 20, 200, 2, 2001},
    // Alternate radius configuration.
    {{}, 3, 10, 20,  2,  200, 20, 100, 2, 3001},
    // Exact matching.
    {{}, 5, 15, 30,  0,  100, 20,  50, 0, 4001},
    // Two primary layers.
    {{20, 5}, 0, 0, 0,  2,  200, 20, 100, 2, 5001},
    // Four primary layers.
    {{40, 28, 18, 8}, 0, 0, 0,  2,  200, 20, 100, 2, 6001},
    // Five primary layers.
    {{50, 35, 24, 16, 8}, 0, 0, 0,  2,  200, 20, 100, 2, 7001},
  };

  int pass_count = 0;
  int fail_count = 0;

  for (size_t i = 0; i < configs.size(); ++i) {
    const auto& c = configs[i];
    std::cerr << "Test " << (i + 1) << "/" << configs.size()
              << ": " << config_label(c)
              << " tol=" << c.tolerance
              << " seqs=" << c.num_seqs << " len=" << c.seq_len
              << " queries=" << c.num_queries
              << " mutations=" << c.query_mutations
              << " ... ";

    auto result = run_test(c);

    if (result.passed) {
      std::cerr << "PASS";
      pass_count++;
    } else {
      std::cerr << "FAIL";
      fail_count++;
    }
    std::cerr << " (bf_hits=" << result.total_bf_hits
              << " adaptive_hits=" << result.total_adaptive_hits
              << " missed=" << result.total_missed_hits
              << " fn_queries=" << result.fn_queries
              << "/" << result.total_queries << ")\n";
  }

  std::cerr << "Test " << (configs.size() + 1) << "/"
            << (configs.size() + 1)
            << ": multi-contig reference-backed valid windows ... ";
  const auto reference_result = run_reference_backed_test();
  if (reference_result.passed) {
    std::cerr << "PASS";
    pass_count++;
  } else {
    std::cerr << "FAIL";
    fail_count++;
  }
  std::cerr << " (bf_hits=" << reference_result.total_bf_hits
            << " adaptive_hits=" << reference_result.total_adaptive_hits
            << " missed=" << reference_result.total_missed_hits
            << " fn_queries=" << reference_result.fn_queries
            << "/" << reference_result.total_queries << ")\n";

  std::cerr << "\n=== Summary: " << pass_count << " passed, "
            << fail_count << " failed ===\n";

  if (fail_count > 0) {
    std::cout << "FAILED\n";
    return 1;
  }
  std::cout << "ALL PASSED\n";
  return 0;
}
