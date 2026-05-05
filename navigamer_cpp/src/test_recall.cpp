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

  navigamer::BioGeometryIndexBuilder builder(cfg.r_sw, cfg.r_mw, cfg.r_lw);
  builder.build(seqs);

  navigamer::BioGeometrySearchEngine engine(builder);

  std::vector<std::shared_ptr<navigamer::BioSequence>> unique_list;
  for (const auto& p : builder.unique_sequences) unique_list.push_back(p.second);

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
    auto [bf_res, bf_st] = engine.search_brute_force(q, cfg.tolerance, unique_list);
    auto [ad_res, ad_st] = engine.search_adaptive(q, cfg.tolerance);

    std::unordered_set<std::string> ad_ids;
    for (const auto& h : ad_res) ad_ids.insert(h->id);

    result.total_bf_hits += bf_res.size();
    result.total_adaptive_hits += ad_res.size();

    size_t missed = 0;
    for (const auto& h : bf_res) {
      if (ad_ids.find(h->id) == ad_ids.end()) {
        missed++;
      }
    }
    result.total_missed_hits += missed;
    if (missed > 0) result.fn_queries++;
  }

  result.passed = (result.total_missed_hits == 0);
  return result;
}

}  // namespace

int main() {
  std::cerr << "=== NavigaMer v8 Recall Test (tolerance < R_SW) ===\n\n";

  std::vector<TestConfig> configs = {
    // Basic small-scale test: tolerance=1 < R_SW=5.
    {5, 15, 30,  1,  100, 20,  50, 1, 42},
    // Moderate tolerance.
    {5, 15, 30,  2,  100, 20,  50, 2, 123},
    // Larger tolerance under the SW radius.
    {5, 15, 30,  3,  200, 20, 100, 3, 456},
    // Boundary case: tolerance is still below R_SW.
    {5, 15, 30,  4,  200, 20, 100, 4, 789},
    // Longer reads.
    {5, 15, 30,  2,  100, 50,  50, 2, 1001},
    // Larger index.
    {5, 15, 30,  2,  500, 20, 200, 2, 2001},
    // Alternate radius configuration.
    {3, 10, 20,  2,  200, 20, 100, 2, 3001},
    // Exact matching.
    {5, 15, 30,  0,  100, 20,  50, 0, 4001},
  };

  int pass_count = 0;
  int fail_count = 0;

  for (size_t i = 0; i < configs.size(); ++i) {
    const auto& c = configs[i];
    std::cerr << "Test " << (i + 1) << "/" << configs.size()
              << ": R_SW=" << c.r_sw << " R_MW=" << c.r_mw << " R_LW=" << c.r_lw
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

  std::cerr << "\n=== Summary: " << pass_count << " passed, "
            << fail_count << " failed ===\n";

  if (fail_count > 0) {
    std::cout << "FAILED\n";
    return 1;
  }
  std::cout << "ALL PASSED\n";
  return 0;
}
